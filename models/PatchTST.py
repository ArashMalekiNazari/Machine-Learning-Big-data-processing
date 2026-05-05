
"""
PatchTST — From-scratch PyTorch implementation
Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)

Pipeline (per channel, with shared weights across channels):
  x ∈ (B, M, L)
    → split per channel  (channel-independence)
    → RevIN normalize    (zero mean / unit std per instance)
    → Patching           (sliding window of length P, stride S → N patches)
    → Patch embedding    (P → D, linear, shared across patches AND channels)
    → + Positional enc.  (D × N learnable; tokens are PATCHES, not timesteps)
    → Transformer enc.   (multi-head attention + BatchNorm + FFN, from scratch)
    → Flatten + head     (D·N → T)
    → RevIN inverse
    → concatenate channels back to (B, M, T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 1. RevIN — Reversible Instance Normalization
# ---------------------------------------------------------------------------
# Why: training and test data drift in mean/scale (distribution shift).
# Normalize each (batch, channel) instance to zero-mean / unit-std BEFORE
# patching, then reverse the transform on the model's prediction.
# Operates per-instance, per-channel — no statistics shared across batches.
class RevIN(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            # learnable per-channel affine (paper allows this; harmless if disabled)
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, L)
        # compute stats along time axis only — each (batch, channel) is its own instance
        self.mean = x.mean(dim=-1, keepdim=True).detach()      # (B, M, 1)
        self.std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).detach()
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, T) — predictions
        if self.affine:
            x = (x - self.beta.view(1, -1, 1)) / (self.gamma.view(1, -1, 1) + self.eps)
        x = x * self.std + self.mean
        return x


# ---------------------------------------------------------------------------
# 2. Patching — sliding window with last-value padding
# ---------------------------------------------------------------------------
# Why: turn each univariate series of length L into N tokens of length P.
#   - reduces attention cost from O(L^2) to O(N^2) where N ≈ L / S
#   - each token now has LOCAL SEMANTIC MEANING (a fragment of behavior),
#     unlike a single timestep which is just a scalar.
# Number of patches: N = floor((L - P) / S) + 2
#   The "+2" comes from padding S copies of the LAST value to the end so the
#   final patch always lines up — guarantees the last observed value appears
#   in some patch even when L is not divisible by S.
class Patching(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.P = patch_len
        self.S = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, L)
        # 1) pad S copies of the last value at the end
        last = x[..., -1:].expand(-1, -1, self.S)              # (B, M, S)
        x_padded = torch.cat([x, last], dim=-1)                # (B, M, L+S)
        # 2) slide a window of length P with stride S over the time axis
        # unfold returns shape (B, M, N, P); each row along N is one patch.
        patches = x_padded.unfold(dimension=-1, size=self.P, step=self.S)
        # final shape: (B, M, N, P) — N patches per channel, each of length P
        return patches


# ---------------------------------------------------------------------------
# 3. Patch Embedding — linear projection P → D
# ---------------------------------------------------------------------------
# Why: the Transformer needs D-dimensional tokens. Each patch is mapped to a
# D-vector by a SHARED linear layer (same W_p across patches AND channels).
# The weights are shared because patches are positionally distinguished by
# the positional encoding — sharing forces the projection to learn *content*
# features, not position-conditioned ones.
#
# SUBTLETY 1 — Patch length P appears in TWO places in PatchTST:
#   (a) here: as the input dimension of the patch embedding.
#   (b) in the self-supervised reconstruction head (D → P), reconstructing
#       each masked patch's original P values.
# In SUPERVISED mode (this script) only (a) is used; the forecast head maps
# the flattened encoder output to T directly. In SELF-SUPERVISED mode the
# head shape would be D → P. Don't conflate the two roles.
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, N, P) → (B, M, N, D)
        return self.proj(x)


# ---------------------------------------------------------------------------
# 4. Positional Encoding — LEARNABLE, shape (D, N)
# ---------------------------------------------------------------------------
# SUBTLETY 2 — Why (D, N) and NOT (D, L)?
# After patching, the Transformer's tokens are PATCHES, not raw timesteps.
# There are exactly N patches, so positional information must tag *which
# patch* (1..N), not which timestep (1..L). Encoding (D, L) would be wrong:
# it would have one slot per timestep, but the encoder never sees timesteps
# as tokens — it sees patches. The positional encoding lives in the same
# space the encoder operates in.
#
# Learnable (not sinusoidal) per the paper — let the model decide which
# patch positions matter.
class PositionalEncoding(nn.Module):
    def __init__(self, num_patches: int, d_model: int):
        super().__init__()
        # shape (N, D) for additive broadcasting against (B, M, N, D).
        # Conceptually this IS the (D × N) matrix W_pos from the paper —
        # just stored in row-major (N, D) for convenient broadcasting.
        self.pos = nn.Parameter(torch.randn(num_patches, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, N, D)
        return x + self.pos                                     # broadcasts over B, M


# ---------------------------------------------------------------------------
# 5. Multi-Head Self-Attention — manual implementation
# ---------------------------------------------------------------------------
# Standard scaled dot-product attention, written out so every step is visible.
# No nn.MultiheadAttention, no nn.Transformer — pure linear layers and matmul.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.H = n_heads
        self.d_k = d_model // n_heads

        # one Linear each for Q, K, V — projects D → D, then we reshape to heads.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BM, N, D) — channels already folded into the batch dim by the encoder
        BM, N, D = x.shape

        # 1) project to Q, K, V then split into H heads → (BM, H, N, d_k)
        Q = self.W_q(x).view(BM, N, self.H, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(BM, N, self.H, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(BM, N, self.H, self.d_k).transpose(1, 2)

        # 2) scaled dot-product: (BM, H, N, N) attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # 3) weighted sum of values → (BM, H, N, d_k) → concat heads → (BM, N, D)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(BM, N, D)

        # 4) output projection
        return self.W_o(out)


# ---------------------------------------------------------------------------
# 6. Transformer Encoder Layer — BatchNorm (NOT LayerNorm)
# ---------------------------------------------------------------------------
# The paper uses BatchNorm (per Zerveas 2021 — works better than LayerNorm
# for time-series Transformers). BatchNorm1d normalizes across the batch
# dimension for each feature, which preserves relative scale relationships
# between patches in a series — useful when scale carries information.
#
# Order: x → MHA → residual+BN → FFN → residual+BN
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, attn_dropout)
        # BatchNorm1d over the D feature dim; we transpose around it to feed
        # (BM, D, N) which is BatchNorm1d's expected layout (norm over D).
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _bn(self, bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        # x: (BM, N, D) → bring D to the channel dim for BN1d → back
        return bn(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BM, N, D)
        # --- attention sub-block: residual then BN
        x = self._bn(self.bn1, x + self.dropout(self.attn(x)))
        # --- FFN sub-block: residual then BN
        x = self._bn(self.bn2, x + self.dropout(self.ffn(x)))
        return x


# ---------------------------------------------------------------------------
# 7. Transformer Encoder — stack of n layers
# ---------------------------------------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, attn_dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, attn_dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# 8. PatchTST — full model
# ---------------------------------------------------------------------------
# SUBTLETY 3 — Channel independence vs channel mixing
# All M channels go through the SAME encoder weights but are processed
# INDEPENDENTLY. We achieve this by folding channels into the batch dim:
#   (B, M, N, D) → reshape → (B*M, N, D)
# The encoder never sees information from other channels: each row in the
# (B*M) batch attends only over its own N patches. This is structurally
# enforced — no accidental cross-channel mixing is possible.
class PatchTSTModel(nn.Module):
    def __init__(
        self,
        num_channels: int,        # M
        seq_len: int,             # L
        pred_len: int,            # T
        patch_len: int = 16,      # P
        stride: int = 8,          # S
        d_model: int = 128,       # D
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,          # F
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        revin_affine: bool = True,
    ):
        super().__init__()
        self.M = num_channels
        self.L = seq_len
        self.T = pred_len
        self.P = patch_len
        self.S = stride
        self.D = d_model

        # number of patches: N = floor((L - P) / S) + 2  (with last-value padding of S)
        self.N = (seq_len - patch_len) // stride + 2

        self.revin = RevIN(num_channels, affine=revin_affine)
        self.patching = Patching(patch_len, stride)
        self.embed = PatchEmbedding(patch_len, d_model)
        self.pos_enc = PositionalEncoding(self.N, d_model)
        self.encoder = TransformerEncoder(
            n_layers=n_layers, d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout, attn_dropout=attn_dropout,
        )
        self.head_dropout = nn.Dropout(dropout)
        # Forecast head: flatten (D · N) → T. Shared across channels.
        # Note this is the SUPERVISED head; self-supervised mode would replace
        # this with a per-patch (D → P) layer to reconstruct masked patches.
        self.head = nn.Linear(d_model * self.N, pred_len)

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        # x: (B, M, L)
        B, M, L = x.shape
        assert M == self.M and L == self.L, \
            f"expected (*, {self.M}, {self.L}), got {tuple(x.shape)}"

        # --- 1. RevIN normalize (per-channel, per-instance)
        x = self.revin.normalize(x)                          # (B, M, L)
        if debug: print(f"after RevIN:        {tuple(x.shape)}")

        # --- 2. Patching
        x = self.patching(x)                                 # (B, M, N, P)
        if debug: print(f"after patching:     {tuple(x.shape)}  (N={self.N}, P={self.P})")

        # --- 3. Patch embedding (P → D), shared
        x = self.embed(x)                                    # (B, M, N, D)
        if debug: print(f"after embedding:    {tuple(x.shape)}")

        # --- 4. Positional encoding (over N, the patch axis)
        x = self.pos_enc(x)                                  # (B, M, N, D)
        if debug: print(f"after pos enc:      {tuple(x.shape)}")

        # --- 5. Channel-independence: fold M into the batch dim
        x = x.reshape(B * M, self.N, self.D)                 # (B*M, N, D)
        if debug: print(f"after CI reshape:   {tuple(x.shape)}")

        # --- 6. Transformer encoder (vanilla blocks, BatchNorm, manual MHA)
        x = self.encoder(x)                                  # (B*M, N, D)
        if debug: print(f"after encoder:      {tuple(x.shape)}")

        # --- 7. Flatten + forecast head
        x = x.reshape(B * M, self.N * self.D)                # (B*M, N*D)
        x = self.head_dropout(x)
        x = self.head(x)                                     # (B*M, T)
        x = x.reshape(B, M, self.T)                          # (B, M, T)  ← concat channels
        if debug: print(f"after head:         {tuple(x.shape)}")

        # --- 8. RevIN inverse (restore original scale per channel)
        x = self.revin.denormalize(x)                        # (B, M, T)
        if debug: print(f"after RevIN inv:    {tuple(x.shape)}")

        return x


# ---------------------------------------------------------------------------
# 9. Minimal working example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Realistic ETTh1-style dimensions
    B, M, L, T = 4, 7, 336, 96
    P, S = 16, 8

    model = PatchTSTModel(
        num_channels=M,
        seq_len=L,
        pred_len=T,
        patch_len=P,
        stride=S,
        d_model=128,
        n_heads=16,
        n_layers=3,
        d_ff=256,
        dropout=0.2,
    )

    x = torch.randn(B, M, L)
    print(f"input:              {tuple(x.shape)}")
    y = model(x, debug=True)
    print(f"output:             {tuple(y.shape)}  (expect ({B}, {M}, {T}))")

    # sanity: gradient flows end-to-end
    loss = F.mse_loss(y, torch.randn_like(y))
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params:   {n_params:,}")
    print(f"loss:               {loss.item():.4f}  (gradients OK)")
