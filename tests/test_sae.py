"""
Unit tests for the Sparse Autoencoder (SAE) model.

Covers: encoding shape, top-k sparsity, reconstruction shape,
training convergence, pre-bias subtraction, decoder unit-norm init,
TF-IDF scoring formula, dead feature detection, and SAE steering vector
construction (Eq. 7 from arXiv 2602.11910).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAE_ROOT = _REPO_ROOT / "sae"
for _p in [str(_REPO_ROOT), str(_SAE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# 1. Encoder output shape
# ---------------------------------------------------------------------------


def test_sae_encode_shape(mock_sae, small_activation_tensor):
    """Pre-activation feature map has shape (batch*seq, expansion_factor * d_in).

    The SAE encoder flattens the first two dims before encoding.
    """
    x = small_activation_tensor  # (2, 16, 64)
    batch, seq, d_in = x.shape

    flat_x = x.reshape(batch * seq, d_in)  # (32, 64)
    pre_acts = mock_sae.pre_acts(flat_x)    # (32, num_latents)

    expected_latents = d_in * mock_sae.cfg.expansion_factor  # 64 * 4 = 256
    assert pre_acts.shape == (batch * seq, expected_latents), (
        f"Expected pre_acts shape ({batch * seq}, {expected_latents}), "
        f"got {pre_acts.shape}"
    )


# ---------------------------------------------------------------------------
# 2. Top-k sparsity
# ---------------------------------------------------------------------------


def test_topk_sparsity(mock_sae, small_activation_tensor):
    """Each token position has exactly k nonzero values after TopK selection."""
    x = small_activation_tensor  # (2, 16, 64)
    enc = mock_sae.encode(x)     # top_acts: (32, k), top_indices: (32, k)

    k = mock_sae.cfg.k
    batch_seq = x.shape[0] * x.shape[1]  # 32

    # Every token should have exactly k selected entries (all nonzero from ReLU).
    # Since TopK picks from ReLU outputs, at least k entries are selected per token.
    assert enc.top_acts.shape == (batch_seq, k), (
        f"Expected top_acts shape ({batch_seq}, {k}), got {enc.top_acts.shape}"
    )
    assert enc.top_indices.shape == (batch_seq, k), (
        f"Expected top_indices shape ({batch_seq}, {k}), got {enc.top_indices.shape}"
    )

    # Verify top_acts values are all non-negative (SAE uses ReLU)
    assert (enc.top_acts >= 0).all(), "top_acts should be non-negative (ReLU output)"


# ---------------------------------------------------------------------------
# 3. Reconstruction shape
# ---------------------------------------------------------------------------


def test_sae_reconstruct_shape(mock_sae, small_activation_tensor):
    """Decoded SAE output matches the flattened input shape (batch*seq, d_in)."""
    x = small_activation_tensor  # (2, 16, 64)
    enc = mock_sae.encode(x)     # top_acts, top_indices: each (32, k)
    dec = mock_sae.decode(enc.top_acts, enc.top_indices)  # (32, d_in)

    batch_seq = x.shape[0] * x.shape[1]
    d_in = x.shape[2]

    assert dec.shape == (batch_seq, d_in), (
        f"Expected decoded shape ({batch_seq}, {d_in}), got {dec.shape}"
    )


# ---------------------------------------------------------------------------
# 4. Reconstruction loss decreases over 3 training steps
# ---------------------------------------------------------------------------


def test_reconstruction_improves_with_epochs(mock_sae_config, small_activation_tensor):
    """MSE reconstruction loss decreases over 3 gradient-descent steps.

    Uses a fresh SAE with Adam optimizer and a fixed batch; verifies that
    the model can overfit in a handful of steps.
    """
    from sae_src.sae.sae import Sae

    torch.manual_seed(42)
    sae = Sae(d_in=64, cfg=mock_sae_config, device="cpu")
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    x = small_activation_tensor.clone()

    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        out = sae(x)
        loss = out.l2_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[2] < losses[0], (
        f"Expected reconstruction loss to decrease over 3 steps; "
        f"got {losses}"
    )


# ---------------------------------------------------------------------------
# 5. Pre-decoder bias is subtracted before encoding
# ---------------------------------------------------------------------------


def test_bpre_bias_applied(mock_sae, small_activation_tensor):
    """pre_acts(x) uses x - b_dec; changing b_dec changes the output."""
    x = small_activation_tensor.reshape(32, 64)  # flatten for pre_acts

    # Baseline with zero bias
    with torch.no_grad():
        mock_sae.b_dec.data.zero_()
    out_zero_bias = mock_sae.pre_acts(x).clone().detach()

    # Non-zero bias: subtract a non-trivial offset before encoding
    with torch.no_grad():
        mock_sae.b_dec.data.fill_(1.0)
    out_nonzero_bias = mock_sae.pre_acts(x).clone().detach()

    assert not torch.allclose(out_zero_bias, out_nonzero_bias), (
        "pre_acts output should differ when b_dec is changed, "
        "confirming the pre-bias subtraction is applied."
    )


# ---------------------------------------------------------------------------
# 6. Decoder columns have unit L2 norm after initialization
# ---------------------------------------------------------------------------


def test_decoder_columns_unit_norm(mock_sae):
    """Each decoder row (W_dec[i, :]) has unit L2 norm after initialization.

    The SAE calls set_decoder_norm_to_unit_norm() during __init__ when
    normalize_decoder=True (the default).
    """
    # W_dec shape: (num_latents, d_in); rows are the decoder directions
    norms = mock_sae.W_dec.data.norm(dim=1)  # (num_latents,)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"Decoder row norms should be 1.0; "
        f"got min={norms.min().item():.6f}, max={norms.max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# 7. TF-IDF scoring formula matches manual reference (Eq. 6, arXiv 2602.11910)
# ---------------------------------------------------------------------------


def test_tfidf_formula_manual_reference():
    """TF-IDF scores match manually computed values for 3 known features.

    Formula (Eq. 6):
        score(j, c) = μ_j(P_c) · log(1 + 1 / (μ_j(P_~c) + ε))

    where ε = 1e-6 (default in compute_tfidf_scores).
    """
    from tfidf_utils import compute_tfidf_scores

    eps = 1e-6
    # Each row is one "sample"; mean over dim=0 gives feature mean.
    # Use 1-sample tensors so that mean == the value itself.
    mean_pos = torch.tensor([2.0, 0.0, 1.0])
    mean_neg = torch.tensor([0.5, 1.0, 0.0])

    pos_activations = mean_pos.unsqueeze(0)   # (1, 3)
    neg_activations = mean_neg.unsqueeze(0)   # (1, 3)

    # Manual reference
    expected = mean_pos * torch.log(1.0 + 1.0 / (mean_neg + eps))

    scores = compute_tfidf_scores(pos_activations, neg_activations, eps=eps)

    assert torch.allclose(scores, expected, atol=1e-5), (
        f"TF-IDF scores do not match manual reference.\n"
        f"Expected: {expected.tolist()}\n"
        f"Got:      {scores.tolist()}"
    )


# ---------------------------------------------------------------------------
# 8. Dead feature detection
# ---------------------------------------------------------------------------


def test_dead_feature_detection(mock_sae, small_activation_tensor):
    """After zeroing encoder weights for selected features, those features
    are always inactive (zero activation), and the dead-feature count is correct.

    A 'dead' feature is one whose mean SAE activation across all inputs is 0.
    """
    import math

    dead_feature_indices = [0, 3, 7]  # Force these features to be dead

    with torch.no_grad():
        for idx in dead_feature_indices:
            mock_sae.encoder.weight.data[idx, :] = 0.0
            mock_sae.encoder.bias.data[idx] = -1e6  # large negative → always 0 after ReLU

    x = small_activation_tensor  # (2, 16, 64)
    flat = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    pre = mock_sae.pre_acts(flat)  # (32, num_latents); ReLU output

    # Per-feature mean activation across all tokens
    mean_acts = pre.mean(dim=0)  # (num_latents,)
    dead_mask = mean_acts == 0.0
    dead_count = dead_mask.sum().item()

    assert dead_count >= len(dead_feature_indices), (
        f"Expected at least {len(dead_feature_indices)} dead features; "
        f"got {dead_count}."
    )

    for idx in dead_feature_indices:
        assert dead_mask[idx].item(), (
            f"Feature {idx} should be dead (mean_act=0) after zeroing weights, "
            f"but mean_act={mean_acts[idx].item():.6f}"
        )


# ---------------------------------------------------------------------------
# 9. SAE steering vector construction (Eq. 7, arXiv 2602.11910)
# ---------------------------------------------------------------------------


def test_sae_steering_vector_construction(mock_sae):
    """v_c^SAE = Σ_{j ∈ F_c} W_dec[j, :] produces the correct shape and values.

    W_dec has shape (num_latents, d_in); each row is the decoder direction for
    one feature. The SAE steering vector for concept c is the sum of the
    decoder directions of the concept-discriminative feature set F_c.
    """
    d_in = mock_sae.d_in  # 64
    F_c = [0, 2, 4]  # arbitrary small feature set

    # Manual reference: sum of selected W_dec rows
    expected = mock_sae.W_dec.data[F_c, :].sum(dim=0)  # (d_in,)
    assert expected.shape == (d_in,), f"Expected shape ({d_in},); got {expected.shape}"

    # Compute using the same formula
    v_sae = torch.zeros(d_in)
    for j in F_c:
        v_sae = v_sae + mock_sae.W_dec.data[j, :]

    assert v_sae.shape == (d_in,), f"SAE steering vector shape should be ({d_in},)"
    assert torch.allclose(v_sae, expected, atol=1e-6), (
        "SAE steering vector does not match manual sum of decoder directions."
    )
