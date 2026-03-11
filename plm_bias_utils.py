"""
Shared PLM logit-bias utilities for ProteinMPNN / LigandMPNN.

Provides helpers that both ``run.py`` (CLI) and external notebooks can import
to inject PLM-derived logit biases into the MPNN sampling loop.

Inspired by Alamo et al. 2025—see README for details.
"""

import logging
from typing import TYPE_CHECKING

import torch

from data_utils import restype_int_to_str, restype_str_to_int

if TYPE_CHECKING:
    from plminfillers import PLMInfiller

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chain-topology helpers
# ---------------------------------------------------------------------------


def build_chain_info(chain_letters: list[str]) -> dict[int, tuple[int, list[int]]]:
    """Map every global residue index to (local_idx_in_chain, chain_global_indices).

    Parameters
    ----------
    chain_letters : list[str]
        Per-residue chain IDs, length *L* (e.g. ``["A","A","B","B",…]``).

    Returns
    -------
    dict
        ``{global_idx: (local_idx, [global indices for that chain])}``
    """
    chain_info: dict[int, tuple[int, list[int]]] = {}
    seen: set[str] = set()
    unique_chains: list[str] = []
    for ch in chain_letters:
        if ch not in seen:
            unique_chains.append(ch)
            seen.add(ch)
    for chain_letter in unique_chains:
        ci = [i for i, c in enumerate(chain_letters) if c == chain_letter]
        for local_idx, global_idx in enumerate(ci):
            chain_info[global_idx] = (local_idx, ci)
    return chain_info


# ---------------------------------------------------------------------------
# Agnostic (static) PLM bias
# ---------------------------------------------------------------------------

def inject_agnostic_bias(
    feature_dict: dict,
    plm: "PLMInfiller",
    chain_letters: list[str],
    weight: float = 1.0,
    device: torch.device | str = "cpu",
) -> None:
    """Add a **static** PLM logit bias to ``feature_dict["bias"]``.

    One PLM forward pass per chain: all designed positions in that chain are
    masked simultaneously, and the resulting logits are added (scaled by
    *weight*) to the existing bias tensor.

    Parameters
    ----------
    feature_dict : dict
        The MPNN feature dictionary (must already contain ``"bias"``,
        ``"chain_mask"``, ``"S"``).
    plm : PLMInfiller
        Loaded PLM model.
    chain_letters : list[str]
        Per-residue chain IDs (length *L*).
    weight : float
        Scalar multiplier λ for PLM logits.
    device : torch.device | str
        Device for the bias tensor.
    """
    L = feature_dict["X"].shape[1]
    design_mask = feature_dict["chain_mask"][0].cpu().numpy()
    native_ints = feature_dict["S"][0].cpu().numpy()
    native_seq = "".join(restype_int_to_str[aa] for aa in native_ints)
    plm_bias = torch.zeros([1, L, 21], device=device, dtype=torch.float32)

    unique_chains = list(dict.fromkeys(chain_letters))
    for ch in unique_chains:
        ci = [i for i, c in enumerate(chain_letters) if c == ch]
        designed_in_chain = [i for i in ci if design_mask[i] == 1.0]
        if not designed_in_chain:
            continue

        chain_seq = "".join(native_seq[i] for i in ci)
        g_to_l = {g: loc for loc, g in enumerate(ci)}
        local_designed = sorted(g_to_l[g] for g in designed_in_chain)

        log.debug(
            "  [agnostic] chain=%s  seq_len=%d  n_masked=%d",
            ch, len(chain_seq), len(local_designed),
        )
        plm_logits = plm.infill_logits(chain_seq, local_designed)
        for local_pos, pos_logits in zip(local_designed, plm_logits):
            global_idx = ci[local_pos]
            for aa_char, logit_val in pos_logits.items():
                aa_idx = restype_str_to_int.get(aa_char)
                if aa_idx is not None:
                    plm_bias[0, global_idx, aa_idx] = logit_val

    # store raw PLM logits; normalization against MPNN logits happens
    # at sampling time inside model_utils.py
    feature_dict["plm_bias_static"] = plm_bias
    feature_dict["plm_bias_weight"] = weight
    n_biased = int((plm_bias.abs().sum(dim=-1) > 0).sum().item())
    log.debug("  [agnostic] stored %d positions  (λ=%.2f)", n_biased, weight)


# ---------------------------------------------------------------------------
# Aware (per-step) PLM bias callback
# ---------------------------------------------------------------------------

def make_aware_callback(
    plm: "PLMInfiller",
    chain_info: dict[int, tuple[int, list[int]]],
):
    """Return a callback suitable for ``feature_dict["plm_bias_fn"]``.

    The callback is invoked at every designed position during auto-regressive
    decoding.  It reads the *current* partially-sampled sequence ``S``, builds
    the chain context, calls the PLM for the single position being decoded,
    and returns a raw ``[21]`` logit tensor.  Z-score normalization and
    weighting happen in ``model_utils.py``.

    Parameters
    ----------
    plm : PLMInfiller
        Loaded PLM model.
    chain_info : dict
        Output of :func:`build_chain_info`.

    Returns
    -------
    callable
        ``(S_current, S_true_seq, t_indices) → Tensor[21] | None``
    """

    def _plm_bias_fn(S_current, S_true_seq, t_indices):
        t_idx = t_indices[0].item()
        if t_idx not in chain_info:
            return None
        local_idx, ci = chain_info[t_idx]

        B = S_current.shape[0]
        results = torch.zeros(B, 21, device=S_current.device, dtype=torch.float32)

        # Run a separate PLM forward for each sequence in the batch so each
        # gets logits conditioned on *its own* partial decoding context.
        for b in range(B):
            s_np = S_current[b].cpu().numpy()
            s_true_np = S_true_seq[b].cpu().numpy()
            chain_seq_chars = []
            for gi in ci:
                aa_int = s_np[gi]
                if aa_int >= 20:  # X / not yet assigned -> use native
                    chain_seq_chars.append(restype_int_to_str[s_true_np[gi]])
                else:
                    chain_seq_chars.append(restype_int_to_str[aa_int])
            chain_seq = "".join(chain_seq_chars)

            pos_logits = plm.infill_logits(chain_seq, [local_idx])[0]
            for aa_char, logit_val in pos_logits.items():
                aa_idx = restype_str_to_int.get(aa_char)
                if aa_idx is not None:
                    results[b, aa_idx] = logit_val

        log.debug(
            "  [aware] t=%d  local=%d  chain_len=%d  B=%d",
            t_idx, local_idx, len(chain_seq), B,
        )
        # return [B, 21] raw PLM logits; normalization + weighting is done in
        # model_utils.py where MPNN logits are available.
        return results

    return _plm_bias_fn
