"""Advantage-distribution diagnostic for the safe-action Q-value lower bound.

Motivation
----------
The safe-action lower bound clamps Q-targets of *safe* actions up to 0
(see SAFETY_AWARE_IQL.md / DiscreteIQLLossActionSafeLB). When most actions are
safe and the offline data is failure-dominated, the natural targets
``r + gamma * V(s')`` for safe actions are negative almost everywhere, so the
clamp lifts them all to a common 0. This flattens the IQL advantage
``A = Q(s, a) - V(s)`` and the advantage-weighted actor loses its preference
signal -- but only among safe actions that lie on *failure-bound* paths. Safe
actions on *goal-reaching* paths keep their real, positive, shaped targets
(the clamp is a no-op there) and should remain distinguishable.

This module logs the distribution of ``A`` for safe-action transitions, split
by trajectory outcome (goal-reaching vs failure-reaching). If the hypothesis
holds you will see: failure-path safe advantages piled up near a single value
(the flattened floor), while goal-path safe advantages retain spread.

The advantage is computed exactly as torchrl's DiscreteIQLLoss.actor_loss does:
``A = min_a' Q(s, a_taken) - V(s)`` using the *target* Q params and the current
V params, both under ``no_grad``.
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDictBase

from torchrl.objectives.utils import _pseudo_vmap


# Per-episode outcome codes.
OUTCOME_FAILURE = 0
OUTCOME_GOAL = 1
OUTCOME_TIMEOUT = 2
_OUTCOME_NAME = {OUTCOME_FAILURE: "failure", OUTCOME_GOAL: "goal", OUTCOME_TIMEOUT: "timeout"}


def build_episode_outcomes(replay_buffer) -> dict[int, int]:
    """Map each ``episode`` id to its outcome (goal / failure / timeout).

    Outcome is read off the trajectory's terminal transition:
    - ``terminated`` and ``reward > 0``  -> goal
    - ``terminated`` and ``reward <= 0`` -> failure
    - otherwise (no terminated transition in the episode) -> timeout

    Works on the full underlying storage so the labels are stable regardless of
    how minibatches are later sampled. Returns a plain ``{episode_id: code}``
    dict for cheap lookup during training.
    """
    storage = replay_buffer.storage
    data = storage[:]  # full TensorDict view of all stored transitions

    episode = data.get("episode").reshape(-1).cpu()
    reward = data.get(("next", "reward")).reshape(-1).cpu()
    terminated = data.get(("next", "terminated")).reshape(-1).bool().cpu()

    outcomes: dict[int, int] = {}
    for ep, r, term in zip(episode.tolist(), reward.tolist(), terminated.tolist()):
        if term:
            # A terminal transition decides the outcome for the whole episode.
            outcomes[ep] = OUTCOME_GOAL if r > 0 else OUTCOME_FAILURE
        else:
            # Only set timeout if we have not already seen a terminal transition.
            outcomes.setdefault(ep, OUTCOME_TIMEOUT)
    return outcomes


@torch.no_grad()
def compute_advantages(loss_module, tensordict: TensorDictBase) -> torch.Tensor:
    """Compute IQL advantages ``A = min_a' Q(s, a_taken) - V(s)`` for a batch.

    Mirrors torchrl's DiscreteIQLLoss.actor_loss: target Q params for Q, current
    value params for V. Returns a 1-D tensor aligned with ``tensordict`` batch.
    """
    keys = loss_module.tensor_keys

    # --- min over Q ensemble of the chosen (dataset) action's value ---
    td_q = tensordict.select(*loss_module.qvalue_network.in_keys, strict=False)
    td_q = loss_module._vmap_qvalue_networkN0(
        td_q, loss_module.target_qvalue_network_params
    )
    state_action_value = td_q.get(keys.state_action_value)
    action = tensordict.get(keys.action)

    if loss_module.action_space == "categorical":
        if action.ndim < (state_action_value.ndim - (td_q.ndim - tensordict.ndim)):
            action = action.unsqueeze(-1)
        vmap = _pseudo_vmap if loss_module.deactivate_vmap else torch.vmap
        chosen = vmap(
            lambda sav, a: torch.gather(sav, -1, index=a).squeeze(-1),
            (0, None),
        )(state_action_value, action)
    elif loss_module.action_space == "one_hot":
        chosen = (state_action_value * action.to(torch.float)).sum(-1)
    else:
        raise RuntimeError(f"Unknown action space {loss_module.action_space}.")
    min_q = chosen.min(0)[0]

    # --- V(s) under current value params ---
    td_v = tensordict.select(*loss_module.value_network.in_keys, strict=False)
    with loss_module.value_network_params.to_module(loss_module.value_network):
        loss_module.value_network(td_v)
    value = td_v.get(keys.value).squeeze(-1)

    return (min_q - value).reshape(-1)


def _summary(name: str, adv: np.ndarray) -> dict:
    """Distribution summary for one group of advantages."""
    if adv.size == 0:
        return {"group": name, "count": 0}
    return {
        "group": name,
        "count": int(adv.size),
        "mean": float(adv.mean()),
        "std": float(adv.std()),
        "min": float(adv.min()),
        "p10": float(np.percentile(adv, 10)),
        "median": float(np.percentile(adv, 50)),
        "p90": float(np.percentile(adv, 90)),
        "max": float(adv.max()),
        # Fraction within a tight band around 0 -- the "flattened to the floor"
        # signature. High here for failure-path safe actions supports the
        # collapse hypothesis.
        "frac_near_zero": float(np.mean(np.abs(adv) < 1e-3)),
    }


@torch.no_grad()
def diagnose_advantage_distribution(
    loss_module,
    replay_buffer,
    episode_outcomes: dict[int, int],
    num_transitions: int = 20000,
    batch_size: int = 4096,
    histogram: bool = True,
    hist_bins: int = 31,
    hist_range: tuple[float, float] | None = None,
) -> dict:
    """Log the advantage distribution for safe actions, split by outcome.

    Samples ``num_transitions`` transitions from ``replay_buffer``, computes the
    IQL advantage for each, keeps only ``is_action_safe`` transitions, and splits
    them into goal-path vs failure-path groups via ``episode_outcomes``.

    Returns a dict of per-group summaries; also prints a readable report.
    """
    was_training = loss_module.training
    loss_module.eval()
    try:
        collected_adv: list[np.ndarray] = []
        collected_safe: list[np.ndarray] = []
        collected_outcome: list[np.ndarray] = []

        n_seen = 0
        while n_seen < num_transitions:
            take = min(batch_size, num_transitions - n_seen)
            batch = replay_buffer.sample(take)

            adv = compute_advantages(loss_module, batch).cpu().numpy()
            # Vanilla DiscreteIQLLoss has no `is_action_safe` tensor key; fall
            # back to the raw batch key so the same diagnostic works on both arms.
            safe_key = getattr(loss_module.tensor_keys, "is_action_safe", "is_action_safe")
            safe = batch.get(safe_key)
            safe = safe.reshape(-1).bool().cpu().numpy()
            episode = batch.get("episode").reshape(-1).cpu().numpy()
            outcome = np.array(
                [episode_outcomes.get(int(e), OUTCOME_TIMEOUT) for e in episode]
            )

            collected_adv.append(adv)
            collected_safe.append(safe)
            collected_outcome.append(outcome)
            n_seen += len(adv)

        adv = np.concatenate(collected_adv)
        safe = np.concatenate(collected_safe)
        outcome = np.concatenate(collected_outcome)
    finally:
        loss_module.train(was_training)

    safe_adv = adv[safe]
    safe_outcome = outcome[safe]

    groups = {
        "safe / goal-path": safe_adv[safe_outcome == OUTCOME_GOAL],
        "safe / failure-path": safe_adv[safe_outcome == OUTCOME_FAILURE],
        "safe / timeout-path": safe_adv[safe_outcome == OUTCOME_TIMEOUT],
        "safe / all": safe_adv,
        "unsafe / all": adv[~safe],
    }
    summaries = {name: _summary(name, a) for name, a in groups.items()}

    # ----- readable report -----
    print("\n=== Advantage distribution diagnostic (A = min_a' Q(s,a_taken) - V(s)) ===")
    print(f"sampled transitions: {adv.size}  |  safe: {int(safe.sum())}  "
          f"unsafe: {int((~safe).sum())}")
    header = f"{'group':<22}{'n':>7}{'mean':>9}{'std':>9}{'p10':>9}{'med':>9}{'p90':>9}{'~0%':>8}"
    print(header)
    print("-" * len(header))
    for name in groups:
        s = summaries[name]
        if s["count"] == 0:
            print(f"{name:<22}{0:>7}{'--':>9}{'--':>9}{'--':>9}{'--':>9}{'--':>9}{'--':>8}")
            continue
        print(f"{name:<22}{s['count']:>7}{s['mean']:>9.4f}{s['std']:>9.4f}"
              f"{s['p10']:>9.4f}{s['median']:>9.4f}{s['p90']:>9.4f}"
              f"{100 * s['frac_near_zero']:>7.1f}%")

    if histogram:
        if hist_range is None:
            # Auto-scale to the safe-action advantage range (robust percentiles)
            # so the histogram resolves the actual ~±0.1 scale instead of [-1, 1].
            if safe_adv.size:
                lo = float(np.percentile(safe_adv, 1))
                hi = float(np.percentile(safe_adv, 99))
                if hi <= lo:
                    lo, hi = lo - 1e-3, hi + 1e-3
                hist_range = (lo, hi)
            else:
                hist_range = (-1.0, 1.0)
        print(f"\nhistograms (auto range [{hist_range[0]:.3f}, {hist_range[1]:.3f}], "
              f"rows: groups, cols: advantage bins):")
        edges = np.linspace(hist_range[0], hist_range[1], hist_bins + 1)
        for name in ("safe / goal-path", "safe / failure-path"):
            a = groups[name]
            if a.size == 0:
                continue
            counts, _ = np.histogram(np.clip(a, *hist_range), bins=edges)
            peak = counts.max() if counts.max() > 0 else 1
            bars = "".join(
                " ▁▂▃▄▅▆▇█"[min(8, int(round(8 * c / peak)))] for c in counts
            )
            print(f"  {name:<22}  {bars}")
            summaries[name]["histogram"] = counts.tolist()
            summaries[name]["histogram_edges"] = edges.tolist()

    # Interpretation hint based on the key contrast.
    g = summaries["safe / goal-path"]
    f = summaries["safe / failure-path"]
    if g["count"] and f["count"]:
        print(
            f"\ncontrast: failure-path safe std={f['std']:.4f} "
            f"(~0%={100*f['frac_near_zero']:.1f}) vs "
            f"goal-path safe std={g['std']:.4f} "
            f"(~0%={100*g['frac_near_zero']:.1f})"
        )
        if f["std"] < 0.5 * g["std"] and f["frac_near_zero"] > 0.3:
            print("  -> consistent with the collapse hypothesis: failure-path safe "
                  "advantages are flattened relative to goal-path ones.")

    return summaries


if __name__ == "__main__":
    # Minimal smoke test on the CSV example data + a freshly-built loss.
    # This does not train; it just checks the plumbing runs end to end.
    from .models import create_actor, create_q_module, create_v_module
    from .load_dataset import read_trajectories, create_replay_buffer
    from .loss import DiscreteIQLLossActionSafeLB

    td = read_trajectories("examples/iql/trajectories_test.csv", action_dim=6)
    # The CSV path stores `safety` but not `is_action_safe`; synthesize a stand-in
    # so the smoke test can exercise the code path.
    if "is_action_safe" not in td.keys():
        td.set("is_action_safe", (td.get("safety").bool()))
    rb = create_replay_buffer(td, num_slices=2, batch_size=64)

    state_dim = td["observation"].shape[-1]
    loss = DiscreteIQLLossActionSafeLB(
        actor_network=create_actor(state_dim, 6),
        qvalue_network=create_q_module(state_dim, 6),
        value_network=create_v_module(state_dim),
        action_space="categorical",
    )
    outcomes = build_episode_outcomes(rb)
    diagnose_advantage_distribution(loss, rb, outcomes, num_transitions=512, batch_size=128)
