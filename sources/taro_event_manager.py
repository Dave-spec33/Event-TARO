from __future__ import annotations

import asyncio
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.utils.ray_utils import auto_await


# ============================================================
# Utility / empirical cost model
# ============================================================


def beta_expected_variance(rewards: list[float]) -> float:
    rewards = np.asarray(rewards, dtype=np.float64)

    s = int(np.sum(rewards > 0.5))
    f = len(rewards) - s

    a = 1.0 + s
    b = 1.0 + f

    return float(
        a * b / ((a + b) * (a + b + 1.0))
    )


class MakespanModel:
    """
    Empirical marginal-makespan model based on:

        makespan_v2_results.csv

    Expected columns:
        background_long_jobs
        candidate_tokens
        delta_makespan_s
    """

    def __init__(self, csv_path: str):
        rows = []

        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "state": float(row["background_long_jobs"]),
                        "length": float(row["candidate_tokens"]),
                        "delta": float(row["delta_makespan_s"]),
                    }
                )

        if not rows:
            raise RuntimeError(
                f"Empty makespan table: {csv_path}"
            )

        self.states = np.asarray(
            sorted({r["state"] for r in rows}),
            dtype=np.float64,
        )

        self.lengths = np.asarray(
            sorted({r["length"] for r in rows}),
            dtype=np.float64,
        )

        self.table: dict[float, dict[float, float]] = {}

        for state in self.states:
            self.table[float(state)] = {
                r["length"]: r["delta"]
                for r in rows
                if r["state"] == float(state)
            }

        # Standalone 8192-token job.
        self.reference_time = self.delta(
            candidate_length=8192.0,
            equivalent_long_jobs=0.0,
        )

    def _at_state(
        self,
        state: float,
        candidate_length: float,
    ) -> float:
        x = np.concatenate(
            ([0.0], self.lengths)
        )

        y = np.asarray(
            [0.0]
            + [
                self.table[float(state)][float(length)]
                for length in self.lengths
            ],
            dtype=np.float64,
        )

        candidate_length = float(
            np.clip(
                candidate_length,
                0.0,
                float(self.lengths[-1]),
            )
        )

        return float(
            np.interp(
                candidate_length,
                x,
                y,
            )
        )

    def delta(
        self,
        candidate_length: float,
        equivalent_long_jobs: float,
    ) -> float:
        costs = np.asarray(
            [
                self._at_state(
                    float(state),
                    candidate_length,
                )
                for state in self.states
            ],
            dtype=np.float64,
        )

        # Current makespan table only contains states up to the
        # largest benchmarked background load.
        #
        # Event-TARO itself may run with max_inflight=16.
        # Until state=12/16 benchmarks are collected, high-load
        # cost prediction is intentionally clipped to the
        # largest measured state rather than extrapolated.
        load = float(
            np.clip(
                equivalent_long_jobs,
                float(self.states[0]),
                float(self.states[-1]),
            )
        )

        return float(
            np.interp(
                load,
                self.states,
                costs,
            )
        )

    @staticmethod
    def equivalent_load(
        active_lengths: list[float],
    ) -> float:
        """
        Convert active predicted lengths to equivalent 8192-token
        long jobs.

        Example:
            [4096, 4096] -> 1.0
        """
        return float(
            sum(
                np.clip(
                    float(length),
                    0.0,
                    8192.0,
                )
                / 8192.0
                for length in active_lengths
            )
        )

    def normalized_marginal_cost(
        self,
        candidate_length: float,
        active_lengths: list[float],
    ) -> float:
        load = self.equivalent_load(
            active_lengths
        )

        cost = self.delta(
            candidate_length,
            load,
        )

        return float(
            cost / max(self.reference_time, 1e-8)
        )


# ============================================================
# Event state
# ============================================================


@dataclass
class PromptState:
    prompt_id: int

    rewards: list[float] = field(
        default_factory=list
    )

    lengths: list[int] = field(
        default_factory=list
    )

    completed_g: int = 0
    inflight_g: int = 0
    dispatched_g: int = 0

    next_rollout_n: int = 0

    # A prompt becomes selected when its first post-pilot
    # rollout is dispatched.
    selected: bool = False

    # Number of currently in-flight post-pilot rollouts used
    # to complete the G2 -> Gmin commitment window.
    commit_inflight_g: int = 0

    # Number of currently in-flight rollouts beyond Gmin.
    #
    # These are reward-unobserved adaptive/speculative jobs.
    adaptive_inflight_g: int = 0

    # Snapshot of utility at exactly G0 pilot completions.
    #
    # This is deliberately separate from live utility.
    #
    # Otherwise a selected prompt at G4/G6 would have lower
    # posterior expected variance merely because it has more
    # observations, and a fresh G2 prompt would receive an
    # artificial activation advantage.
    pilot_utility: float | None = None

    commit_inflight_g: int = 0
    adaptive_inflight_g: int = 0

    @property
    def utility(self) -> float:
        if not self.rewards:
            return 0.0

        return beta_expected_variance(
            self.rewards
        )

    def predicted_length(
        self,
        default_length: float,
    ) -> float:
        if not self.lengths:
            return float(default_length)

        return float(
            np.clip(
                np.mean(self.lengths),
                1.0,
                8192.0,
            )
        )


@dataclass
class ActiveJob:
    dispatch_id: int

    prompt_id: int
    rollout_n: int

    predicted_length: float
    dispatch_time: float


# ============================================================
# Event-driven worker
# ============================================================


class TAROEventAgentLoopWorker(AgentLoopWorker):
    """
    Event-driven TARO-T rollout collector.

    Core execution model:

        launch pilots
            ↓
        FIRST_COMPLETED
            ↓
        reward becomes available
            ↓
        update prompt posterior
            ↓
        compute:

            ΔU_i - λ ΔM_i(A_t)

            ↓
        refill available slot
            ↓
        repeat

    There is NO:

        pilot barrier
        commit barrier
        extra barrier
    """

    # --------------------------------------------------------
    # Config / helpers
    # --------------------------------------------------------

    def _taro_cfg(self):
        return self.config.trainer.get(
            "taro_online",
            {},
        )

    @staticmethod
    def _value_at(
        values: Any,
        index: int,
    ):
        try:
            return values[index]
        except (
            TypeError,
            IndexError,
            KeyError,
        ):
            return values

    def _active_predicted_lengths(
        self,
        active_jobs: dict[
            asyncio.Task,
            ActiveJob,
        ],
    ) -> list[float]:
        """
        Use the real current in-flight request set.

        V2 does not yet estimate remaining tokens.

        Each active request therefore keeps its dispatch-time
        predicted response length until completion.
        """

        return [
            float(job.predicted_length)
            for job in active_jobs.values()
        ]

    @staticmethod
    def _response_tokens(
        output,
    ) -> int:
        return int(
            output.response_mask.sum().item()
        )

    # --------------------------------------------------------
    # Reservation constraint
    # --------------------------------------------------------

    @staticmethod
    def _pilot_utility_upper_bound(
        state: PromptState,
        pilot_g: int,
    ) -> float:
        """
        Optimistic utility upper bound for a prompt whose pilot
        rollouts have not all finished yet.

        We enumerate every possible binary outcome of the
        remaining pilot samples.

        With pilot_g=2:

            no observation -> UB = 0.20
            [1]            -> UB = 0.20
            [0]            -> UB = 0.20
            [1,1]          -> 0.15
            [0,0]          -> 0.15
            [1,0]          -> 0.20

        This is used only for late-pilot protection. It is NOT
        the normal TARO utility estimate.
        """

        observed = list(
            state.rewards[
                : min(
                    len(state.rewards),
                    pilot_g,
                )
            ]
        )

        remaining = max(
            0,
            pilot_g - len(observed),
        )

        if remaining == 0:
            if state.pilot_utility is not None:
                return float(state.pilot_utility)

            return float(
                beta_expected_variance(observed)
            )

        upper_bound = 0.0

        # Binary reward setting.
        for future_successes in range(
            remaining + 1
        ):
            future_failures = (
                remaining
                - future_successes
            )

            hypothetical = (
                observed
                + [1.0] * future_successes
                + [0.0] * future_failures
            )

            upper_bound = max(
                upper_bound,
                beta_expected_variance(
                    hypothetical
                ),
            )

        return float(upper_bound)

    @staticmethod
    def _reservation_snapshot(
        states: dict[int, PromptState],
        *,
        min_selected: int,
        pilot_g: int,
        gmin: int,
    ) -> dict[str, int]:
        """
        Compute rollout budget that must remain unused by
        arbitrary extra dispatches.

        Two components:

        1. debt_rollouts

           Selected prompts that have not yet reached Gmin.

        2. future_activation_rollouts

           If fewer than min_selected prompts have been
           selected, reserve enough extra rollouts to turn
           future G0 pilot groups into Gmin trainable groups.

        Example:

            pilot_g = 2
            gmin = 4
            min_selected = 3

            current selected_count = 1

        Then two additional prompts still need to become
        trainable:

            future_groups = 2

        Each already owns its two pilot rollouts, so each
        requires only:

            4 - 2 = 2

        extra rollouts.

        Therefore:

            future_activation_rollouts = 4
        """

        selected_count = sum(
            int(state.selected)
            for state in states.values()
        )

        debt_rollouts = sum(
            max(
                0,
                gmin - state.dispatched_g,
            )
            for state in states.values()
            if state.selected
        )

        future_groups = max(
            0,
            min_selected - selected_count,
        )

        future_activation_rollouts = (
            future_groups
            * max(
                0,
                gmin - pilot_g,
            )
        )

        return {
            "selected_count":
                int(selected_count),

            "debt_rollouts":
                int(debt_rollouts),

            "future_groups":
                int(future_groups),

            "future_activation_rollouts":
                int(future_activation_rollouts),

            "total_reserved_rollouts":
                int(
                    debt_rollouts
                    + future_activation_rollouts
                ),
        }

    def _budget_feasible_after_dispatch(
        self,
        *,
        state: PromptState,
        states: dict[int, PromptState],
        total_dispatched: int,
        total_budget: int,
        min_selected: int,
        pilot_g: int,
        gmin: int,
    ) -> tuple[
        bool,
        dict[str, int],
    ]:
        """
        Check whether dispatching one rollout to `state`
        preserves enough remaining rollout budget to satisfy
        all trainability guarantees.

        A post-pilot dispatch to an unselected prompt activates
        that prompt and therefore creates a Gmin debt.
        """

        remaining_after_dispatch = (
            total_budget
            - (
                total_dispatched
                + 1
            )
        )

        selected_count_after = 0
        debt_after = 0

        for other in states.values():
            is_target = (
                other.prompt_id
                == state.prompt_id
            )

            # Any adaptive dispatch to an unselected prompt
            # activates it.
            selected_after = (
                other.selected
                or (
                    is_target
                    and not other.selected
                )
            )

            dispatched_after = (
                other.dispatched_g
                + (
                    1
                    if is_target
                    else 0
                )
            )

            if selected_after:
                selected_count_after += 1

                debt_after += max(
                    0,
                    gmin
                    - dispatched_after,
                )

        future_groups_after = max(
            0,
            min_selected
            - selected_count_after,
        )

        future_activation_after = (
            future_groups_after
            * max(
                0,
                gmin - pilot_g,
            )
        )

        required_after = (
            debt_after
            + future_activation_after
        )

        info = {
            "remaining_after_dispatch":
                int(
                    remaining_after_dispatch
                ),

            "selected_count_after":
                int(
                    selected_count_after
                ),

            "debt_rollouts_after":
                int(
                    debt_after
                ),

            "future_groups_after":
                int(
                    future_groups_after
                ),

            "future_activation_rollouts_after":
                int(
                    future_activation_after
                ),

            "total_reserved_rollouts_after":
                int(
                    required_after
                ),
        }

        return (
            remaining_after_dispatch
            >= required_after,
            info,
        )

    # --------------------------------------------------------
    # TARO scheduler
    # --------------------------------------------------------

    def _choose_next_prompt(
        self,
        *,
        states: dict[int, PromptState],
        active_jobs: dict[
            asyncio.Task,
            ActiveJob,
        ],
        total_dispatched: int,
        total_budget: int,
        pilot_g: int,
        gmin: int,
        gmax: int,
        min_selected: int,
        activation_epsilon: float,
        commit_window: int,
        adaptive_window: int,
        lambda_time: float,
        default_predicted_length: float,
        cost_model: MakespanModel,
        rng: np.random.Generator,
    ) -> tuple[
        int | None,
        dict[str, Any] | None,
    ]:
        """
        Choose one rollout to dispatch.

        Priority:

        1. Trainability debt.
        2. Utility-only activation until min_selected reached.
        3. After min_selected:
               new prompts must pass pilot-utility gate.
        4. Normal Event-TARO:
               ΔU - λ ΔM(A_t)

        All choices must satisfy budget reservation.
        """

        active_lengths = (
            self._active_predicted_lengths(
                active_jobs
            )
        )

        reservation_before = (
            self._reservation_snapshot(
                states,
                min_selected=min_selected,
                pilot_g=pilot_g,
                gmin=gmin,
            )
        )

        selected_count = (
            reservation_before[
                "selected_count"
            ]
        )

        # ----------------------------------------------------
        # Activation gate uses matched G0 evidence.
        # ----------------------------------------------------

        selected_activation_utilities = [
            float(state.pilot_utility)
            for state in states.values()
            if (
                state.selected
                and state.pilot_utility
                is not None
            )
        ]

        if selected_activation_utilities:
            worst_selected_activation_utility = min(
                selected_activation_utilities
            )
        else:
            worst_selected_activation_utility = None

        if (
            worst_selected_activation_utility
            is not None
        ):
            activation_threshold = float(
                worst_selected_activation_utility
                + activation_epsilon
            )
        else:
            activation_threshold = None

        feasibility: dict[
            int,
            dict[str, int],
        ] = {}

        # ----------------------------------------------------
        # Budget feasibility helper
        # ----------------------------------------------------

        def is_budget_feasible(
            state: PromptState,
        ) -> bool:
            ok, after = (
                self._budget_feasible_after_dispatch(
                    state=state,
                    states=states,
                    total_dispatched=total_dispatched,
                    total_budget=total_budget,
                    min_selected=min_selected,
                    pilot_g=pilot_g,
                    gmin=gmin,
                )
            )

            feasibility[
                state.prompt_id
            ] = after

            return ok

        # ----------------------------------------------------
        # TARO score helper
        # ----------------------------------------------------

        def score_state(
            state: PromptState,
        ) -> tuple[
            float,
            dict[str, Any],
        ]:
            # completed + inflight approximates the G level
            # that has already received rollout budget.
            effective_g = (
                state.completed_g
                + state.inflight_g
            )

            utility = state.utility

            delta_u = (
                utility
                / max(
                    effective_g + 1,
                    1,
                )
            )

            predicted_length = (
                state.predicted_length(
                    default_predicted_length
                )
            )

            normalized_cost = (
                cost_model
                .normalized_marginal_cost(
                    predicted_length,
                    active_lengths,
                )
            )

            score = (
                delta_u
                - lambda_time
                * normalized_cost
            )

            info: dict[str, Any] = {
                "prompt":
                    state.prompt_id,

                "completed_g":
                    state.completed_g,

                "inflight_g":
                    state.inflight_g,

                "dispatched_g":
                    state.dispatched_g,

                "commit_inflight_g":
                    state.commit_inflight_g,

                "adaptive_inflight_g":
                    state.adaptive_inflight_g,

                "commit_window":
                    commit_window,

                "adaptive_window":
                    adaptive_window,

                "selected":
                    state.selected,

                "utility":
                    utility,

                "delta_u":
                    delta_u,

                "predicted_length":
                    predicted_length,

                "normalized_delta_makespan":
                    normalized_cost,

                "score":
                    score,

                "active_predicted_lengths":
                    list(
                        active_lengths
                    ),

                "reservation_before":
                    dict(
                        reservation_before
                    ),

                "reservation_after":
                    dict(
                        feasibility.get(
                            state.prompt_id,
                            {},
                        )
                    ),
            }

            # Add activation diagnostics for an unselected
            # candidate.
            if not state.selected:
                if (
                    state.pilot_utility
                    is not None
                ):
                    activation_utility = float(
                        state.pilot_utility
                    )
                else:
                    activation_utility = float(
                        utility
                    )

                gate_required = (
                    selected_count
                    >= min_selected
                )

                gate_passed = (
                    not gate_required
                    or (
                        activation_threshold
                        is not None
                        and activation_utility
                        > activation_threshold
                    )
                )

                info.update(
                    {
                        "activation_gate_required":
                            gate_required,

                        "activation_epsilon":
                            activation_epsilon,

                        "activation_utility":
                            activation_utility,

                        "worst_selected_activation_utility":
                            worst_selected_activation_utility,

                        "activation_threshold":
                            activation_threshold,

                        "activation_gate_passed":
                            gate_passed,
                    }
                )

            return score, info

        # ----------------------------------------------------
        # Choose highest score, random tie-break.
        # ----------------------------------------------------

        def choose(
            candidates: list[PromptState],
            reason: str,
        ):
            scored = []

            for state in candidates:
                score, info = score_state(
                    state
                )

                info["reason"] = reason

                scored.append(
                    (
                        score,
                        float(
                            rng.random()
                        ),
                        state,
                        info,
                    )
                )

            if not scored:
                return None, None

            scored.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                ),
                reverse=True,
            )

            _, _, chosen, info = (
                scored[0]
            )

            return (
                chosen.prompt_id,
                info,
            )

        # ====================================================
        # 1. Existing trainability debt has priority.
        # ====================================================

        debt_candidates = [
            state
            for state in states.values()
            if (
                state.selected
                and state.dispatched_g < gmin
                and state.dispatched_g < gmax

                # New:
                # G2 -> Gmin may speculate several rollouts,
                # but only up to commit_window.
                and state.commit_inflight_g
                < commit_window
            )
        ]

        if debt_candidates:
            feasible_debt = [
                state
                for state in debt_candidates
                if is_budget_feasible(
                    state
                )
            ]

            if not feasible_debt:
                raise RuntimeError(
                    "Event-TARO reservation invariant "
                    "was violated: selected prompt debt "
                    "exists but no debt dispatch is "
                    "budget-feasible. "
                    f"reservation="
                    f"{reservation_before}, "
                    f"dispatched="
                    f"{total_dispatched}/"
                    f"{total_budget}"
                )

            return choose(
                feasible_debt,
                "trainability_debt",
            )

        # ====================================================
        # 2. Gather normal candidates.
        #
        # IMPORTANT:
        # There is deliberately NO all-pilots-complete
        # condition here.
        # ====================================================

        candidates: list[
            PromptState
        ] = []

        for state in states.values():

            # A prompt needs its own G0 evidence before TARO
            # can make a utility decision about it.
            if state.completed_g < pilot_g:
                continue

            if state.dispatched_g >= gmax:
                continue

            if not is_budget_feasible(
                state
            ):
                continue
            if (
                state.selected
                and state.dispatched_g >= gmin
            ):
                # No post-pilot feedback has arrived yet.
                if (
                    state.completed_g
                    <= pilot_g
                ):
                    continue

                # Only a bounded number of post-Gmin
                # speculative rollouts may be outstanding.
                if (
                    state.adaptive_inflight_g
                    >= adaptive_window
                ):
                    continue

            # =================================================
            # Bounded speculative admission
            # =================================================
            #
            # Selected prompt:
            #
            #   G2 -> G3/G4 may be committed together.
            #
            # But after Gmin, new rollout allocation must
            # consume fresh reward feedback.
            #
            if (
                state.selected
                and state.dispatched_g >= gmin
            ):
                # G3/G4 have been dispatched, but neither has
                # returned reward yet.
                if state.completed_g <= pilot_g:
                    continue

                # Beyond Gmin only a bounded number of
                # reward-unobserved trajectories may exist.
                if (
                    state.adaptive_inflight_g
                    >= adaptive_window
                ):
                    continue

            # -----------------------------------------------
            # Activation gate after min_selected.
            # -----------------------------------------------

            if (
                not state.selected
                and selected_count
                >= min_selected
            ):
                if (
                    state.pilot_utility
                    is not None
                ):
                    activation_utility = float(
                        state.pilot_utility
                    )
                else:
                    activation_utility = float(
                        state.utility
                    )

                if (
                    activation_threshold
                    is None
                    or activation_utility
                    <= activation_threshold
                ):
                    # Equal-quality G2 prompts no longer get
                    # automatically activated merely because
                    # their current G is smaller.
                    continue

            candidates.append(
                state
            )

        if not candidates:
            return None, None

        # ====================================================
        # 3. Until min_selected is reached, activation is
        #    Utility-only.
        #
        # If no unselected prompt has enough completed pilot
        # evidence yet, selected prompts are STILL allowed to
        # receive G5/G6/... work as long as the reservation
        # constraint is satisfied.
        #
        # This is the key work-conserving replacement for the
        # old all-pilots-complete barrier.
        # ====================================================

                # ====================================================
        # Utility-only activation before min_selected.
        #
        # V3 adds late-pilot protection:
        #
        # Do not permanently consume one of the remaining
        # trainable-group slots if unfinished pilot prompts
        # could still outrank the current candidate.
        # ====================================================

        if selected_count < min_selected:

            unselected = [
                state
                for state in candidates
                if not state.selected
            ]

            if unselected:
                ranked = []

                for state in unselected:
                    if state.pilot_utility is not None:
                        activation_utility = float(
                            state.pilot_utility
                        )
                    else:
                        activation_utility = float(
                            state.utility
                        )

                    ranked.append(
                        (
                            activation_utility,
                            float(rng.random()),
                            state,
                        )
                    )

                ranked.sort(
                    key=lambda item: (
                        item[0],
                        item[1],
                    ),
                    reverse=True,
                )

                (
                    candidate_utility,
                    _,
                    chosen,
                ) = ranked[0]

                activation_safe = True
                safety_threshold = None

                unresolved_states = [
                    state
                    for state in states.values()
                    if (
                        not state.selected
                        and state.prompt_id
                        != chosen.prompt_id
                        and state.completed_g < pilot_g
                    )
                ]

                unresolved_upper_bounds = [
                    self._pilot_utility_upper_bound(
                        state,
                        pilot_g,
                    )
                    for state in unresolved_states
                ]

                # --------------------------------------------
                # Late-pilot protection applies ONLY when
                # choosing the last required trainable group.
                #
                # Example min_selected=3:
                #
                # selected_count 0 -> activate normally
                # selected_count 1 -> activate normally
                # selected_count 2 -> protect final slot
                # --------------------------------------------

                protect_final_slot = (
                    selected_count
                    == min_selected - 1
                )

                if (
                    protect_final_slot
                    and unresolved_upper_bounds
                ):
                    safety_threshold = float(
                        max(
                            unresolved_upper_bounds
                        )
                    )

                    activation_safe = (
                        candidate_utility
                        + activation_epsilon
                        >= safety_threshold
                    )

                _, info = score_state(
                    chosen
                )

                info.update(
                    {
                        "activation_utility_only":
                            True,

                        "late_pilot_guard":
                            bool(
                                protect_final_slot
                            ),

                        "unresolved_pilot_count":
                            int(
                                len(
                                    unresolved_states
                                )
                            ),

                        "unresolved_pilot_ids":
                            [
                                state.prompt_id
                                for state
                                in unresolved_states
                            ],

                        "unresolved_utility_upper_bounds":
                            [
                                float(value)
                                for value
                                in unresolved_upper_bounds
                            ],

                        "activation_safety_threshold":
                            safety_threshold,

                        "activation_safe":
                            bool(
                                activation_safe
                            ),
                    }
                )

                if activation_safe:
                    info[
                        "reason"
                    ] = "utility_activation"

                    return (
                        chosen.prompt_id,
                        info,
                    )

                # The final trainable slot is protected.
                #
                # Do not return immediately: already-selected
                # prompts may still have feedback-safe work.
                candidates = [
                    state
                    for state in candidates
                    if state.selected
                ]

                if not candidates:
                    return None, None

        # ====================================================
        # 4. Standard Event-TARO dispatch:
        #
        #       S_i = ΔU_i - λ ΔM_i(A_t)
        # ====================================================

        return choose(
            candidates,
            "taro_dispatch",
        )

    # --------------------------------------------------------
    # Main Event-TARO rollout loop
    # --------------------------------------------------------

    async def generate_sequences_event_taro(
        self,
        batch,
    ):
        cfg = self._taro_cfg()

        pilot_g = int(
            cfg.get(
                "pilot_g",
                2,
            )
        )

        gmin = int(
            cfg.get(
                "gmin",
                4,
            )
        )

        gmax = int(
            cfg.get(
                "gmax",
                8,
            )
        )

        total_budget = int(
            cfg.get(
                "total_budget",
                20,
            )
        )

        # V2:
        # Match the current vLLM max_num_seqs=16.
        max_inflight = int(
            cfg.get(
                "max_inflight",
                16,
            )
        )

        min_selected = int(
            cfg.get(
                "min_selected",
                3,
            )
        )

        activation_epsilon = float(
            cfg.get(
                "activation_epsilon",
                1e-6,
            )
        )
                # G2 -> Gmin may be issued together.
        commit_window = int(
            cfg.get(
                "commit_window",
                2,
            )
        )

        # Beyond Gmin, only one reward-unobserved rollout is
        # allowed by default.
        adaptive_window = int(
            cfg.get(
                "adaptive_window",
                1,
            )
        )

        lambda_time = float(
            cfg.get(
                "lambda_time",
                0.02,
            )
        )

        default_predicted_length = float(
            cfg.get(
                "default_predicted_length",
                4096.0,
            )
        )

        makespan_csv = str(
            cfg.get(
                "makespan_csv"
            )
        )

        log_dir = Path(
            str(
                cfg.get(
                    "log_dir",
                    "/tmp/taro_event",
                )
            )
        )

        base_seed = int(
            cfg.get(
                "seed",
                self.config.data.get(
                    "seed",
                    42,
                ),
            )
        )

        # ====================================================
        # Basic validation
        # ====================================================

        pool_size = len(batch)

        if commit_window < 1:
            raise RuntimeError(
                "commit_window must be >= 1"
            )

        if adaptive_window < 1:
            raise RuntimeError(
                "adaptive_window must be >= 1"
            )

        if pool_size <= 0:
            raise RuntimeError(
                "Event-TARO received an "
                "empty prompt pool."
            )

        if (
            pool_size * pilot_g
            > total_budget
        ):
            raise RuntimeError(
                "Pilot budget exceeds total "
                "rollout budget: "
                f"{pool_size} * {pilot_g} "
                f"> {total_budget}"
            )

        if max_inflight < 1:
            raise RuntimeError(
                "max_inflight must be >= 1"
            )

        if (
            min_selected < 1
            or min_selected > pool_size
        ):
            raise RuntimeError(
                "min_selected must be in "
                f"[1, {pool_size}], got "
                f"{min_selected}"
            )

        if activation_epsilon < 0:
            raise RuntimeError(
                "activation_epsilon "
                "must be >= 0"
            )

        if (
            not makespan_csv
            or makespan_csv == "None"
        ):
            raise RuntimeError(
                "trainer.taro_online."
                "makespan_csv is required."
            )

        # ====================================================
        # Cost model
        # ====================================================

        if not hasattr(
            self,
            "_taro_event_cost_model",
        ):
            self._taro_event_cost_model = (
                MakespanModel(
                    makespan_csv
                )
            )

        cost_model: MakespanModel = (
            self._taro_event_cost_model
        )

        # ====================================================
        # Rollout sampling config
        # ====================================================

        config = self.rollout_config

        sampling_params = {
            "temperature":
                config.temperature,

            "top_p":
                config.top_p,

            "top_k":
                config.top_k,

            "repetition_penalty":
                1.0,

            "logprobs":
                config.calculate_log_probs,
        }

        # Stock AgentLoop expects agent_name.
        if (
            "agent_name"
            not in batch.non_tensor_batch
        ):
            default_agent_loop = (
                config.agent
                .default_agent_loop
            )

            batch.non_tensor_batch[
                "agent_name"
            ] = np.array(
                [
                    default_agent_loop
                ]
                * pool_size,
                dtype=object,
            )

        global_step = int(
            batch.meta_info.get(
                "global_steps",
                -1,
            )
        )

        rng = np.random.default_rng(
            base_seed
            + max(
                global_step,
                0,
            )
        )

        if (
            "index"
            in batch.non_tensor_batch
        ):
            sample_indices = (
                batch.non_tensor_batch[
                    "index"
                ]
            )
        else:
            sample_indices = np.arange(
                pool_size
            )

        # ====================================================
        # Scheduler state
        # ====================================================

        states = {
            prompt_id: PromptState(
                prompt_id=prompt_id
            )
            for prompt_id
            in range(pool_size)
        }

        active_jobs: dict[
            asyncio.Task,
            ActiveJob,
        ] = {}

        all_outputs: list[Any] = []
        all_parent_indices: list[int] = []
        all_dispatch_ids: list[int] = []
        all_rollout_ns: list[int] = []

        events: list[
            dict[str, Any]
        ] = []

        decision_trace: list[
            dict[str, Any]
        ] = []

        total_dispatched = 0
        total_completed = 0

        dispatch_id = 0

        t0 = time.perf_counter()

        # Integral of active request count over wall-clock
        # time. Used to calculate avg_active.
        active_area = 0.0
        last_area_time = t0
        last_active_count = 0

        def update_active_area(
            now: float,
        ):
            nonlocal active_area, last_area_time, last_active_count

            active_area += (
                last_active_count
                * (
                    now
                    - last_area_time
                )
            )

            last_area_time = now
            last_active_count = len(
                active_jobs
            )

        def build_kwargs(
            prompt_id: int,
        ) -> dict[str, Any]:
            return {
                key: self._value_at(
                    value,
                    prompt_id,
                )
                for key, value
                in batch.non_tensor_batch.items()
            }

        # ====================================================
        # Launch one trajectory
        # ====================================================

        def launch(
            prompt_id: int,
            reason: str,
            decision_info: (
                dict[str, Any]
                | None
            ) = None,
        ):
            nonlocal total_dispatched, dispatch_id, last_active_count

            state = states[
                prompt_id
            ]

            if (
                state.dispatched_g
                >= gmax
            ):
                raise RuntimeError(
                    f"Prompt {prompt_id} "
                    f"exceeds gmax={gmax}."
                )

            if (
                total_dispatched
                >= total_budget
            ):
                raise RuntimeError(
                    "launch() called after "
                    "total rollout budget was "
                    "already exhausted."
                )

            # The first post-pilot dispatch activates
            # the prompt.
            if (
                state.dispatched_g
                >= pilot_g
            ):
                state.selected = True

            rollout_n = (
                state.next_rollout_n
            )

            state.next_rollout_n += 1
            state.dispatched_g += 1
            state.inflight_g += 1

            # ------------------------------------------------
            # Bounded speculation bookkeeping
            # ------------------------------------------------
            #
            # rollout_n is zero-based:
            #
            #   rollout_n < pilot_g
            #       -> pilot rollout
            #
            #   pilot_g <= rollout_n < gmin
            #       -> G2 -> Gmin commitment rollout
            #
            #   rollout_n >= gmin
            #       -> post-Gmin adaptive rollout
            #
            # Example for pilot_g=2, gmin=4:
            #
            #   rollout_n 0,1 : pilot
            #   rollout_n 2,3 : G3,G4
            #   rollout_n 4+  : G5+
            #
            if (
                rollout_n >= pilot_g
                and rollout_n < gmin
            ):
                state.commit_inflight_g += 1

            elif rollout_n >= gmin:
                state.adaptive_inflight_g += 1

            predicted_length = (
                state.predicted_length(
                    default_predicted_length
                )
            )

            now = time.perf_counter()

            job = ActiveJob(
                dispatch_id=dispatch_id,
                prompt_id=prompt_id,
                rollout_n=rollout_n,
                predicted_length=(
                    predicted_length
                ),
                dispatch_time=now,
            )

            trajectory = {
                "step":
                    global_step,

                "sample_index":
                    int(
                        sample_indices[
                            prompt_id
                        ]
                    ),

                "rollout_n":
                    rollout_n,

                "validate":
                    False,
            }

            kwargs = build_kwargs(
                prompt_id
            )

            task = asyncio.create_task(
                self._run_agent_loop(
                    sampling_params,
                    trajectory,
                    trace=True,
                    **kwargs,
                )
            )

            active_jobs[
                task
            ] = job

            total_dispatched += 1
            dispatch_id += 1

            last_active_count = len(
                active_jobs
            )

            record: dict[
                str,
                Any,
            ] = {
                "type":
                    "dispatch",

                "t":
                    now - t0,

                "dispatch_id":
                    job.dispatch_id,

                "prompt":
                    prompt_id,

                "rollout_n":
                    rollout_n,

                "predicted_length":
                    predicted_length,

                "reason":
                    reason,

                "active_after":
                    len(
                        active_jobs
                    ),

                "total_dispatched":
                    total_dispatched,
            }

            if (
                decision_info
                is not None
            ):
                record[
                    "decision"
                ] = decision_info

            events.append(
                record
            )

        # ====================================================
        # Initial pilot jobs
        # ====================================================

        pilot_queue: list[int] = []

        for prompt_id in range(
            pool_size
        ):
            pilot_queue.extend(
                [prompt_id]
                * pilot_g
            )

        # With the current intended setup:
        #
        #     pool = 4
        #     pilot_g = 2
        #     max_inflight = 16
        #
        # all eight pilots launch immediately.
        while (
            pilot_queue
            and len(active_jobs)
            < max_inflight
        ):
            launch(
                pilot_queue.pop(0),
                reason="pilot",
            )

        # General fallback if max_inflight happens to be
        # below total pilot count.
        pending_pilots = pilot_queue

        # ====================================================
        # Event loop
        # ====================================================

        while active_jobs:

            # ------------------------------------------------
            # Wait ONLY until at least one trajectory finishes.
            # ------------------------------------------------

            done, _ = await asyncio.wait(
                list(
                    active_jobs.keys()
                ),
                return_when=(
                    asyncio.FIRST_COMPLETED
                ),
            )

            now = time.perf_counter()

            update_active_area(
                now
            )

            # ------------------------------------------------
            # Process all completions already ready at this
            # event boundary.
            # ------------------------------------------------

            for task in done:

                job = active_jobs.pop(
                    task
                )

                output = await task

                finish = (
                    time.perf_counter()
                )

                state = states[
                    job.prompt_id
                ]

                state.inflight_g -= 1
                state.completed_g += 1

                # ------------------------------------------------
                # Release bounded-speculation occupancy.
                #
                # This must mirror launch() exactly.
                # ------------------------------------------------
                if (
                    job.rollout_n >= pilot_g
                    and job.rollout_n < gmin
                ):
                    state.commit_inflight_g -= 1

                    if state.commit_inflight_g < 0:
                        raise RuntimeError(
                            "Negative commit_inflight_g "
                            f"for prompt {job.prompt_id}"
                        )

                elif job.rollout_n >= gmin:
                    state.adaptive_inflight_g -= 1

                    if state.adaptive_inflight_g < 0:
                        raise RuntimeError(
                            "Negative adaptive_inflight_g "
                            f"for prompt {job.prompt_id}"
                        )

                if (
                    output.reward_score
                    is None
                ):
                    raise RuntimeError(
                        "Event-TARO requires "
                        "per-trajectory reward "
                        "before refill, but "
                        "AgentLoopOutput."
                        "reward_score is None."
                    )

                reward = float(
                    output.reward_score
                )

                response_tokens = (
                    self._response_tokens(
                        output
                    )
                )

                state.rewards.append(
                    reward
                )

                state.lengths.append(
                    response_tokens
                )

                # Snapshot matched G0 utility exactly once.
                if (
                    state.completed_g
                    == pilot_g
                    and state.pilot_utility
                    is None
                ):
                    state.pilot_utility = (
                        state.utility
                    )

                all_outputs.append(
                    output
                )

                all_parent_indices.append(
                    job.prompt_id
                )

                all_dispatch_ids.append(
                    job.dispatch_id
                )

                all_rollout_ns.append(
                    job.rollout_n
                )

                total_completed += 1

                events.append(
                    {
                        "type":
                            "finish",

                        "t":
                            finish - t0,

                        "dispatch_id":
                            job.dispatch_id,

                        "prompt":
                            job.prompt_id,

                        "rollout_n":
                            job.rollout_n,

                        "reward":
                            reward,

                        "response_tokens":
                            response_tokens,

                        "latency_s":
                            finish
                            - job.dispatch_time,

                        "completed_g":
                            state.completed_g,

                        "active_after":
                            len(
                                active_jobs
                            ),

                        "total_completed":
                            total_completed,
                    }
                )

            last_active_count = len(
                active_jobs
            )

            # =================================================
            # Immediate refill loop
            # =================================================

            while (
                len(active_jobs)
                < max_inflight
                and total_dispatched
                < total_budget
            ):

                # Complete the initial G0 pilot dispatch set
                # before adaptive rollouts if max_inflight
                # happened to be very small.
                if pending_pilots:
                    prompt_id = (
                        pending_pilots.pop(
                            0
                        )
                    )

                    launch(
                        prompt_id,
                        reason="pilot",
                    )

                    continue

                prompt_id, info = (
                    self._choose_next_prompt(
                        states=states,

                        active_jobs=
                            active_jobs,

                        total_dispatched=
                            total_dispatched,

                        total_budget=
                            total_budget,

                        pilot_g=
                            pilot_g,

                        gmin=
                            gmin,

                        gmax=
                            gmax,

                        min_selected=
                            min_selected,

                        # IMPORTANT:
                        # This argument was missing in the
                        # earlier attachment version.
                        activation_epsilon=
                            activation_epsilon,


                        commit_window=
                            commit_window,

                        adaptive_window=
                            adaptive_window,

                        lambda_time=
                            lambda_time,

                        default_predicted_length=
                            default_predicted_length,

                        cost_model=
                            cost_model,

                        rng=
                            rng,
                    )
                )

                if prompt_id is None:
                    # No prompt currently has enough completed
                    # evidence for another safe dispatch.
                    #
                    # Do NOT create fake work. Keep currently
                    # active trajectories running and wait for
                    # the next completion event.
                    break

                info = dict(
                    info or {}
                )

                info.update(
                    {
                        "event_t":
                            time.perf_counter()
                            - t0,

                        "total_dispatched_before":
                            total_dispatched,

                        "active_count_before":
                            len(
                                active_jobs
                            ),
                    }
                )

                decision_trace.append(
                    info
                )

                launch(
                    prompt_id,

                    reason=info.get(
                        "reason",
                        "taro_dispatch",
                    ),

                    decision_info=info,
                )

            last_active_count = len(
                active_jobs
            )

            # ------------------------------------------------
            # Deadlock guard
            # ------------------------------------------------

            if (
                not active_jobs
                and total_dispatched
                < total_budget
            ):
                reservation = (
                    self._reservation_snapshot(
                        states,
                        min_selected=
                            min_selected,
                        pilot_g=
                            pilot_g,
                        gmin=
                            gmin,
                    )
                )

                raise RuntimeError(
                    "Event-TARO deadlocked "
                    "before exhausting rollout "
                    "budget. "
                    f"dispatched="
                    f"{total_dispatched}/"
                    f"{total_budget}, "
                    f"reservation="
                    f"{reservation}, "
                    f"states={states}"
                )

        # ====================================================
        # Generation finished
        # ====================================================

        t_end = time.perf_counter()

        update_active_area(
            t_end
        )

        if (
            total_dispatched
            != total_budget
            or total_completed
            != total_budget
        ):
            raise RuntimeError(
                "Event-TARO rollout budget "
                "mismatch: "
                f"dispatched="
                f"{total_dispatched}, "
                f"completed="
                f"{total_completed}, "
                f"budget="
                f"{total_budget}"
            )

        # ====================================================
        # Build variable-G GRPO batch
        # ====================================================

        selected_prompt_ids = {
            state.prompt_id
            for state
            in states.values()
            if state.selected
        }

        if (
            len(selected_prompt_ids)
            < min_selected
        ):
            raise RuntimeError(
                "Event-TARO ended with only "
                f"{len(selected_prompt_ids)} "
                "selected prompts, below "
                f"min_selected="
                f"{min_selected}."
            )

        for prompt_id in (
            selected_prompt_ids
        ):
            state = states[
                prompt_id
            ]

            if (
                state.completed_g
                < gmin
            ):
                raise RuntimeError(
                    f"Selected prompt "
                    f"{prompt_id} ended "
                    f"with G="
                    f"{state.completed_g} "
                    f"< gmin={gmin}."
                )

        # Non-selected prompt pilots are generated but not used
        # by GRPO.
        train_indices = [
            index
            for index, parent_idx
            in enumerate(
                all_parent_indices
            )
            if (
                parent_idx
                in selected_prompt_ids
            )
        ]

        if not train_indices:
            raise RuntimeError(
                "Event-TARO produced no "
                "trainable prompt groups."
            )

        train_outputs = [
            all_outputs[index]
            for index
            in train_indices
        ]

        train_parent_indices = [
            all_parent_indices[index]
            for index
            in train_indices
        ]

        train_dispatch_ids = [
            all_dispatch_ids[index]
            for index
            in train_indices
        ]

        train_rollout_ns = [
            all_rollout_ns[index]
            for index
            in train_indices
        ]

        output = self._postprocess(
            train_outputs,
            input_non_tensor_batch=None,
        )

        # ====================================================
        # Reconstruct parent non-tensor fields
        # ====================================================

        for (
            key,
            values,
        ) in (
            batch.non_tensor_batch.items()
        ):
            if (
                key
                in output.non_tensor_batch
            ):
                continue

            repeated = np.empty(
                len(
                    train_parent_indices
                ),
                dtype=object,
            )

            repeated[:] = [
                self._value_at(
                    values,
                    parent_idx,
                )
                for parent_idx
                in train_parent_indices
            ]

            output.non_tensor_batch[
                key
            ] = repeated

        if (
            "multi_modal_inputs"
            not in output.non_tensor_batch
        ):
            multi_modal_inputs = (
                np.empty(
                    len(
                        train_parent_indices
                    ),
                    dtype=object,
                )
            )

            multi_modal_inputs[:] = [
                {}
                for _ in (
                    train_parent_indices
                )
            ]

            output.non_tensor_batch[
                "multi_modal_inputs"
            ] = multi_modal_inputs

        # Diagnostic IDs. Original prompt uid remains available
        # in the batch and is what GRPO group normalization uses.
        output.non_tensor_batch[
            "taro_prompt_id"
        ] = np.asarray(
            train_parent_indices,
            dtype=np.int32,
        )

        output.non_tensor_batch[
            "taro_dispatch_id"
        ] = np.asarray(
            train_dispatch_ids,
            dtype=np.int32,
        )

        output.non_tensor_batch[
            "taro_rollout_n"
        ] = np.asarray(
            train_rollout_ns,
            dtype=np.int32,
        )

        # ====================================================
        # Metrics / invariants
        # ====================================================

        rollout_time = (
            t_end - t0
        )

        avg_active = (
            active_area
            / max(
                rollout_time,
                1e-8,
            )
        )

        final_g = {
            str(prompt_id):
                int(
                    state.completed_g
                )
            for (
                prompt_id,
                state,
            )
            in states.items()
        }

        selected_g = [
            states[
                prompt_id
            ].completed_g
            for prompt_id
            in sorted(
                selected_prompt_ids
            )
        ]

        final_reservation = (
            self._reservation_snapshot(
                states,
                min_selected=
                    min_selected,
                pilot_g=
                    pilot_g,
                gmin=
                    gmin,
            )
        )

        # Every promised trainability obligation must have been
        # satisfied by the end.
        if (
            final_reservation[
                "total_reserved_rollouts"
            ]
            != 0
        ):
            raise RuntimeError(
                "Event-TARO finished "
                "with unresolved rollout "
                "reservation: "
                f"{final_reservation}"
            )

        taro_metrics = {
            "taro_event/generated_rollouts":
                float(
                    total_completed
                ),

            "taro_event/train_rollouts":
                float(
                    len(
                        train_outputs
                    )
                ),

            "taro_event/dropped_rollouts":
                float(
                    total_completed
                    - len(
                        train_outputs
                    )
                ),

            "taro_event/num_selected":
                float(
                    len(
                        selected_prompt_ids
                    )
                ),

            "taro_event/final_g_min":
                float(
                    min(
                        selected_g
                    )
                ),

            "taro_event/final_g_mean":
                float(
                    np.mean(
                        selected_g
                    )
                ),

            "taro_event/final_g_max":
                float(
                    max(
                        selected_g
                    )
                ),

            "taro_event/rollout_time_s":
                float(
                    rollout_time
                ),

            "taro_event/avg_active":
                float(
                    avg_active
                ),

            "taro_event/max_inflight":
                float(
                    max_inflight
                ),

            "taro_event/commit_window":
                float(
                    commit_window
                ),

            "taro_event/adaptive_window":
                float(
                    adaptive_window
                ),

            "taro_event/final_reserved_rollouts":
                float(
                    final_reservation[
                        "total_reserved_rollouts"
                    ]
                ),
        }

        # ====================================================
        # Detailed JSONL trace
        # ====================================================

        record = {
            "global_step":
                global_step,

            "pool_size":
                pool_size,

            "pilot_g":
                pilot_g,

            "gmin":
                gmin,

            "gmax":
                gmax,

            "min_selected":
                min_selected,

            "activation_epsilon":
                activation_epsilon,

            "commit_window":
                commit_window,

            "adaptive_window":
                adaptive_window,

            "total_budget":
                total_budget,

            "max_inflight":
                max_inflight,

            "lambda_time":
                lambda_time,

            "selected_prompts":
                sorted(
                    selected_prompt_ids
                ),

            "final_g_generated":
                final_g,

            "prompt_rewards": {
                str(prompt_id):
                    states[
                        prompt_id
                    ].rewards
                for prompt_id
                in range(
                    pool_size
                )
            },

            "prompt_lengths": {
                str(prompt_id):
                    states[
                        prompt_id
                    ].lengths
                for prompt_id
                in range(
                    pool_size
                )
            },

            "pilot_utilities": {
                str(prompt_id):
                    states[
                        prompt_id
                    ].pilot_utility
                for prompt_id
                in range(
                    pool_size
                )
            },

            "rollout_time_s":
                rollout_time,

            "avg_active":
                avg_active,

            "final_reservation":
                final_reservation,

            "generated_rollouts":
                total_completed,

            "train_rollouts":
                len(
                    train_outputs
                ),

            "decisions":
                decision_trace,

            "events":
                events,
        }

        log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        trace_path = (
            log_dir
            / "taro_event_decisions.jsonl"
        )

        with trace_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        output.meta_info[
            "taro_metrics"
        ] = taro_metrics

        output.meta_info[
            "taro_event_record"
        ] = record

        return output


# ============================================================
# Custom manager
# ============================================================


class TAROEventAgentLoopManager(
    AgentLoopManager
):
    """
    Custom AgentLoopManager for Event-TARO.

    Training:
        routes the whole prompt pool to one Event-TARO worker
        so one scheduler has a consistent live active set.

    Validation:
        uses stock verl generation behavior.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.agent_loop_workers_class = (
            ray.remote(
                TAROEventAgentLoopWorker
            )
        )

        super().__init__(
            *args,
            **kwargs,
        )

    @auto_await
    async def generate_sequences(
        self,
        prompts,
    ):
        # ====================================================
        # Validation: preserve stock verl behavior
        # ====================================================

        if prompts.meta_info.get(
            "validate",
            False,
        ):
            chunks = prompts.chunk(
                len(
                    self.agent_loop_workers
                )
            )

            outputs = await asyncio.gather(
                *[
                    worker.generate_sequences.remote(
                        chunk
                    )
                    for (
                        worker,
                        chunk,
                    )
                    in zip(
                        self.agent_loop_workers,
                        chunks,
                        strict=True,
                    )
                ]
            )

            output = type(
                prompts
            ).concat(
                outputs
            )

            metrics = [
                item.meta_info.pop(
                    "metrics"
                )
                for item
                in outputs
            ]

            timing = (
                self._performance_metrics(
                    metrics,
                    output,
                )
            )

            output.meta_info = {
                "timing":
                    timing,
                **outputs[0].meta_info,
            }

            return output

        # ====================================================
        # Training: one scheduler must own the whole pool
        # ====================================================

        if (
            len(
                self.agent_loop_workers
            )
            != 1
        ):
            raise RuntimeError(
                "Event-TARO currently requires "
                "actor_rollout_ref.rollout.agent."
                "num_workers=1 so one scheduler "
                "owns the complete prompt pool "
                "and live active set."
            )

        worker = (
            self.agent_loop_workers[0]
        )

        output = await (
            worker
            .generate_sequences_event_taro
            .remote(
                prompts
            )
        )

        metrics_list = (
            output.meta_info.pop(
                "metrics",
                [],
            )
        )

        if metrics_list:
            timing = (
                self._performance_metrics(
                    [metrics_list],
                    output,
                )
            )
        else:
            timing = {}

        taro_metrics = (
            output.meta_info.pop(
                "taro_metrics",
                {},
            )
        )

        taro_event_record = (
            output.meta_info.pop(
                "taro_event_record",
                None,
            )
        )

        output.meta_info = {
            "timing":
                timing,

            "taro_metrics":
                taro_metrics,

            "taro_event_record":
                taro_event_record,

            **output.meta_info,
        }

        return output