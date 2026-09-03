import numpy as np

from taro_event_manager import (
    PromptState,
    TAROEventAgentLoopWorker,
)


# ============================================================
# Dummy cost model
#
# Activation-gate targeted test does not care about makespan.
# Set marginal cost to zero so the test only exercises the
# activation logic.
# ============================================================


class ZeroCostModel:
    def normalized_marginal_cost(
        self,
        candidate_length,
        active_lengths,
    ):
        return 0.0


def make_selected_state(
    prompt_id: int,
    pilot_utility: float = 0.15,
):
    """
    Build an already selected/trainable prompt.

    We deliberately put it at gmax so it cannot compete for
    another rollout. This isolates the activation-gate test.
    """

    state = PromptState(
        prompt_id=prompt_id
    )

    state.rewards = [
        1.0,
        1.0,
        1.0,
        1.0,
    ]

    state.lengths = [
        4096,
        4096,
        4096,
        4096,
    ]

    state.completed_g = 4
    state.inflight_g = 0
    state.dispatched_g = 4
    state.next_rollout_n = 4

    state.selected = True
    state.pilot_utility = pilot_utility

    return state


def make_candidate_state(
    prompt_id: int,
    rewards,
):
    """
    Build an unselected prompt with exactly G0=2 pilot
    completions.
    """

    state = PromptState(
        prompt_id=prompt_id
    )

    state.rewards = list(rewards)

    state.lengths = [
        4096,
        4096,
    ]

    state.completed_g = 2
    state.inflight_g = 0
    state.dispatched_g = 2
    state.next_rollout_n = 2

    state.selected = False

    # Snapshot utility at exactly pilot_g=2.
    state.pilot_utility = state.utility

    return state


def run_case(
    name: str,
    candidate_rewards,
    expect_pass: bool,
):
    # Avoid AgentLoopWorker initialization.
    # _choose_next_prompt only needs helper methods from self.
    worker = object.__new__(
        TAROEventAgentLoopWorker
    )

    states = {
        0: make_selected_state(0, 0.15),
        1: make_selected_state(1, 0.15),
        2: make_selected_state(2, 0.15),
        3: make_candidate_state(
            3,
            candidate_rewards,
        ),
    }

    # Three selected prompts:
    #
    #   3 * G4 = 12
    #
    # Candidate:
    #
    #   G2 = 2
    #
    # Already dispatched:
    #
    #   14
    #
    # Give total budget 16.
    #
    # Activating prompt 3 dispatches G3, leaving exactly one
    # rollout for its G4 trainability debt.
    #
    # Therefore activation is budget-feasible.
    total_dispatched = sum(
        state.dispatched_g
        for state in states.values()
    )

    assert total_dispatched == 14

    chosen_prompt, info = (
        worker._choose_next_prompt(
            states=states,
            active_jobs={},
            total_dispatched=total_dispatched,
            total_budget=16,
            pilot_g=2,
            gmin=4,
            gmax=4,
            min_selected=3,
            activation_epsilon=1e-6,
            lambda_time=0.0,
            default_predicted_length=4096.0,
            cost_model=ZeroCostModel(),
            rng=np.random.default_rng(0),
        )
    )

    candidate = states[3]

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        "candidate pilot utility:",
        candidate.pilot_utility,
    )

    print(
        "expected threshold:",
        0.15 + 1e-6,
    )

    print(
        "chosen prompt:",
        chosen_prompt,
    )

    print(
        "decision info:",
        info,
    )

    if expect_pass:
        # All selected prompts are already at gmax.
        # Therefore prompt 3 is the only legal candidate if
        # it passes the activation gate.
        assert chosen_prompt == 3, (
            "PASS case failed: "
            "candidate should have been activated."
        )

        assert info is not None

        assert (
            info[
                "activation_gate_required"
            ]
            is True
        )

        assert (
            info[
                "activation_gate_passed"
            ]
            is True
        )

        assert (
            info["activation_utility"]
            > info["activation_threshold"]
        )

        print(
            "[PASS] activation gate accepted "
            "prompt 3 as expected."
        )

    else:
        # Selected prompts are at gmax and candidate is
        # rejected by the gate. Therefore there should be no
        # legal next dispatch.
        assert chosen_prompt is None, (
            "REJECT case failed: "
            "candidate should have been blocked."
        )

        assert info is None

        print(
            "[PASS] activation gate rejected "
            "prompt 3 as expected."
        )


def main():

    # --------------------------------------------------------
    # Reject:
    #
    # rewards [1, 1]
    #
    # Beta expected variance:
    #
    #   0.15
    #
    # threshold:
    #
    #   0.150001
    #
    # 0.15 <= 0.150001
    #     -> reject
    # --------------------------------------------------------

    run_case(
        name="ACTIVATION GATE REJECT",
        candidate_rewards=[
            1.0,
            1.0,
        ],
        expect_pass=False,
    )

    # --------------------------------------------------------
    # Pass:
    #
    # rewards [1, 0]
    #
    # Beta expected variance:
    #
    #   0.20
    #
    # 0.20 > 0.150001
    #     -> pass
    # --------------------------------------------------------

    run_case(
        name="ACTIVATION GATE PASS",
        candidate_rewards=[
            1.0,
            0.0,
        ],
        expect_pass=True,
    )

    print()
    print("=" * 70)
    print(
        "ALL ACTIVATION-GATE TARGETED "
        "TESTS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()