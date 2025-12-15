# stag_game.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Any

# Actions: 0 = Stag, 1 = Hare
PAYOFFS: Dict[Tuple[int, int], Tuple[float, float]] = {
    (0, 0): (4.0, 4.0),
    (0, 1): (0.0, 3.0),
    (1, 0): (3.0, 0.0),
    (1, 1): (2.0, 2.0),
}


@dataclass
class QLearningConfig:
    episodes: int = 10_000
    alpha: float = 0.1
    gamma: float = 0.0
    epsilon: float = 0.1
    noise_p: float = 0.0
    record_q_every: int = 10


def play_stag_hunt(a1: int, a2: int, noise_p: float, rng: np.random.Generator):
    if rng.random() < noise_p:
        a1 = 1 - a1
    if rng.random() < noise_p:
        a2 = 1 - a2
    r1, r2 = PAYOFFS[(a1, a2)]
    return (a1, a2), (r1, r2)


def epsilon_greedy(Q: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, len(Q)))
    return int(np.argmax(Q))


def run_two_qlearners(
    config: QLearningConfig,
    rng: np.random.Generator,
    record_q_history: bool = False,
):
    """
    Run two Q-learning agents against each other in the Stag Hunt.

    Returns:
        coop_rate: fraction of episodes where executed outcome was (Stag, Stag)
        stats: dict with detailed stats for this run
        q1_hist, q2_hist, episode_marks: Q-value history (or None if not recorded)
    """
    # Q1[a], Q2[a] for a in {0=Stag, 1=Hare}
    Q1 = np.zeros(2, dtype=float)
    Q2 = np.zeros(2, dtype=float)

    coop_count = 0
    outcome_counts = {"SS": 0, "SH": 0, "HS": 0, "HH": 0}

    # Track intended and executed actions for each agent
    intended_counts_1 = {"Stag": 0, "Hare": 0}
    intended_counts_2 = {"Stag": 0, "Hare": 0}
    executed_counts_1 = {"Stag": 0, "Hare": 0}
    executed_counts_2 = {"Stag": 0, "Hare": 0}

    q1_hist: List[Tuple[float, float]] = []
    q2_hist: List[Tuple[float, float]] = []
    episode_marks: List[int] = []

    for ep in range(config.episodes):
        # Intended actions (epsilon-greedy)
        a1_intended = epsilon_greedy(Q1, config.epsilon, rng)
        a2_intended = epsilon_greedy(Q2, config.epsilon, rng)

        intended_counts_1["Stag" if a1_intended == 0 else "Hare"] += 1
        intended_counts_2["Stag" if a2_intended == 0 else "Hare"] += 1

        # Environment applies action noise
        (a1_exec, a2_exec), (r1, r2) = play_stag_hunt(
            a1_intended, a2_intended, config.noise_p, rng
        )

        executed_counts_1["Stag" if a1_exec == 0 else "Hare"] += 1
        executed_counts_2["Stag" if a2_exec == 0 else "Hare"] += 1

        # Track outcomes
        if a1_exec == 0 and a2_exec == 0:
            coop_count += 1
            outcome_counts["SS"] += 1
        elif a1_exec == 0 and a2_exec == 1:
            outcome_counts["SH"] += 1
        elif a1_exec == 1 and a2_exec == 0:
            outcome_counts["HS"] += 1
        else:
            outcome_counts["HH"] += 1

        # Q-learning updates using intended action
        Q1[a1_intended] += config.alpha * (r1 - Q1[a1_intended])
        Q2[a2_intended] += config.alpha * (r2 - Q2[a2_intended])

        # Optionally record Q trajectories
        if record_q_history and (ep % config.record_q_every == 0):
            q1_hist.append((Q1[0], Q1[1]))
            q2_hist.append((Q2[0], Q2[1]))
            episode_marks.append(ep)

    coop_rate = coop_count / config.episodes

    # Build a detailed stats dict for this run
    stats: Dict[str, Any] = {
        "n_episodes": config.episodes,
        "noise_p": config.noise_p,
        "alpha": config.alpha,
        "epsilon": config.epsilon,
        "coop_rate": coop_rate,
        "outcomes": outcome_counts,
        "agent1": {
            "intended_counts": intended_counts_1,
            "executed_counts": executed_counts_1,
            "final_Q": {"Stag": float(Q1[0]), "Hare": float(Q1[1])},
            "preferred_action": "Stag" if Q1[0] >= Q1[1] else "Hare",
        },
        "agent2": {
            "intended_counts": intended_counts_2,
            "executed_counts": executed_counts_2,
            "final_Q": {"Stag": float(Q2[0]), "Hare": float(Q2[1])},
            "preferred_action": "Stag" if Q2[0] >= Q2[1] else "Hare",
        },
    }

    if record_q_history:
        return (
            coop_rate,
            stats,
            np.array(q1_hist),
            np.array(q2_hist),
            np.array(episode_marks),
        )
    else:
        return coop_rate, stats, None, None, None
