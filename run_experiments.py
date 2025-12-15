# run_experiments.py
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from stag_game import (
    QLearningConfig,
    run_two_qlearners,
    play_stag_hunt,
    epsilon_greedy,
)


def determine_equilibrium_label(coop_rate_mean: float, threshold: float = 0.8) -> str:
    """
    Simple heuristic to label the equilibrium type based on cooperation rate.
    """
    if coop_rate_mean >= threshold:
        return "payoff_dominant"  # mostly Stag–Stag
    elif coop_rate_mean <= 1.0 - threshold:
        return "risk_dominant"    # mostly Hare–Hare
    else:
        return "mixed_or_unclear"


def sweep_noise_alpha_epsilon(
    noise_values,
    low_alpha=0.1,
    high_alpha=0.5,
    low_epsilon=0.05,
    high_epsilon=0.3,
    episodes=10_000,
    n_runs_per_combo=1,
    seed=0,
):
    """
    For each noise level, run 2 Q-learning agents for all 4 combos of:
      (low alpha, low epsilon),
      (low alpha, high epsilon),
      (high alpha, low epsilon),
      (high alpha, high epsilon).

    Returns:
        results: dict mapping config_name -> {
            'alpha', 'epsilon',
            'noise_stats': {
                str(noise_p): {
                    ... detailed aggregated stats ...
                },
                ...
            }
        }
    """
    rng_master = np.random.default_rng(seed)

    # Define the four (alpha, epsilon) configs
    configs = {
        "low_a_low_e":  (low_alpha, low_epsilon),
        "low_a_high_e": (low_alpha, high_epsilon),
        "high_a_low_e": (high_alpha, low_epsilon),
        "high_a_high_e":(high_alpha, high_epsilon),
    }

    # Initialize result containers
    results = {}
    for name, (alpha, epsilon) in configs.items():
        results[name] = {
            "alpha": alpha,
            "epsilon": epsilon,
            "noise_stats": {},  # filled per noise level
        }

    # Sweep over noise values
    for noise_p in noise_values:
        print(f"\n=== Noise p = {noise_p:.2f} ===")
        noise_key = f"{noise_p:.3f}"

        for name, (alpha, epsilon) in configs.items():
            coop_rates = []

            # Aggregate counts over runs
            total_outcomes = {"SS": 0, "SH": 0, "HS": 0, "HH": 0}
            total_intended_1 = {"Stag": 0, "Hare": 0}
            total_intended_2 = {"Stag": 0, "Hare": 0}
            total_executed_1 = {"Stag": 0, "Hare": 0}
            total_executed_2 = {"Stag": 0, "Hare": 0}
            final_Q1_list = []
            final_Q2_list = []

            for _ in range(n_runs_per_combo):
                cfg = QLearningConfig(
                    episodes=episodes,
                    alpha=alpha,
                    gamma=0.0,
                    epsilon=epsilon,
                    noise_p=noise_p,
                )
                rng = np.random.default_rng(rng_master.integers(0, 1_000_000))
                coop_rate, stats, _, _, _ = run_two_qlearners(
                    cfg, rng, record_q_history=False
                )
                coop_rates.append(coop_rate)

                # Aggregate outcomes and counts
                for key in total_outcomes:
                    total_outcomes[key] += stats["outcomes"][key]

                for k in total_intended_1:
                    total_intended_1[k] += stats["agent1"]["intended_counts"][k]
                    total_intended_2[k] += stats["agent2"]["intended_counts"][k]
                    total_executed_1[k] += stats["agent1"]["executed_counts"][k]
                    total_executed_2[k] += stats["agent2"]["executed_counts"][k]

                final_Q1_list.append(stats["agent1"]["final_Q"])
                final_Q2_list.append(stats["agent2"]["final_Q"])

            coop_rates = np.array(coop_rates)
            mean_coop = float(coop_rates.mean())
            std_coop = float(coop_rates.std()) if len(coop_rates) > 1 else 0.0

            print(
                f"  {name:12s} (alpha={alpha:.2f}, eps={epsilon:.2f}) "
                f"=> mean coop={mean_coop:.3f}, std={std_coop:.3f}"
            )

            # Average outcome counts over runs
            avg_outcomes = {
                k: total_outcomes[k] / n_runs_per_combo for k in total_outcomes
            }

            # Average counts and final Q-values
            avg_intended_1 = {
                k: total_intended_1[k] / n_runs_per_combo for k in total_intended_1
            }
            avg_intended_2 = {
                k: total_intended_2[k] / n_runs_per_combo for k in total_intended_2
            }
            avg_executed_1 = {
                k: total_executed_1[k] / n_runs_per_combo for k in total_executed_1
            }
            avg_executed_2 = {
                k: total_executed_2[k] / n_runs_per_combo for k in total_executed_2
            }

            # Average final Q across runs
            def avg_final_Q(Q_list):
                if not Q_list:
                    return {"Stag": 0.0, "Hare": 0.0}
                stag_vals = [q["Stag"] for q in Q_list]
                hare_vals = [q["Hare"] for q in Q_list]
                return {
                    "Stag": float(np.mean(stag_vals)),
                    "Hare": float(np.mean(hare_vals)),
                }

            avg_Q1 = avg_final_Q(final_Q1_list)
            avg_Q2 = avg_final_Q(final_Q2_list)

            # Determine equilibrium label from mean_coop
            equilibrium = determine_equilibrium_label(mean_coop, threshold=0.8)

            results[name]["noise_stats"][noise_key] = {
                "noise_p": noise_p,
                "n_runs": n_runs_per_combo,
                "n_episodes": episodes,
                "coop_rate_mean": mean_coop,
                "coop_rate_std": std_coop,
                "avg_outcomes": avg_outcomes,
                "agent1": {
                    "avg_intended_counts": avg_intended_1,
                    "avg_executed_counts": avg_executed_1,
                    "avg_final_Q": avg_Q1,
                    "preferred_action":
                        "Stag" if avg_Q1["Stag"] >= avg_Q1["Hare"] else "Hare",
                },
                "agent2": {
                    "avg_intended_counts": avg_intended_2,
                    "avg_executed_counts": avg_executed_2,
                    "avg_final_Q": avg_Q2,
                    "preferred_action":
                        "Stag" if avg_Q2["Stag"] >= avg_Q2["Hare"] else "Hare",
                },
                "equilibrium": equilibrium,
            }

    return results


def plot_coop_vs_noise_multi(results, output_dir: Path, filename="coop_vs_noise_multi.png"):
    """
    Plot cooperation vs noise for each (alpha, epsilon) combination.
    """
    plt.figure(figsize=(7, 5))

    for name, data in results.items():
        noise_vals = []
        coop_means = []

        # noise_stats keys are strings like "0.000", "0.050"
        for noise_key, ns in sorted(
            data["noise_stats"].items(), key=lambda kv: float(kv[0])
        ):
            noise_vals.append(float(noise_key))
            coop_means.append(ns["coop_rate_mean"])

        alpha = data["alpha"]
        epsilon = data["epsilon"]
        label = f"{name} (α={alpha}, ε={epsilon})"

        plt.plot(noise_vals, coop_means, "-o", label=label)

    plt.xlabel("Action noise probability p")
    plt.ylabel("Cooperation rate (Stag–Stag frequency)")
    plt.title("Cooperation vs Noise for Different (α, ε) Settings")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = output_dir / filename
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def run_example_q_trajectory(
    noise_p=0.1,
    alpha=0.1,
    epsilon=0.1,
    episodes=10_000,
    seed=42,
):
    """
    Run a single example run for one (alpha, epsilon, noise) combo
    and record Q-values over time for plotting.
    """
    cfg = QLearningConfig(
        episodes=episodes,
        alpha=alpha,
        gamma=0.0,
        epsilon=epsilon,
        noise_p=noise_p,
        record_q_every=10,
    )
    rng = np.random.default_rng(seed)
    coop_rate, stats, q1_hist, q2_hist, episode_marks = run_two_qlearners(
        cfg, rng, record_q_history=True
    )
    print(
        f"\nExample run: noise={noise_p:.2f}, alpha={alpha:.2f}, eps={epsilon:.2f}"
    )
    print(f"  Cooperation rate: {coop_rate:.3f}")
    print(f"  Outcomes: {stats['outcomes']}")
    return coop_rate, stats, q1_hist, q2_hist, episode_marks


def plot_q_trajectory(
    q1_hist,
    q2_hist,
    episode_marks,
    output_dir: Path,
    filename="q_values_example.png",
):
    plt.figure(figsize=(7, 4))

    plt.plot(episode_marks, q1_hist[:, 0], label="Agent 1: Q(Stag)")
    plt.plot(episode_marks, q1_hist[:, 1], label="Agent 1: Q(Hare)", linestyle="--")
    plt.plot(episode_marks, q2_hist[:, 0], label="Agent 2: Q(Stag)")
    plt.plot(episode_marks, q2_hist[:, 1], label="Agent 2: Q(Hare)", linestyle=":")

    plt.xlabel("Episode")
    plt.ylabel("Q-value")
    plt.title("Q-value Trajectories in Stag Hunt (Example Run)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / filename
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# ==== NEW: Time series of cooperation rate ====

def plot_coop_timeseries(
    alpha=0.1,
    epsilon=0.1,
    noise_p=0.1,
    episodes=10_000,
    window=200,
    seed=123,
    out_path=None,
):
    """
    Run a single Q-learning run and plot the moving average
    of cooperation (Stag–Stag) over time.
    """
    rng = np.random.default_rng(seed)

    Q1 = np.zeros(2, dtype=float)
    Q2 = np.zeros(2, dtype=float)

    coop_flags = []

    for ep in range(episodes):
        a1_int = epsilon_greedy(Q1, epsilon, rng)
        a2_int = epsilon_greedy(Q2, epsilon, rng)

        (a1_exec, a2_exec), (r1, r2) = play_stag_hunt(a1_int, a2_int, noise_p, rng)

        coop_flags.append(1 if (a1_exec == 0 and a2_exec == 0) else 0)

        Q1[a1_int] += alpha * (r1 - Q1[a1_int])
        Q2[a2_int] += alpha * (r2 - Q2[a2_int])

    coop_flags = np.array(coop_flags, dtype=float)

    if window > 1:
        kernel = np.ones(window) / window
        moving = np.convolve(coop_flags, kernel, mode="valid")
        x = np.arange(window - 1, episodes)
    else:
        moving = coop_flags
        x = np.arange(episodes)

    plt.figure(figsize=(7, 4))
    plt.plot(x, moving)
    plt.ylim(0, 1.0)
    plt.xlabel("Episode")
    plt.ylabel(f"Cooperation rate (window={window})")
    plt.title(
        f"Cooperation over time (α={alpha}, ε={epsilon}, noise={noise_p})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=200)
        print(f"Saved {out_path}")
    plt.close()


# ==== NEW: Equilibrium map (α–ε grid) ====

def plot_equilibrium_map(summary, noise_p=0.0, output_dir: Path = Path("."), filename="equilibrium_map.png"):
    """
    Make a 2x2 grid for (low/high alpha, low/high epsilon) at a fixed noise_p.
    Cell colors show payoff-dominant / risk-dominant / mixed.
    """
    noise_key = f"{noise_p:.3f}"

    # order: rows = alpha (low, high), cols = epsilon (low, high)
    config_order = [
        ("low_a_low_e", 0, 0),
        ("low_a_high_e", 0, 1),
        ("high_a_low_e", 1, 0),
        ("high_a_high_e", 1, 1),
    ]

    label_to_val = {
        "payoff_dominant": 1,
        "risk_dominant": -1,
        "mixed_or_unclear": 0,
    }
    cmap = {
        1: "#4c72b0",   # blue
        0: "#bbbbbb",   # gray
        -1: "#dd8452",  # orange/red
    }

    grid = np.zeros((2, 2), dtype=int)
    text_labels = [["", ""], ["", ""]]

    for name, i, j in config_order:
        ns = summary[name]["noise_stats"].get(noise_key)
        if ns is None:
            val = 0
            txt = "N/A"
        else:
            eq = ns["equilibrium"]
            val = label_to_val.get(eq, 0)
            txt = {
                "payoff_dominant": "Payoff",
                "risk_dominant": "Risk",
                "mixed_or_unclear": "Mixed",
            }.get(eq, "Mixed")
        grid[i, j] = val
        text_labels[i][j] = txt

    plt.figure(figsize=(4, 4))
    ax = plt.gca()
    for i in range(2):
        for j in range(2):
            v = grid[i, j]
            ax.add_patch(
                plt.Rectangle((j, 1 - i), 1, 1, color=cmap[v], ec="black")
            )
            plt.text(
                j + 0.5,
                1 - i + 0.5,
                text_labels[i][j],
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )

    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.xticks([0.5, 1.5], ["ε low", "ε high"])
    plt.yticks([0.5, 1.5], ["α high", "α low"])  # flipped for intuitive reading
    plt.title(f"Equilibrium map at noise={noise_p}")
    ax.set_aspect("equal", "box")
    plt.tight_layout()

    out_path = output_dir / filename
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# ==== NEW: Action distribution plot ====

def plot_action_distribution(summary, noise_p=0.1, output_dir: Path = Path("."), filename="action_dist.png"):
    """
    For each config, show fraction of executed Stag vs Hare actions
    (averaged over the two agents) at a fixed noise level.
    """
    noise_key = f"{noise_p:.3f}"

    config_names = ["low_a_low_e", "low_a_high_e", "high_a_low_e", "high_a_high_e"]
    labels = [
        "lowα, lowε",
        "lowα, highε",
        "highα, lowε",
        "highα, highε",
    ]

    stag_fracs = []
    hare_fracs = []

    for name in config_names:
        ns = summary[name]["noise_stats"].get(noise_key)
        if ns is None:
            stag_fracs.append(0.0)
            hare_fracs.append(1.0)
            continue

        e1 = ns["agent1"]["avg_executed_counts"]
        e2 = ns["agent2"]["avg_executed_counts"]

        stag = e1["Stag"] + e2["Stag"]
        hare = e1["Hare"] + e2["Hare"]
        total = stag + hare if (stag + hare) > 0 else 1.0

        stag_fracs.append(stag / total)
        hare_fracs.append(hare / total)

    x = np.arange(len(config_names))
    width = 0.6

    plt.figure(figsize=(7, 4))
    plt.bar(x, stag_fracs, width, label="Stag", alpha=0.8)
    plt.bar(x, hare_fracs, width, bottom=stag_fracs, label="Hare", alpha=0.8)

    plt.xticks(x, labels, rotation=20)
    plt.ylim(0, 1.0)
    plt.ylabel("Fraction of executed actions")
    plt.title(f"Action distribution at noise={noise_p}")
    plt.legend()
    plt.tight_layout()

    out_path = output_dir / filename
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    # Noise levels to test
    noise_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # === Create results/run_YYYY-MM-DD_HH-MM-SS/ ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_dir = Path("results") / f"run_{timestamp}"
    root_dir.mkdir(parents=True, exist_ok=True)

    # Meta info for this experiment
    config_meta = {
        "noise_values": noise_values,
        "low_alpha": 0.1,
        "high_alpha": 0.5,
        "low_epsilon": 0.05,
        "high_epsilon": 0.3,
        "episodes": 10_000,
        "n_runs_per_combo": 1,
        "seed": 0,
    }
    with open(root_dir / "config_meta.json", "w") as f:
        json.dump(config_meta, f, indent=2)
    print(f"Saved {root_dir / 'config_meta.json'}")

    # === Run the big sweep over noise + (alpha, epsilon) configs ===
    results = sweep_noise_alpha_epsilon(
        noise_values,
        low_alpha=config_meta["low_alpha"],
        high_alpha=config_meta["high_alpha"],
        low_epsilon=config_meta["low_epsilon"],
        high_epsilon=config_meta["high_epsilon"],
        episodes=config_meta["episodes"],
        n_runs_per_combo=config_meta["n_runs_per_combo"],
        seed=config_meta["seed"],
    )

    # Save summary.json
    summary_path = root_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {summary_path}")

    # === Plot 4 curves on one figure ===
    plot_coop_vs_noise_multi(results, root_dir, filename="coop_vs_noise_multi.png")

    # === Q-value trajectory example ===
    coop_rate, stats, q1_hist, q2_hist, episode_marks = run_example_q_trajectory(
        noise_p=0.1,
        alpha=0.1,
        epsilon=0.1,
        episodes=config_meta["episodes"],
        seed=123,
    )
    if q1_hist is not None:
        plot_q_trajectory(
            q1_hist,
            q2_hist,
            episode_marks,
            root_dir,
            filename="q_values_example_default.png",
        )

    # === NEW plots ===

    # Time series: low alpha, low epsilon, noise=0.1
    plot_coop_timeseries(
        alpha=config_meta["low_alpha"],
        epsilon=config_meta["low_epsilon"],
        noise_p=0.1,
        episodes=config_meta["episodes"],
        window=200,
        seed=123,
        out_path=root_dir / "coop_timeseries_lowA_lowE_p0.1.png",
    )

    # Time series: high alpha, high epsilon, noise=0.1
    plot_coop_timeseries(
        alpha=config_meta["high_alpha"],
        epsilon=config_meta["high_epsilon"],
        noise_p=0.1,
        episodes=config_meta["episodes"],
        window=200,
        seed=456,
        out_path=root_dir / "coop_timeseries_highA_highE_p0.1.png",
    )

    # Equilibrium maps at noise=0.0 and 0.1
    plot_equilibrium_map(
        results,
        noise_p=0.0,
        output_dir=root_dir,
        filename="equilibrium_map_p0.0.png",
    )
    plot_equilibrium_map(
        results,
        noise_p=0.1,
        output_dir=root_dir,
        filename="equilibrium_map_p0.1.png",
    )

    # Action distribution at noise=0.1
    plot_action_distribution(
        results,
        noise_p=0.1,
        output_dir=root_dir,
        filename="action_dist_p0.1.png",
    )
