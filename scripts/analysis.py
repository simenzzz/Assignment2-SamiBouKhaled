import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
GRID_SIZES = [4, 8, 16]
PROC_COUNTS = [1, 2, 4]
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# Obstacle / terminal cells (mirrors the C files)
OBSTACLES = {
    4:  {2, 5},
    8:  {10, 21, 42, 53},
    16: {18, 37, 74, 93, 122, 157, 198, 221},
}
GOAL = {4: 15, 8: 63, 16: 255}
TRAPS = {
    4:  {11},
    8:  {58, 62},
    16: {238, 251, 254},
}


def parse_csv(filepath):
    """Parse a result file and return a dict with all sections."""
    data = {}
    episode_rewards = []
    policy = {}
    action_dist = {}
    avg_reward = {}       # state -> [up, down, left, right]
    state_stats = {}      # state -> {optimal, visits, opt_count, pct}

    mode = None
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line in {"EPISODE_REWARDS", "POLICY", "ACTION_DIST",
                        "AVG_REWARD", "STATE_STATS"}:
                mode = line
                continue
            if line.startswith("END_"):
                mode = None
                continue

            if mode == "EPISODE_REWARDS":
                parts = line.split(",")
                episode_rewards.append((int(parts[0]), float(parts[1])))
            elif mode == "POLICY":
                parts = line.split(",")
                policy[int(parts[0])] = parts[1]
            elif mode == "ACTION_DIST":
                parts = line.split(",")
                action_dist[parts[0]] = int(parts[1])
            elif mode == "AVG_REWARD":
                parts = line.split(",")
                state = int(parts[0])
                avg_reward[state] = [float(x) for x in parts[1:5]]
            elif mode == "STATE_STATS":
                parts = line.split(",")
                state_stats[int(parts[0])] = {
                    "optimal":   parts[1],
                    "visits":    int(parts[2]),
                    "opt_count": int(parts[3]),
                    "pct":       float(parts[4]),
                }
            elif line.startswith("---"):
                continue
            else:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    data[parts[0]] = parts[1]

    data["episode_rewards"] = episode_rewards
    data["policy"] = policy
    data["action_dist"] = action_dist
    data["avg_reward"] = avg_reward
    data["state_stats"] = state_stats
    return data


def smooth_rewards(rewards, window=50):
    smoothed = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        chunk = rewards[start:i + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def plot_episode_rewards():
    """Plot smoothed reward per episode for sequential / OpenMP / MPI."""
    for gs in GRID_SIZES:
        fig, ax = plt.subplots(figsize=(10, 6))

        for tag, label, fname in [
            ("seq", "Sequential", f"seq_{gs}.csv"),
            ("omp", "OpenMP (4 threads)", f"omp_{gs}_t4.csv"),
            ("mpi", "MPI (4 processes)", f"mpi_{gs}_p4.csv"),
        ]:
            path = os.path.join(RESULTS_DIR, fname)
            if not os.path.exists(path):
                continue
            d = parse_csv(path)
            rewards = [r for _, r in d["episode_rewards"]]
            smoothed = smooth_rewards(rewards)
            ax.plot(range(len(smoothed)), smoothed, label=label, linewidth=1.5)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Average reward (50-episode moving average)")
        ax.set_title(f"Episode rewards — {gs}x{gs} grid")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"rewards_{gs}x{gs}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"saved {out}")


def plot_optimal_action_percentage():
    """Per-state bar chart: x=state, y=OptimalActionPercentage[s] in %."""
    for gs in GRID_SIZES:
        path = os.path.join(RESULTS_DIR, f"seq_{gs}.csv")
        if not os.path.exists(path):
            continue
        d = parse_csv(path)
        stats = d["state_stats"]
        total = gs * gs

        states = list(range(total))
        pcts = [stats.get(s, {}).get("pct", 0.0) for s in states]
        colors = []
        for s in states:
            if s in OBSTACLES[gs]:
                colors.append("#888888")         # obstacle — no visits possible
            elif s == GOAL[gs]:
                colors.append("#2ca02c")         # goal
            elif s in TRAPS[gs]:
                colors.append("#d62728")         # trap
            else:
                colors.append("#1f77b4")

        # cap width so the 16x16 grid does not produce a 70-inch figure
        width = min(20, max(10, total * 0.25))
        fig, ax = plt.subplots(figsize=(width, 5))
        ax.bar(states, pcts, color=colors)

        ax.set_xlabel("State")
        ax.set_ylabel("OptimalActionPercentage (%)")
        ax.set_title(
            f"Optimal action selection per state — {gs}x{gs} grid\n"
            "(blue=empty, green=goal, red=trap, grey=obstacle)"
        )
        ax.set_ylim(0, 100)
        if total <= 64:
            ax.set_xticks(states)
        else:
            step = total // 32
            ax.set_xticks(range(0, total, step))
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"opt_action_pct_{gs}x{gs}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"saved {out}")


def plot_speedup_and_efficiency():
    """Speedup and efficiency in a single figure per grid size."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for col, gs in enumerate(GRID_SIZES):
        seq_time = float(parse_csv(os.path.join(RESULTS_DIR, f"seq_{gs}.csv"))["TIME"])

        omp_s, mpi_s = [], []
        for n in PROC_COUNTS:
            omp_s.append(seq_time / float(parse_csv(
                os.path.join(RESULTS_DIR, f"omp_{gs}_t{n}.csv"))["TIME"]))
            mpi_s.append(seq_time / float(parse_csv(
                os.path.join(RESULTS_DIR, f"mpi_{gs}_p{n}.csv"))["TIME"]))

        ax = axes[0, col]
        ax.plot(PROC_COUNTS, omp_s, "o-", label="OpenMP", linewidth=2)
        ax.plot(PROC_COUNTS, mpi_s, "s-", label="MPI", linewidth=2)
        ax.plot(PROC_COUNTS, PROC_COUNTS, "--", color="gray",
                label="Ideal", alpha=0.5)
        ax.set_xlabel("Threads / Processes (N)")
        ax.set_ylabel("Speedup  S = T_seq / T_par")
        ax.set_title(f"Speedup — {gs}x{gs}")
        ax.set_xticks(PROC_COUNTS)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        omp_e = [s / n for s, n in zip(omp_s, PROC_COUNTS)]
        mpi_e = [s / n for s, n in zip(mpi_s, PROC_COUNTS)]
        ax.plot(PROC_COUNTS, omp_e, "o-", label="OpenMP", linewidth=2)
        ax.plot(PROC_COUNTS, mpi_e, "s-", label="MPI", linewidth=2)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Ideal")
        ax.set_xlabel("Threads / Processes (N)")
        ax.set_ylabel("Efficiency  E = S / N")
        ax.set_title(f"Efficiency — {gs}x{gs}")
        ax.set_xticks(PROC_COUNTS)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "speedup.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def print_timing_table():
    print("\n=== Execution time (seconds, averaged) ===\n")
    print(f"{'Version':<20} {'4x4':>12} {'8x8':>12} {'16x16':>12}")
    print("-" * 58)

    times = [float(parse_csv(os.path.join(RESULTS_DIR, f"seq_{g}.csv"))["TIME"])
             for g in GRID_SIZES]
    print(f"{'Sequential':<20} {times[0]:>12.6f} {times[1]:>12.6f} {times[2]:>12.6f}")

    for n in PROC_COUNTS:
        ts = [float(parse_csv(os.path.join(RESULTS_DIR, f"omp_{g}_t{n}.csv"))["TIME"])
              for g in GRID_SIZES]
        print(f"{'OpenMP ('+str(n)+'t)':<20} {ts[0]:>12.6f} {ts[1]:>12.6f} {ts[2]:>12.6f}")

    for n in PROC_COUNTS:
        ts = [float(parse_csv(os.path.join(RESULTS_DIR, f"mpi_{g}_p{n}.csv"))["TIME"])
              for g in GRID_SIZES]
        print(f"{'MPI ('+str(n)+'p)':<20} {ts[0]:>12.6f} {ts[1]:>12.6f} {ts[2]:>12.6f}")


def write_markdown_tables():
    """Emit tables/tables.md with the rubric-required tables."""
    out_path = os.path.join(RESULTS_DIR, "tables.md")
    lines = []

    # Timing + speedup + efficiency
    lines.append("## Execution time, speedup, and efficiency\n")
    lines.append("| Version | 4x4 (s) | 8x8 (s) | 16x16 (s) "
                 "| Speedup 4x4 | Speedup 8x8 | Speedup 16x16 "
                 "| Eff 4x4 | Eff 8x8 | Eff 16x16 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    seq_times = {g: float(parse_csv(os.path.join(RESULTS_DIR, f"seq_{g}.csv"))["TIME"])
                 for g in GRID_SIZES}
    lines.append(
        f"| Sequential | {seq_times[4]:.6f} | {seq_times[8]:.6f} "
        f"| {seq_times[16]:.6f} | 1.00× | 1.00× | 1.00× | 1.00 | 1.00 | 1.00 |"
    )

    for label_prefix, file_prefix in [("OpenMP", "omp"), ("MPI", "mpi")]:
        suffix = "t" if file_prefix == "omp" else "p"
        for n in PROC_COUNTS:
            ts = {g: float(parse_csv(os.path.join(
                    RESULTS_DIR, f"{file_prefix}_{g}_{suffix}{n}.csv"))["TIME"])
                  for g in GRID_SIZES}
            s = {g: seq_times[g] / ts[g] for g in GRID_SIZES}
            e = {g: s[g] / n for g in GRID_SIZES}
            lines.append(
                f"| {label_prefix} ({n}{suffix}) "
                f"| {ts[4]:.6f} | {ts[8]:.6f} | {ts[16]:.6f} "
                f"| {s[4]:.2f}× | {s[8]:.2f}× | {s[16]:.2f}× "
                f"| {e[4]:.2f} | {e[8]:.2f} | {e[16]:.2f} |"
            )
    lines.append("")

    # 4x4 AvgReward table (full)
    lines.append("## AvgReward[s][a] — 4x4 grid (sequential run)\n")
    lines.append("| State | UP | DOWN | LEFT | RIGHT |")
    lines.append("|---|---|---|---|---|")
    d4 = parse_csv(os.path.join(RESULTS_DIR, "seq_4.csv"))
    for s in range(16):
        av = d4["avg_reward"].get(s, [0.0] * 4)
        lines.append(
            f"| {s} | {av[0]:.4f} | {av[1]:.4f} | {av[2]:.4f} | {av[3]:.4f} |"
        )
    lines.append("")

    # 4x4 optimal policy in grid layout
    lines.append("## Optimal policy — 4x4 grid\n")
    pol4 = d4["policy"]
    lines.append("Rendered as a 4x4 grid (OBST = obstacle, GOAL, TRAP):\n")
    lines.append("```")
    for row in range(4):
        cells = []
        for col in range(4):
            s = row * 4 + col
            if s in OBSTACLES[4]:
                cells.append("OBST ")
            elif s == GOAL[4]:
                cells.append("GOAL ")
            elif s in TRAPS[4]:
                cells.append("TRAP ")
            else:
                cells.append(f"{pol4[s]:<5}")
        lines.append("  ".join(cells))
    lines.append("```\n")

    # 8x8 and 16x16 as compact tables
    for gs in (8, 16):
        lines.append(f"## Optimal policy — {gs}x{gs} grid\n")
        d = parse_csv(os.path.join(RESULTS_DIR, f"seq_{gs}.csv"))
        pol = d["policy"]
        lines.append("```")
        for row in range(gs):
            cells = []
            for col in range(gs):
                s = row * gs + col
                if s in OBSTACLES[gs]:
                    cells.append("OBST")
                elif s == GOAL[gs]:
                    cells.append("GOAL")
                elif s in TRAPS[gs]:
                    cells.append("TRAP")
                else:
                    abbr = {"UP": " UP ", "DOWN": "DOWN", "LEFT": "LEFT", "RIGHT": "RGHT"}[pol[s]]
                    cells.append(abbr)
            lines.append(" ".join(cells))
        lines.append("```\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"saved {out_path}")


if __name__ == "__main__":
    print("generating plots and tables...")
    plot_episode_rewards()
    plot_optimal_action_percentage()
    plot_speedup_and_efficiency()
    write_markdown_tables()
    print_timing_table()
    print("done.")
