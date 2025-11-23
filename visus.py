#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib

# Use non-interactive backend for headless execution
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="tab10")

# Load file
txt = Path("tournament_results.txt").read_text(encoding="utf-8")
lines = txt.splitlines()

records = []
current_ai = None

# Regex to parse standings
standings_re = re.compile(
    r"\s*\d+\.\s+(?P<agent>.+?)\s+Wins:\s+(?P<wins>\d+)\s+\|\s+Win%:\s+(?P<win_pct>[\d\.]+)%\s+\|\s+AvgPlace:\s+(?P<avg_place>[\d\.]+)"
)

i = 0
while i < len(lines):
    line = lines[i].strip()

    # Detect AI blocks
    if "Pobablistic" in line:
        current_ai = "Probabilistic"

    elif "Heuristic Probabilistic Attacker" in line:
        current_ai = "Heuristic Probabilistic Attacker"

    # Detect strategy section
    elif line.startswith("="):
        if i + 1 >= len(lines):
            break

        strategy = lines[i + 1].strip()
        if not strategy or strategy.startswith("="):
            i += 1
            continue

        i += 2
        meta = {"ai_block": current_ai, "strategy": strategy}

        # Parse metadata and standings
        while i < len(lines):
            l = lines[i].strip()

            if l.startswith("Tournament date"):
                meta["tournament_date"] = l.split(":", 1)[1].strip()

            elif l.startswith("Total games"):
                meta["total_games"] = int(l.split(":", 1)[1])

            elif l.startswith("Finished"):
                meta["finished"] = int(l.split(":", 1)[1])

            elif l.startswith("Draws"):
                meta["draws"] = int(l.split(":", 1)[1])

            elif l.startswith("Average turns"):
                meta["avg_turns"] = float(l.split(":", 1)[1])

            elif l.startswith("Average duration"):
                meta["avg_duration_s"] = float(
                    l.split(":", 1)[1].strip().rstrip("s")
                )

            elif l.startswith("Total time"):
                meta["total_time_min"] = float(
                    l.split(":", 1)[1].split()[0]
                )

            elif l.startswith("Speed"):
                meta["speed_games_per_s"] = float(
                    l.split(":", 1)[1].split()[0]
                )

            elif l.startswith("Final standings"):
                i += 1
                while i < len(lines) and lines[i].strip():
                    m = standings_re.match(lines[i])
                    if m:
                        rec = meta.copy()
                        rec.update(
                            {
                                "agent": m.group("agent").strip(),
                                "wins": int(m.group("wins")),
                                "win_pct": float(m.group("win_pct")),
                                "avg_place": float(m.group("avg_place")),
                            }
                        )
                        records.append(rec)
                    i += 1
                break

            i += 1

    i += 1

# Convert to dataframe
df = pd.DataFrame(records)
if df.empty:
    raise SystemExit("No data parsed from tournament_results.txt")

# Metrics
df["draw_pct"] = (df["draws"] / df["total_games"] * 100).round(2)
df["finished_pct"] = (df["finished"] / df["total_games"] * 100).round(2)

df["strategy_clean"] = (
    df["strategy"]
    .str.replace("Baseline :", "Baseline", regex=False)
    .str.replace("&", "+")
    .str.strip()
)

df["agent_clean"] = (
    df["agent"]
    .str.replace("random, fully", "random", regex=False)
    .str.strip()
)

# Detect main AI (the non-random agent)
df["is_main_ai"] = ~df["agent_clean"].str.contains("random", case=False)

main_df = df[df["is_main_ai"]].sort_values(["ai_block", "win_pct"])

# Generate plots for each AI block automatically
for ai in main_df["ai_block"].dropna().unique():
    subset = main_df[main_df["ai_block"] == ai].copy()

    order = subset.sort_values("win_pct")["strategy_clean"]
    subset["strategy_clean"] = pd.Categorical(
        subset["strategy_clean"], categories=order, ordered=True
    )
    subset_sorted = subset.sort_values("strategy_clean")

    # Baseline detection
    baseline_rows = subset_sorted[
        subset_sorted["strategy_clean"].str.contains("baseline", case=False)
    ]
    baseline_win = baseline_rows["win_pct"].iloc[0] if not baseline_rows.empty else None

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=subset_sorted, x="win_pct", y="strategy_clean", ax=ax)

    ax.set_title(f"Win% {ai}")
    ax.set_xlabel("Taux de victoire (%)")
    ax.set_ylabel("Stratégie testée")

    # Labels with deltas
    for bar, (_, row) in zip(ax.patches, subset_sorted.iterrows()):
        val = bar.get_width()
        label = f"{val:.1f}%"
        if baseline_win is not None:
            delta = val - baseline_win
            label += f" ({delta:+.1f}%)"

        ax.text(
            val + 0.5,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=9,
        )

    fig.suptitle("Taux de victoire du moteur testé par stratégie", fontsize=14)
    fig.tight_layout()

    out_file = f"main_ai_winrates_{ai.lower().replace(' ', '_')}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

print("✔️ Graphiques générés avec succès !")
