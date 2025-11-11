"""
Simplest possible Risk AI tournament runner — with live progress counter and text results.

Usage:
  python tournament_simple.py --ais 3 1 1 2 --games 200 --max-turns 500
"""

import argparse
import random
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# --- import your project bits ---
from model.world import World
from model.state import State
from model.state_informations import undefeated_armies, army_units, territories_occupied_by_army
from boards.polytech_board import risk_map
from game.game import game
from game.setup import random_initial_state

from ais.neutrals import strategy_neutral_fully_random, strategy_neutral_uniform_random
from ais.randoms import strategy_fully_random, strategy_uniform_random
from ais.probabilistic import strategy_probabilistic
from ais.balanced_aggressor import strategy_balanced_aggressor

# Map numeric ids -> strategies
AIs = {
    -2: strategy_neutral_fully_random,
    -1: strategy_neutral_uniform_random,
     1: strategy_fully_random,
     2: strategy_uniform_random,
     3: strategy_probabilistic,
     4: strategy_balanced_aggressor,
}

# ----------------------------
# Helpers
# ----------------------------

def balanced_seatings(ai_indices: List[int], num_games: int) -> List[List[int]]:
    """Each AI visits each seat roughly equally across games."""
    S = len(ai_indices)
    base = ai_indices[:]
    plans: List[List[int]] = []
    reps = num_games // S + (num_games % S != 0)
    for _ in range(reps):
        random.shuffle(base)
        for shift in range(S):
            plan = base[shift:] + base[:shift]
            plans.append(plan[:])
    return plans[:num_games]

def assign_unique_names(ai_indices: List[int]) -> List[str]:
    """Give duplicate strategies unique names like 'Name #2'."""
    base = [AIs[i].name for i in ai_indices]
    total = Counter(base)
    seen = defaultdict(int)
    out = []
    for b in base:
        seen[b] += 1
        out.append(b if total[b] == 1 else f"{b} #{seen[b]}")
    return out

# ----------------------------
# One game
# ----------------------------

def play_one_game(ai_order: List[int], max_turns: int) -> Tuple[Optional[str], Dict[str, int], int, float, bool]:
    """Runs a single game and returns (winner, placements, turns, duration, ended)."""
    t0 = time.time()
    strategies = [AIs[idx] for idx in ai_order]
    world = World(strategies, *risk_map())

    # Assign names
    base_names = [s.name for s in strategies]
    total = Counter(base_names)
    seen = defaultdict(int)
    for army in world.armies:
        b = army.strategy.name
        seen[b] += 1
        label = b if total[b] == 1 else f"{b} #{seen[b]}"
        setattr(army, "display_name", label)

    # Run game
    s0 = random_initial_state(world)
    history = list(game(s0, max_turns))
    if not history:
        return None, {}, 0, 0.0, False

    final_state: State = history[-1][1]
    turns = len(history)

    # Evaluate players
    cards = []
    for army in world.armies:
        name = getattr(army, "display_name", army.strategy.name)
        terr = territories_occupied_by_army(final_state, army)
        units = army_units(final_state, army)
        alive = len(terr) > 0
        cards.append({"name": name, "alive": alive, "terr": len(terr), "units": units})

    cards.sort(key=lambda x: (x["alive"], x["terr"], x["units"], x["name"]), reverse=True)
    placements = {c["name"]: i + 1 for i, c in enumerate(cards)}

    # Detect end of game
    try:
        ended = (len(list(undefeated_armies(final_state))) == 1)
    except Exception:
        ended = sum(1 for c in cards if c["alive"]) == 1

    winner = cards[0]["name"] if ended else None
    duration = time.time() - t0

    return winner, placements, turns, duration, ended

# ----------------------------
# Stats
# ----------------------------

class Stats:
    def __init__(self, players: List[str]):
        self.players = players
        self.finished = 0
        self.draws = 0
        self.games = 0
        self.wins = {p: 0 for p in players}
        self.places = {p: [] for p in players}
        self.turns: List[int] = []
        self.durations: List[float] = []

    def add(self, winner: Optional[str], placements: Dict[str, int], turns: int, dur: float, ended: bool):
        self.games += 1
        self.turns.append(turns)
        self.durations.append(dur)
        if ended and winner:
            self.finished += 1
            self.wins[winner] += 1
        else:
            self.draws += 1
        for p, pos in placements.items():
            self.places[p].append(pos)

    def win_rate(self, p: str) -> float:
        return (self.wins[p] / self.finished * 100) if self.finished else 0.0

    def avg_place(self, p: str) -> float:
        q = self.places[p]
        return sum(q) / len(q) if q else 0.0

    def avg_turns(self) -> float:
        return sum(self.turns) / len(self.turns) if self.turns else 0.0

    def avg_duration(self) -> float:
        return sum(self.durations) / len(self.durations) if self.durations else 0.0

# ----------------------------
# Save results to file
# ----------------------------

def save_results_txt(stats: Stats, players: List[str], total_time: float, filename="tournament_results.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Tournament date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total games: {stats.games}\n")
        f.write(f"Finished: {stats.finished}\n")
        f.write(f"Draws: {stats.draws}\n")
        f.write(f"Average turns: {stats.avg_turns():.1f}\n")
        f.write(f"Average duration: {stats.avg_duration():.2f}s\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n")
        f.write(f"Speed: {stats.games/total_time:.1f} games/sec\n\n")
        f.write("Final standings:\n")
        order = sorted(players, key=lambda p: (stats.wins[p], -1.0 / (stats.avg_place(p)+1e-9)), reverse=True)
        for i, p in enumerate(order, 1):
            f.write(f"{i:>2}. {p:<25}  Wins: {stats.wins[p]:>3} | Win%: {stats.win_rate(p):>6.1f}% | AvgPlace: {stats.avg_place(p):>5.2f}\n")
    print(f"Results saved to '{filename}'")

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Simplest Risk AI Tournament (with live progress and text output)")
    ap.add_argument("--ais", type=int, nargs="+", required=True, help="AI indices (>=2 players)")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--max-turns", type=int, default=500)
    args = ap.parse_args()

    if any(i not in AIs for i in args.ais):
        bad = [i for i in args.ais if i not in AIs]
        raise SystemExit(f"Unknown AI ids: {bad}. Allowed: {sorted(AIs.keys())}")

    if len(args.ais) < 2:
        raise SystemExit("Need at least 2 AIs.")

    roster_names = assign_unique_names(args.ais)
    print("Players:")
    for i, (idx, name) in enumerate(zip(args.ais, roster_names), 1):
        print(f"  {i}. [{idx}] {name}")
    print(f"\nGames: {args.games} | Max turns: {args.max_turns}\n")

    stats = Stats(roster_names)
    seatings = balanced_seatings(args.ais, args.games)

    t0 = time.time()
    for g in range(args.games):
        winner, placements, turns, dur, ended = play_one_game(seatings[g], args.max_turns)
        stats.add(winner, placements, turns, dur, ended)

        done = g + 1
        pct = done / args.games * 100
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        print(f"\rProgress: {done}/{args.games} ({pct:5.1f}%) | Speed: {speed:5.2f} games/s", end="", flush=True)

    print()  # newline
    total = time.time() - t0

    print("\n=== Final Stats ===")
    print(f"Games: {stats.games} | Finished: {stats.finished} | Draws: {stats.draws}")
    print(f"Avg turns: {stats.avg_turns():.1f} | Avg duration: {stats.avg_duration():.2f}s | Speed: {stats.games/total:.1f} games/s")

    order = sorted(stats.players, key=lambda p: (stats.wins[p], -1.0 / (stats.avg_place(p) + 1e-9)), reverse=True)

    print("\nRank  Player                           Wins   Win%   AvgPlace")
    for i, p in enumerate(order, 1):
        print(f"{i:>4}  {p:<30}  {stats.wins[p]:>4}  {stats.win_rate(p):>6.1f}%   {stats.avg_place(p):>7.2f}")

    save_results_txt(stats, roster_names, total)

if __name__ == "__main__":
    main()
