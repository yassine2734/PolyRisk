#! /bin/python
# tournament_parallel.py
"""
Parallel Risk AI Tournament - Optimized for Apple Silicon (M5)

Usage:
  python tournament_parallel.py --ais 4 1 1 1 --games 500 --workers 8

Benefits:
  - Runs multiple games simultaneously
  - Utilizes all CPU cores
  - 5-10x faster on M5 MacBook Pro
"""

import sys
import time
import argparse
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from model.world import World
from model.state import State
from model.state_informations import undefeated_armies, army_units, territories_occupied_by_army
from boards.polytech_board import risk_map
from game.game import game
from game.setup import random_initial_state

from ais.neutrals import strategy_neutral_fully_random, strategy_neutral_uniform_random
from ais.randoms import strategy_fully_random, strategy_uniform_random
from ais.probabilistic import strategy_probabilistic_v0
from ais.borderlock import strategy_borderlock
from ais.borderlock_V2 import strategy_borderlock_V2
from ais.mcts import strategy_mcts

# ANSI Colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def _assign_unique_strategy_names(ai_indices: List[int]) -> Tuple[List[str], Dict[str, Tuple[str, ...]]]:
    """
    Generate stable unique names per AI seat so that statistics do not merge
    players that share the same base strategy. Also return, for each base strategy
    name, the ordered list of labels assigned to it so workers can replicate the
    mapping regardless of army ordering.
    """
    base_names = [AIs[idx].name for idx in ai_indices]
    totals = Counter(base_names)
    seen: Dict[str, int] = defaultdict(int)
    unique: List[str] = []
    buckets: Dict[str, List[str]] = defaultdict(list)
    for base in base_names:
        seen[base] += 1
        label = base if totals[base] == 1 else f"{base} #{seen[base]}"
        unique.append(label)
        buckets[base].append(label)
    frozen_buckets: Dict[str, Tuple[str, ...]] = {base: tuple(labels) for base, labels in buckets.items()}
    return unique, frozen_buckets


AIs = {
    -2: strategy_neutral_fully_random,
    -1: strategy_neutral_uniform_random,
    1: strategy_fully_random,
    2: strategy_uniform_random,
    3: strategy_probabilistic_v0,
    5: strategy_borderlock,
    6: strategy_borderlock_V2,
    7: strategy_mcts
}

class TournamentStats:
    """Track statistics across multiple games."""
    
    def __init__(self, player_names: List[str]):
        self.games_played = 0
        self.wins = {name: 0 for name in player_names}
        self.placements = {name: [] for name in player_names}
        self.total_turns = []
        self.game_durations = []
        self.game_log = []
        
    def record_game(self, winner: str, placements: Dict[str, int], turns: int, duration: float, game_num: int):
        self.games_played += 1
        self.wins[winner] += 1
        self.total_turns.append(turns)
        self.game_durations.append(duration)
        
        for player, position in placements.items():
            self.placements[player].append(position)
        
        self.game_log.append({
            'game': game_num,
            'winner': winner,
            'placements': placements.copy(),
            'turns': turns,
            'duration': duration
        })
    
    def get_win_rate(self, player: str) -> float:
        if self.games_played == 0:
            return 0.0
        return (self.wins[player] / self.games_played) * 100
    
    def get_avg_placement(self, player: str) -> float:
        if not self.placements[player]:
            return 0.0
        return sum(self.placements[player]) / len(self.placements[player])
    
    def get_avg_turns(self) -> float:
        return sum(self.total_turns) / len(self.total_turns) if self.total_turns else 0
    
    def get_avg_duration(self) -> float:
        return sum(self.game_durations) / len(self.game_durations) if self.game_durations else 0

def clear_screen():
    print("\033[2J\033[H", end="")

def print_header(title: str, width: int = 80):
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("═" * width)
    print(f"{title:^{width}}")
    print("═" * width)
    print(Colors.ENDC)

# ============================================================================
# Worker Function (runs in separate process)
# ============================================================================

def run_single_game_worker(args: Tuple[List[int], Dict[str, Tuple[str, ...]], int]) -> Optional[Tuple]:
    """
    Worker function that runs a single game.
    This runs in a separate process for parallelization.
    
    Args:
        args: Tuple of (ai_indices, label_buckets, game_num)
    
    Returns:
        (game_num, winner_strategy_name, placements_by_strategy, turns, duration) or None
    """
    ai_indices, label_buckets, game_num = args
    start_time = time.time()
    
    try:
        # Create strategies for this process
        strategies = [AIs[idx] for idx in ai_indices]
        
        # Create world
        world = World(strategies, *risk_map())
        label_pool = {base: list(labels) for base, labels in label_buckets.items()}
        for army in world.armies:
            base = army.strategy.name
            pool = label_pool.get(base)
            if pool:
                label = pool.pop(0)
            else:
                label = f"{army.colour} ({base})"
            setattr(army, "display_name", label)
        
        # Run game
        s_0 = random_initial_state(world)
        max_turns = 500
        game_history = list(game(s_0, max_turns))
        
        if not game_history:
            return None
        
        final_state = game_history[-1][1]
        turns = len(game_history)
        
        # Get all armies and their stats
        all_armies = list(world.armies)
        army_stats = []
        
        for army in all_armies:
            territories = territories_occupied_by_army(final_state, army)
            units = army_units(final_state, army)
            strategy_name = getattr(army, "display_name", army.strategy.name)
            
            army_stats.append({
                'strategy': strategy_name,
                'territories': len(territories),
                'units': units,
                'alive': len(territories) > 0
            })
        
        # Determine placements
        sorted_armies = sorted(army_stats, 
                              key=lambda x: (x['alive'], x['territories'], x['units']),
                              reverse=True)
        
        # Use strategy names as keys (consistent across games)
        placements = {}
        for i, army_info in enumerate(sorted_armies):
            placements[army_info['strategy']] = i + 1
        
        winner_strategy = sorted_armies[0]['strategy']
        duration = time.time() - start_time
        
        return (game_num, winner_strategy, placements, turns, duration)
        
    except Exception as e:
        print(f"{Colors.FAIL}Error in game {game_num}: {e}{Colors.ENDC}")
        return None

# ============================================================================
# Main Tournament Function
# ============================================================================

def render_live_dashboard(stats: TournamentStats,
                          player_names: List[str],
                          games_completed: int,
                          total_games: int,
                          workers: int,
                          start_time: float):
    """Display live dashboard with performance metrics."""
    clear_screen()
    
    title = f"PARALLEL RISK AI TOURNAMENT - {len(player_names)} PLAYERS"
    print_header(title)
    
    # Progress bar
    safe_total = max(total_games, 1)
    progress = min(games_completed / safe_total, 1.0)
    bar_width = 50
    filled = int(bar_width * progress)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    # Calculate speed metrics
    elapsed = time.time() - start_time
    games_per_sec = games_completed / elapsed if elapsed > 0 else 0
    eta_seconds = (total_games - games_completed) / games_per_sec if games_per_sec > 0 else 0
    eta_minutes = eta_seconds / 60
    
    print(f"{Colors.OKCYAN}Progress: [{bar}] {games_completed}/{total_games} ({progress*100:.1f}%){Colors.ENDC}")
    print(f"{Colors.BOLD}Speed: {games_per_sec:.1f} games/sec | Workers: {workers} | ETA: {eta_minutes:.1f} min{Colors.ENDC}")
    
    avg_turns = stats.get_avg_turns()
    avg_duration = stats.get_avg_duration()
    total_minutes = elapsed / 60
    print(f"{Colors.DIM}Avg Turns: {avg_turns:.1f} | Avg Game: {avg_duration:.2f}s | Elapsed: {total_minutes:.1f} min{Colors.ENDC}\n")
    
    # Standings
    print(f"{Colors.BOLD}Current Standings{Colors.ENDC}")
    print("─" * 80)
    print(f"{'Rank':<6} {'Player':<30} {'Wins':>6} {'Win %':>8} {'Avg Place':>11}")
    print("─" * 80)
    
    sorted_players = sorted(
        player_names,
        key=lambda p: (stats.wins[p], -stats.get_avg_placement(p)),
        reverse=True
    )
    
    for rank, player in enumerate(sorted_players, 1):
        wins = stats.wins[player]
        win_rate = stats.get_win_rate(player)
        avg_place = stats.get_avg_placement(player)
        
        if rank == 1:
            color = Colors.OKGREEN + Colors.BOLD
        elif rank == 2:
            color = Colors.OKGREEN
        elif rank == 3:
            color = Colors.OKCYAN
        else:
            color = Colors.ENDC
        
        print(f"{color}{rank:<6} {player:<30} {wins:>6} {win_rate:>7.1f}% {avg_place:>11.2f}{Colors.ENDC}")
    
    print(f"\n{Colors.DIM}Dashboard updates in real-time. Press Ctrl+C to stop.{Colors.ENDC}")

def print_final_statistics(stats: TournamentStats, player_names: List[str], total_time: float):
    """Print comprehensive final statistics."""
    clear_screen()
    print_header("🏆  TOURNAMENT RESULTS  🏆")
    
    print(f"\n{Colors.BOLD}Tournament Summary{Colors.ENDC}")
    print("─" * 80)
    print(f"Total Games:     {stats.games_played}")
    print(f"Total Time:      {total_time/60:.1f} minutes")
    print(f"Speed:           {stats.games_played/total_time:.1f} games/sec")
    print(f"Avg Game Time:   {stats.get_avg_duration():.2f}s")
    print(f"Avg Game Length: {stats.get_avg_turns():.1f} turns")
    print(f"Players:         {len(player_names)}")
    
    print(f"\n{Colors.BOLD}Final Standings{Colors.ENDC}")
    print("─" * 80)
    print(f"{'Rank':<6} {'Player':<30} {'Wins':>6} {'Win %':>8} {'Avg Place':>11}")
    print("─" * 80)
    
    sorted_players = sorted(player_names, 
                           key=lambda p: (stats.wins[p], -stats.get_avg_placement(p)),
                           reverse=True)
    
    for rank, player in enumerate(sorted_players, 1):
        wins = stats.wins[player]
        win_rate = stats.get_win_rate(player)
        avg_place = stats.get_avg_placement(player)
        
        if rank == 1:
            color = Colors.OKGREEN + Colors.BOLD
        elif rank == 2:
            color = Colors.OKGREEN
        elif rank == 3:
            color = Colors.OKCYAN
        else:
            color = Colors.ENDC
        
        print(f"{color}{rank:<6} {player:<30} {wins:>6} {win_rate:>7.1f}% {avg_place:>11.2f}{Colors.ENDC}")
    
    # Placement distribution
    print(f"\n{Colors.BOLD}Placement Distribution{Colors.ENDC}")
    print("─" * 80)
    
    max_position = len(player_names)
    print(f"{'Player':<30}", end='')
    for pos in range(1, max_position + 1):
        print(f" {pos:>4}", end='')
    print()
    print("─" * (30 + 5 * max_position))
    
    for player in sorted_players:
        print(f"{player:<30}", end='')
        for pos in range(1, max_position + 1):
            count = stats.placements[player].count(pos)
            if count > 0:
                if pos == 1:
                    color = Colors.OKGREEN
                elif pos == max_position:
                    color = Colors.FAIL
                else:
                    color = Colors.ENDC
                print(f" {color}{count:>4}{Colors.ENDC}", end='')
            else:
                print(f" {Colors.DIM}   0{Colors.ENDC}", end='')
        print()

def save_results(stats: TournamentStats, player_names: List[str], filename: str, total_time: float):
    """Append tournament results to file."""
    import os
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', encoding='utf-8') as f:
        if file_exists:
            f.write("\n" + "=" * 80 + "\n\n")
        
        f.write(f"Tournament Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Players: {len(player_names)}\n")
        for player in player_names:
            f.write(f"  - {player}\n")
        f.write(f"Games Played: {stats.games_played}\n")
        f.write(f"Total Time: {total_time/60:.1f} minutes\n")
        f.write(f"Speed: {stats.games_played/total_time:.1f} games/sec\n\n")
        
        f.write("FINAL STANDINGS:\n")
        sorted_players = sorted(player_names,
                               key=lambda p: (stats.wins[p], -stats.get_avg_placement(p)),
                               reverse=True)
        
        for rank, player in enumerate(sorted_players, 1):
            wins = stats.wins[player]
            win_rate = stats.get_win_rate(player)
            avg_place = stats.get_avg_placement(player)
            f.write(f"  {rank}. {player}: {wins} wins ({win_rate:.1f}%) | Avg Place: {avg_place:.2f}\n")
        
        f.write(f"\nAvg Game Length: {stats.get_avg_turns():.1f} turns\n")
        f.write(f"Avg Game Time: {stats.get_avg_duration():.2f}s\n")

def run_tournament(ai_indices: List[int], num_games: int, workers: int, update_interval: int):
    """Run parallel tournament."""
    
    # Validate AI indices
    for ai_idx in ai_indices:
        if ai_idx not in AIs:
            print(f"{Colors.FAIL}Error: Invalid AI index {ai_idx}. Available: {list(AIs.keys())}{Colors.ENDC}")
            return
    
    if len(ai_indices) < 2:
        print(f"{Colors.FAIL}Error: Need at least 2 AIs{Colors.ENDC}")
        return
    
    # Build unique player labels and the mapping for duplicate strategies
    player_names, label_buckets = _assign_unique_strategy_names(ai_indices)
    
    stats = TournamentStats(player_names)
    
    # Print header
    clear_screen()
    print_header(f"PARALLEL TOURNAMENT - {len(ai_indices)} PLAYERS")
    print(f"\n{Colors.OKCYAN}Players:{Colors.ENDC}")
    for i, (idx, name) in enumerate(zip(ai_indices, player_names), 1):
        print(f"  {i}. [{idx:2d}] {name}")
    print(f"\n{Colors.OKCYAN}Format:{Colors.ENDC} {num_games} games")
    print(f"{Colors.OKCYAN}Workers:{Colors.ENDC} {workers} parallel processes")
    print(f"{Colors.OKCYAN}Expected speed:{Colors.ENDC} {workers}x faster than single-threaded")
    print(f"\n{Colors.WARNING}Press Ctrl+C to stop early{Colors.ENDC}\n")
    
    input(f"{Colors.BOLD}Press Enter to start...{Colors.ENDC}")
    
    start_time = time.time()
    games_completed = 0
    
    # Prepare game tasks
    game_tasks = [(ai_indices, label_buckets, i) for i in range(1, num_games + 1)]
    
    try:
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            futures = {executor.submit(run_single_game_worker, task): task for task in game_tasks}
            
            # Process results as they complete
            for future in as_completed(futures):
                result = future.result()
                
                if result:
                    game_num, winner, placements, turns, duration = result
                    stats.record_game(winner, placements, turns, duration, game_num)
                    games_completed += 1
                    
                    # Update dashboard periodically
                    if games_completed == 1 or games_completed % update_interval == 0 or games_completed == num_games:
                        render_live_dashboard(stats, player_names, games_completed, num_games, workers, start_time)
        
        total_time = time.time() - start_time
        
        # Final results
        print("\n")
        print_final_statistics(stats, player_names, total_time)
        
        # Save results
        filename = "tournament_results.txt"
        save_results(stats, player_names, filename, total_time)
        print(f"\n{Colors.OKCYAN}📊 Results appended to: {filename}{Colors.ENDC}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}⚠ Tournament interrupted!{Colors.ENDC}\n")
        total_time = time.time() - start_time
        
        if stats.games_played > 0:
            print_final_statistics(stats, player_names, total_time)
            filename = "tournament_results.txt"
            save_results(stats, player_names, filename, total_time)
            print(f"\n{Colors.OKCYAN}📊 Partial results appended to: {filename}{Colors.ENDC}\n")
        else:
            print("No games completed.")

if __name__ == "__main__":
    # Detect CPU core count
    cpu_count = mp.cpu_count()
    recommended_workers = max(cpu_count - 2, 4)  # Leave 2 cores free
    
    parser = argparse.ArgumentParser(
        description='Parallel Risk AI Tournament (Apple Silicon Optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
AI Indices:
  -2: Neutral Fully Random
  -1: Neutral Uniform Random
   1: Fully Random
   2: Uniform Random  
   3: Probabilistic V0
   5: BorderLock
   6: BorderLock V2

Performance Tips:
  - Your M5 has {cpu_count} cores
  - Recommended: --workers {recommended_workers}
  - More workers = faster, but diminishing returns after {cpu_count}
  - Expect 5-10x speedup vs single-threaded

Examples:
  # Fast test (recommended)
  python tournament_parallel.py --ais 6 1 1 1 --games 500 --workers {recommended_workers}
  
  # Maximum speed
  python tournament_parallel.py --ais 6 1 1 1 --games 1000 --workers {cpu_count}
  
  # 2 player comparison
  python tournament_parallel.py --ais 6 5 --games 1000 --workers {recommended_workers}
        """
    )
    
    parser.add_argument('--ais', type=int, nargs='+', required=True,
                       help='AI indices (all play simultaneously)')
    parser.add_argument('--games', type=int, default=500,
                       help='Number of games (default: 500)')
    parser.add_argument('--workers', type=int, default=recommended_workers,
                       help=f'Number of parallel workers (default: {recommended_workers}, max: {cpu_count})')
    parser.add_argument('--update-interval', type=int, default=10,
                       help='Update dashboard every N games (default: 10)')
    
    args = parser.parse_args()
    
    # Validate workers
    if args.workers > cpu_count:
        print(f"{Colors.WARNING}Warning: {args.workers} workers requested but only {cpu_count} cores available{Colors.ENDC}")
        print(f"Setting workers to {cpu_count}\n")
        args.workers = cpu_count
    
    run_tournament(args.ais, args.games, args.workers, args.update_interval)
