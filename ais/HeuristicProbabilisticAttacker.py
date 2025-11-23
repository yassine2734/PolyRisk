"""
Heuristic Probabilistic Attacker Strategy

Key principles from research:
1. Always use maximum dice (3 attacker vs 2 defender = optimal saddle point)
2. Attack only when probability > 65% (from Markov analysis)
3. Defend borders first (from game theory analysis)
4. Maintain 2:1 attack ratio minimum (from probabilistic models)
"""

from strategy.strategy import Strategy
from strategy.defenses import maximum_defense, random_defense
from strategy.reinforcements import reinforce_borders_heavy, random_reinforcement, reinforce_border_first
from strategy.attacks import attack_calculated, random_attack
from strategy.invasions import invasion_pressure_balanced, random_uniform_invasion
from strategy.maneuvers import maneuver_consolidate, random_maneuver

__all__ = ['strategy_heuristic_probabilistic_attacker']


strategy_heuristic_probabilistic_attacker = Strategy(
    "Heuristic Probabilistic Attacker",
    reinforce_borders_heavy,
    attack_calculated,
    maximum_defense,            
    invasion_pressure_balanced, 
    maneuver_consolidate
)
