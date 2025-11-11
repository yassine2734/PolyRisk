# ais/balanced_aggressor.py
"""
Balanced Aggressor Strategy - Based on Risk research papers

Key principles from research:
1. Always use maximum dice (3 attacker vs 2 defender = optimal saddle point)
2. Attack only when probability > 65% (from Markov analysis)
3. Defend borders first (from game theory analysis)
4. Maintain 2:1 attack ratio minimum (from probabilistic models)
"""

from strategy.strategy import Strategy
from strategy.defenses import maximum_defense
from strategy.reinforcements import reinforce_borders_heavy
from strategy.attacks import attack_calculated
from strategy.invasions import invasion_pressure_balanced
from strategy.maneuvers import maneuver_consolidate

__all__ = ['strategy_balanced_aggressor']


strategy_balanced_aggressor = Strategy(
    "Balanced Aggressor (Research-Based)",
    reinforce_borders_heavy,
    attack_calculated,
    maximum_defense,            # Toujours défendre avec maximum
    invasion_pressure_balanced, # Conserve l'élan offensif sur la nouvelle frontière
    maneuver_consolidate
)
