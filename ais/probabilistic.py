# ais/probabilistic.py

from strategy.strategy import Strategy
from strategy.reinforcements import reinforce_border_first
from strategy.attacks import attack_if_favorable
from strategy.defenses import maximum_defense
from strategy.invasions import uniform_invasion
from strategy.maneuvers import no_maneuver


strategy_probabilistic = Strategy(
    "probabilistic",
    reinforce_border_first,
    attack_if_favorable,
    maximum_defense,
    uniform_invasion,
    no_maneuver,
)
