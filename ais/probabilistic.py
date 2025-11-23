from strategy.strategy import Strategy
from strategy.reinforcements import random_reinforcement, random_uniform_reinforcement, reinforce_border_first, reinforce_borders_heavy
from strategy.attacks import attack_if_favorable, random_attack
from strategy.defenses import random_defense, maximum_defense
from strategy.invasions import random_uniform_invasion, uniform_invasion
from strategy.maneuvers import no_maneuver, random_maneuver, random_uniform_maneuver


strategy_probabilistic = Strategy(
    "probabilistic",
    reinforce_border_first ,
    attack_if_favorable,
    maximum_defense,
    random_uniform_invasion,
    no_maneuver
)