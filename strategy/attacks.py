#! /bin/python
# gvim: set fileencoding=utf-8
#
# (c) September 8, 2025, José Martinez, Polytech Nantes, University of Nantes
#
# Licence:  proprietary
# The use of this library is not authorised outside the Polytechnic School of the University of Nantes.
#


"""
Several strategies are provided in order to attack an opponent army.

Notice that the pre- and post-conditions of these functions would not need to be tested.
They are already ensured by the strategies.
However, they are repeated to the programmer for clarity and documentation.
Also, this allows to detect a possible flaw closer to its source.
A model is given below.
Notice that post-conditions can be strengthen.

def model_of_attack_function (s: State,
                              a: Army) -> Optional[Tuple[Territory, Territory, Unit]]:
    '''
    :parameter s:  a current state
    :parameter a:  an army to reinforce

    :pre:  The army must belong to the current state
    :pre:  The army must not be defeated

    :post:  The attacking territory belongs to the current state
    :post:  The defending territory belongs to the current state
    :post:  The attacking territory is occupied by the army
    :post:  The defending territory is not occupied by the army
    :post:  The attacking units is between one and three
    :post:  The attacking units is strictly less than the number of available units
    '''
    # PRE-CONDITION
    assert isinstance(s, State)
    assert isinstance(a, Army)
    assert a in s.world.armies, "The army must belong to the current state"
    assert not is_defeated_army(s, a), "The army must not be defeated"

    result = DO_SOME_STUFF

    # POST-CONDITION
    if result is not None:
        (t_a, t_d, n) = result
        assert isinstance(t_a, Territory)
        assert isinstance(t_d, Territory)
        assert t_a in s.state, "The attacking territory belongs to the current state"
        assert t_d in s.state, "The defending territory belongs to the current state"
        assert territory_occupant(s, t_a) == a, "The attacking territory is occupied by the army"
        assert territory_occupant(s, t_d) != a, "The defending territory is not occupied by the army"
        assert n in { 1, 2, 3 }, "The attacking units is between one and three"
        assert n < territory_units(s, t_a), "The attacking units is strictly less than the number of available units"

    return result
"""


__all__ = ['random_attack','attack_calculated', 'attack_if_favorable']


from functools import lru_cache
from random import choice
from typing import Dict, List, Optional, Tuple
from model.region import Unit
from model.army import Army
from model.territory import Territory
from model.state import State
from model.state_informations import (
    territory_occupant,
    territory_units,
    is_defeated_army,
    territories_occupied_by_army,
)



def random_attack (s: State,
                   a: Army) -> Optional[Tuple[Territory, Territory, Unit]]:
    """
    A totally random attack between any two bordering and enemy territories with any number of units.
    """
    assert isinstance(s, State)
    assert isinstance(a, Army)
    assert a in s.world.armies, "the army must belong to the current state"
    assert not is_defeated_army(s, a), "The army must not be defeated"

    A = [ (t_a, t_d, n)
          for (t_a, t_d) in s.world.adjacencies
          if territory_occupant(s, t_a) == a
          if territory_occupant(s, t_d) != a
          if (u_a := territory_units(s, t_a)) > 1
          for n in range(1, min(3, u_a - 1) + 1) ]
    if len(A) == 0:
        result = None
    else:
        result = choice(A)

    if result is not None:
        (t_a, t_d, n) = result
        assert isinstance(t_a, Territory)
        assert isinstance(t_d, Territory)
        assert t_a in s.state, "The attacking territory belongs to the current state"
        assert t_d in s.state, "The defending territory belongs to the current state"
        assert territory_occupant(s, t_a) == a, "The attacking territory is occupied by the army"
        assert territory_occupant(s, t_d) != a, "The defending territory is not occupied by the army"
        assert n in { 1, 2, 3 }, "The attacking units is between one and three"
        assert n < territory_units(s, t_a), "The attacking units is strictly less than the number of available units"

    return result


# ============================== Shared helpers ==============================

BALANCED_BATTLE_OUTCOMES: Dict[Tuple[int, int], Tuple[Tuple[int, int, float], ...]] = {
    (1, 1): ((1, 0, 0.583), (0, 1, 0.417)),
    (1, 2): ((1, 0, 0.745), (0, 1, 0.255)),
    (2, 1): ((1, 0, 0.421), (0, 1, 0.579)),
    (2, 2): ((2, 0, 0.448), (1, 1, 0.324), (0, 2, 0.228)),
    (3, 1): ((1, 0, 0.255), (0, 1, 0.745)),
    (3, 2): ((2, 0, 0.2926), (1, 1, 0.3357), (0, 2, 0.3717)),
}


@lru_cache(maxsize=200_000)
def _balanced_conquest_probability(attackers: int, defenders: int) -> float:
    a = max(0, int(attackers))
    d = max(0, int(defenders))
    if d <= 0:
        return 1.0
    if a <= 1:
        return 0.0

    A, D = a, d
    # d = 0 -> victoire certaine pour tout a
    row_d0 = [1.0] * (A + 1)
    prev2 = [0.0] * (A + 1)   # d-2 (inexistant au départ)
    prev1 = row_d0            # d-1 = 0

    for cur_d in range(1, D + 1):
        curr = [0.0] * (A + 1)  # par défaut: a=0 ou 1 -> 0
        for cur_a in range(2, A + 1):
            ad = min(3, cur_a - 1)
            dd = min(2, cur_d)
            outcomes = BALANCED_BATTLE_OUTCOMES[(ad, dd)]

            v = 0.0
            for loss_a, loss_d, weight in outcomes:
                aa = cur_a - loss_a
                if aa < 0:
                    aa = 0
                base = curr if loss_d == 0 else (prev1 if loss_d == 1 else prev2)
                v += weight * base[aa]
            curr[cur_a] = v

        prev2, prev1 = prev1, curr

    return prev1[A]


def _neighbors(s: State, t: Territory) -> List[Territory]:
    return [t2 for (src, t2) in s.world.adjacencies if src == t]


def _enemy_neighbors(s: State, a: Army, t: Territory) -> List[Territory]:
    return [n for n in _neighbors(s, t) if territory_occupant(s, n) != a]


def _weak_neighbors(s: State, a: Army, t: Territory) -> List[Tuple[Territory, int]]:
    attackers = territory_units(s, t)
    weak: List[Tuple[Territory, int]] = []
    for neighbor in _neighbors(s, t):
        if territory_occupant(s, neighbor) == a:
            continue
        enemy_units = territory_units(s, neighbor)
        if attackers > enemy_units:
            weak.append((neighbor, enemy_units))
    return sorted(weak, key=lambda item: item[1])


def _region_capture_bonus(s: State, a: Army, target: Territory) -> float:
    region = target.region
    region_territories = s.world.region_territories[region]
    missing = [t for t in region_territories if territory_occupant(s, t) != a]
    if target not in missing:
        return 0.0
    remaining = len(missing) - 1
    if remaining == 0:
        return region.value * 3.0
    if remaining == 1:
        return region.value * 1.4
    if remaining == 2:
        return region.value * 0.5
    return 0.0


def _balanced_elimination_bonus(s: State, defender: Army) -> float:
    territories = territories_occupied_by_army(s, defender)
    size = len(territories)
    if size <= 2:
        return 6.0
    if size <= 4:
        return 3.0
    return 0.0


def _border_delta_balanced(s: State,
                           a: Army,
                           source: Territory,
                           target: Territory) -> float:
    current = len(_enemy_neighbors(s, a, source))
    future_source = max(0, current - 1)
    target_future_neighbors = _enemy_neighbors(s, a, target)
    future_source += len(target_future_neighbors)
    delta = current - future_source
    return delta * 1.3


# ======================== Probabilistic strategy ============================

_ROLL_ODDS: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {
    (1, 1): {(1, 0): 0.583, (0, 1): 0.417},
    (2, 1): {(1, 0): 0.421, (0, 1): 0.579},
    (3, 1): {(1, 0): 0.340, (0, 1): 0.660},
    (2, 2): {(2, 0): 0.448, (1, 1): 0.324, (0, 2): 0.228},
    (3, 2): {(2, 0): 0.292, (1, 1): 0.336, (0, 2): 0.372},
    (1, 2): {(1, 0): 0.745, (0, 1): 0.255},
}


@lru_cache(maxsize=200_000)
def conquest_prob(attackers: int, defenders: int) -> float:
    a = max(0, int(attackers))
    d = max(0, int(defenders))
    if d <= 0:
        return 1.0
    if a <= 1:
        return 0.0

    A, D = a, d
    row_d0 = [1.0] * (A + 1)
    prev2 = [0.0] * (A + 1)
    prev1 = row_d0

    for cur_d in range(1, D + 1):
        curr = [0.0] * (A + 1)
        for cur_a in range(2, A + 1):
            ad = min(3, cur_a - 1)
            dd = min(2, cur_d)
            odds = _ROLL_ODDS.get((ad, dd), {})

            v = 0.0
            for (loss_a, loss_d), prob in odds.items():
                if prob <= 0.0:
                    continue
                aa = cur_a - loss_a
                if aa < 0:
                    aa = 0
                base = curr if loss_d == 0 else (prev1 if loss_d == 1 else prev2)
                v += prob * base[aa]
            curr[cur_a] = v

        prev2, prev1 = prev1, curr

    return prev1[A]


def _prob_border_territories(s: State, a: Army) -> List[Territory]:
    return [t for t in territories_occupied_by_army(s, a) if _enemy_neighbors(s, a, t)]



TAU_ATTACK = 0.65
def attack_if_favorable(s: State,
                        a: Army) -> Optional[Tuple[Territory, Territory, Unit]]:
    """
    Probabilistic helper: pick the border attack with the best conquest odds
    when it clears a configurable probability threshold.
    """
    assert isinstance(s, State)
    assert isinstance(a, Army)
    assert a in s.world.armies, "the army must belong to the current state"

    best_pair: Optional[Tuple[Territory, Territory]] = None
    best_prob = -1.0

    for attacker in _prob_border_territories(s, a):
        attack_units = territory_units(s, attacker)
        if attack_units <= 1:
            continue
        for defender in _enemy_neighbors(s, a, attacker):
            defend_units = territory_units(s, defender)
            prob = conquest_prob(attack_units, defend_units)
            if prob > best_prob:
                best_prob = prob
                best_pair = (attacker, defender)

    if best_pair is None or best_prob < TAU_ATTACK:
        return None

    attacker, defender = best_pair
    dice = min(3, territory_units(s, attacker) - 1)
    if dice <= 0:
        return None
    return attacker, defender, dice


# ======================== Heuristic Probabilistic Attacker strategy ======================

def attack_calculated(s: State,
                      a: Army) -> Optional[Tuple[Territory, Territory, Unit]]:
    """
    Execute an attack only when attackers keep a 2:1 ratio and conquest odds
    exceed ~65%, prioritising region locks and finishing wounded opponents.
    """
    assert isinstance(s, State)
    assert isinstance(a, Army)
    assert a in s.world.armies, "the army must belong to the current state"
    assert not is_defeated_army(s, a), "The army must not be defeated"

    best_attack: Optional[Tuple[Territory, Territory, Unit]] = None
    best_score = 0.0

    for attacker in territories_occupied_by_army(s, a):
        my_units = territory_units(s, attacker)
        if my_units <= 2:
            continue

        weak_neighbors = _weak_neighbors(s, a, attacker)
        if not weak_neighbors:
            continue

        for defender, enemy_units in weak_neighbors:
            defender_army = territory_occupant(s, defender)
            ratio_required = 2.0
            region_bonus = _region_capture_bonus(s, a, defender)
            elimination_bonus = _balanced_elimination_bonus(s, defender_army)
            if region_bonus > 0 or elimination_bonus > 0:
                ratio_required = 1.5

            if my_units < max(3, int(enemy_units * ratio_required)):
                continue

            prob = _balanced_conquest_probability(my_units, enemy_units)
            threshold = 0.65
            if region_bonus > 0:
                threshold -= 0.08
            if elimination_bonus > 0:
                threshold -= 0.05
            threshold = max(0.5, threshold)
            if prob < threshold:
                continue

            enemy_neighbors = len(_enemy_neighbors(s, a, defender))
            border_bonus = 3.0 if enemy_neighbors <= 1 else 0.0
            border_delta = _border_delta_balanced(s, a, attacker, defender)
            exposure_penalty = max(0.0, enemy_neighbors - 3) * 0.5

            strategic_value = (
                enemy_units
                + 1
                + region_bonus
                + elimination_bonus
                + border_bonus
                + border_delta
                - exposure_penalty
            )
            score = prob * strategic_value

            if score > best_score:
                dice = min(3, my_units - 1)
                if dice <= 0:
                    continue
                best_score = score
                best_attack = (attacker, defender, dice)

    return best_attack