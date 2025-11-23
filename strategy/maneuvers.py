#! /bin/python
# gvim: set fileencoding=utf-8
#
# (c) September 11, 2025, José Martinez, Polytech Nantes, University of Nantes
#
# Licence:  proprietary
# The use of this library is not authorised outside the Polytechnic School of the University of Nantes.
#
#


"""
Several strategies are provided in order to move units between adjacent territories occupied by an army.

Notice that the pre- and post-conditions of these functions would not need to be tested.
They are already ensured by the strategies.
However, they are repeated to the programmer for clarity and documentation.
Also, this allows to detect a possible flaw closer to its source.
A model is given below.
Notice that post-conditions can be strengthen.

def model_of_maneuver_function (s: State,
                                a: Army) -> Optional[Tuple[Territory, Territory, Unit]]:
    '''
    :parameter s:  the current state
    :parameter t_a:  the army that might maneuvers

    :pre:  The army must belong to the current state

    :return (t_a, t_d, n):  the number of maneuvering units from a territory to another one, when applied

    :post:  The starting territory belongs to the current state
    :post:  The ending territory belongs to the current state
    :post:  The starting territory is occupied by the army
    :post:  The defending territory is occupied by the army
    :post:  At least one unit is maneuvering
    :post:  At least one unit must be left behind
    '''
    # PRE-CONDITION
    assert isinstance(s, State)
    assert isinstance(a, Army)
    assert a in s.world.armies, "The army must belong to the current state"

    # POST-CONDITION
    if result is not None:
        (t_0, t_1, n) = result
        assert isinstance(t_0, Territory)
        assert isinstance(t_1, Territory)
        assert t_0 in s.state, "The starting territory belongs to the current state"
        assert t_1 in s.state, "The ending territory belongs to the current state"
        assert territory_occupant(s, t_0) == a, "The starting territory is occupied by the army"
        assert territory_occupant(s, t_1) == a, "The defending territory is occupied by the army"
        assert n > 0, "At least one unit is maneuvering"
        assert n < territory_units(s, t_0), "At least one unit must be left behind"

    return result
"""


__all__ = [
    'no_maneuver',
    'random_maneuver',
    'random_uniform_maneuver',
    'random_uniform_largest_maneuver',
    'maneuver_consolidate'
    ]


from typing import Tuple, Optional, List
from random import randint, choice
from model.region import Unit
from model.territory import Territory
from model.army import Army
from model.state import State
from model.state_informations import territory_occupant, territory_units, bordering_territories, territories_occupied_by_army, army_adjacent_territories


def no_maneuver (s: State,
                 a: Army) -> Optional[Tuple[Territory, Territory, Unit]]:
    """
    A peculiar behaviour is *not* to move.

    :return:  Always ineffective
    """
    assert isinstance(s, State)
    assert isinstance(a, Army)
    assert a in s.world.armies, "The army must belong to the current state"

    result = None

    assert result is None, "Always ineffective"

    return result


def random_maneuver (s: State,
                     a: Army) -> Optional[Tuple[Territory, Territory, Unit]]:
    """
    A totally random maneuver.

    :post:  Effective unless the army has only isolated territories or no more than one unit left
    """
    assert isinstance(s, State)
    assert isinstance(a, Army)
    assert a in s.world.armies, "The army must belong to the current state"

    M = [ (t_0, t_1, randint(1, u_t_0 - 1))
          for (t_0, t_1) in army_adjacent_territories(s, a)
          if (u_t_0 := territory_units(s, t_0)) > 1 ]
    result = ( None      if len(M) == 0 else
               choice(M) )

    if result is not None:
        (t_0, t_1, n) = result
        assert isinstance(t_0, Territory)
        assert isinstance(t_1, Territory)
        assert t_0 in s.state, "The starting territory belongs to the current state"
        assert t_1 in s.state, "The ending territory belongs to the current state"
        assert territory_occupant(s, t_0) == a, "The starting territory is occupied by the army"
        assert territory_occupant(s, t_1) == a, "The defending territory is occupied by the army"
        assert n > 0, "At least one unit is maneuvering"
        assert n < territory_units(s, t_0), "At least one unit must be left behind"
    else:
        assert all(territory_occupant(s, b) != a
                   for t in territories_occupied_by_army(s, a)
                   if territory_units(s, t) > 1
                   for b in bordering_territories(s, t)), "Effective unless the army has only isolated territories or no more than one unit left"

    return result


def random_uniform_maneuver (s: State,
                             a: Army) -> Optional[Tuple[Territory, Territory, Unit]]:
    """
    A partly random maneuver that tends to distribute uniformly the units among the territories occupied by the army.

    :post:  Effective unless the army has only isolated territories or not enough units to maneuver
    """
    assert isinstance(s, State)
    assert isinstance(a, Army)
    assert a in s.world.armies, "The army must belong to the current state"

    M = [ (t_0, t_1, (u_t_0 - u_t_1) // 2)
          for (t_0, t_1) in army_adjacent_territories(s, a)
          if (u_t_0 := territory_units(s, t_0)) > (u_t_1 := territory_units(s, t_1)) + 1 ]
    result = ( None      if len(M) == 0 else
               choice(M) )

    if result is not None:
        (t_0, t_1, n) = result
        assert isinstance(t_0, Territory)
        assert isinstance(t_1, Territory)
        assert t_0 in s.state, "The starting territory belongs to the current state"
        assert t_1 in s.state, "The ending territory belongs to the current state"
        assert territory_occupant(s, t_0) == a, "The starting territory is occupied by the army"
        assert territory_occupant(s, t_1) == a, "The defending territory is occupied by the army"
        assert n > 0, "At least one unit is maneuvering"
        assert n < territory_units(s, t_0), "At least one unit must be left behind"
    else:
        assert all(territory_occupant(s, b) != a or abs(territory_units(s, t) - territory_units(s, b)) < 2
                   for t in territories_occupied_by_army(s, a)
                   for b in bordering_territories(s, t)), "Effective unless the army has only isolated territories or not enough units to maneuver"

    return result


def random_uniform_largest_maneuver (s: State,
                                     a: Army) -> Optional[Tuple[Territory, Territory, Unit]]:
    """
    A partly random maneuver that tends to distribute uniformly the units among the territories occupied by the army.
    Large maneuvers are preferred over small ones.

    :post:  Effective unless the army has only isolated territories or not enough units to maneuver
    """
    assert isinstance(s, State)
    assert isinstance(a, Army)
    assert a in s.world.armies, "The army must belong to the current state"

    M = [ (t_0, t_1, (u_t_0 - u_t_1) // 2)
          for (t_0, t_1) in army_adjacent_territories(s, a)
          if (u_t_0 := territory_units(s, t_0)) >= (u_t_1 := territory_units(s, t_1)) + 2 ]
    if len(M) == 0:
        result = None
    else:
        n_max = max(n for (_, _, n) in M)
        result = choice([ ttn
                          for ttn in M
                          if ttn[2] == n_max ])

    if result is not None:
        (t_0, t_1, n) = result
        assert isinstance(t_0, Territory)
        assert isinstance(t_1, Territory)
        assert t_0 in s.state, "The starting territory belongs to the current state"
        assert t_1 in s.state, "The ending territory belongs to the current state"
        assert territory_occupant(s, t_0) == a, "The starting territory is occupied by the army"
        assert territory_occupant(s, t_1) == a, "The defending territory is occupied by the army"
        assert n > 0, "At least one unit is maneuvering"
        assert n < territory_units(s, t_0), "At least one unit must be left behind"
    else:
        assert all(territory_occupant(s, b) != a or abs(territory_units(s, t) - territory_units(s, b)) < 2
                   for t in territories_occupied_by_army(s, a)
                   for b in bordering_territories(s, t)), "Effective unless the army has only isolated territories or not enough units to maneuver"

    return result

# ============================== Shared helpers ==============================

def _neighbors(s: State, t: Territory) -> List[Territory]:
    return [t2 for (src, t2) in s.world.adjacencies if src == t]


def _enemy_neighbors(s: State, a: Army, t: Territory) -> List[Territory]:
    return [n for n in _neighbors(s, t) if territory_occupant(s, n) != a]


def _friendly_neighbors_balanced(s: State, a: Army, t: Territory) -> List[Territory]:
    return [n for n in _neighbors(s, t) if territory_occupant(s, n) == a]


def _balanced_region_completion_value(s: State, a: Army, t: Territory) -> float:
    region = t.region
    missing = sum(1 for terr in s.world.region_territories[region] if territory_occupant(s, terr) != a)
    if missing == 0:
        return 0.0
    if missing == 1:
        return region.value * 2.5
    if missing == 2:
        return region.value * 1.2
    if missing == 3:
        return region.value * 0.5
    return 0.0


def _balanced_frontline_pressure(s: State, a: Army, t: Territory) -> float:
    enemies = _enemy_neighbors(s, a, t)
    if not enemies:
        return 0.2
    enemy_units = sum(territory_units(s, e) for e in enemies)
    friendly_support = sum(territory_units(s, f) for f in _friendly_neighbors_balanced(s, a, t))
    choke_bonus = 1.35 if len(enemies) == 1 else 1.0
    region_bonus = 0.4 * _balanced_region_completion_value(s, a, t)
    return (
        max(0.4, enemy_units - 0.45 * friendly_support + 1.25 * len(enemies))
        * choke_bonus
        + region_bonus
    )


def _border_territories_balanced(s: State, a: Army) -> List[Territory]:
    return [t for t in territories_occupied_by_army(s, a) if _enemy_neighbors(s, a, t)]


# ======================== Heuristic Probabilistic Attacker strategy ======================

def maneuver_consolidate (s: State,
                          a: Army) -> Optional[Tuple[Territory, Territory, Unit]]:
    """
    Move half of the surplus from interior territories to the borders under the
    heaviest pressure.
    """
    assert isinstance(s, State)
    assert isinstance(a, Army)
    assert a in s.world.armies, "The army must belong to the current state"

    borders = set(_border_territories_balanced(s, a))
    if not borders:
        return None

    interior = [t for t in territories_occupied_by_army(s, a) if t not in borders]
    best_move: Optional[Tuple[Territory, Territory, Unit]] = None
    best_gain = 0.0

    for source in interior:
        source_units = territory_units(s, source)
        if source_units <= 2:
            continue
        for neighbor in bordering_territories(s, source):
            if neighbor not in borders:
                continue
            pressure = _balanced_frontline_pressure(s, a, neighbor)
            movable = max(1, (source_units - 1) // 2)
            gain = pressure * movable
            if gain > best_gain:
                best_gain = gain
                best_move = (source, neighbor, movable)

    return best_move


# ======================== Probabilistic strategy ============================

def no_maneuver(_: State,
                __: Army) -> Optional[Tuple[Territory, Territory, Unit]]:
    """Probabilistic helper: skipping the maneuver phase entirely."""
    return None
