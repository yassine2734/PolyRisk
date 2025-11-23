#! /bin/python
# gvim: set fileencoding=utf-8
#
# (c) September 1, 2025, José Martinez, Polytech Nantes, University of Nantes
#
# Licence:  proprietary
# The use of this library is not authorised outside the Polytechnic School of the University of Nantes.
#


"""
Several strategies are provided in order to reinforce an army.

Notice that the pre- and post-conditions of these functions would not need to be tested.
They are already ensured by the strategies.
However, they are repeated to the programmer for clarity and documentation.
Also, this allows to detect a possible flaw closer to its source.
A model is given below.
Notice that post-conditions can be strengthen.

def model_of_reinforcement_function (s: State,
                                     a: Army) -> Dict[Territory, Unit]:
    '''
    :parameter s:  a current state
    :parameter a:  an army to reinforce

    :pre:  The army belongs to the state
    :pre:  The army must not be defeated
    :pre:  The reinforcement units has to be strictly positive (normally, redundant with the previous one)

    :post:  The selected territories belong to the army
    :post:  Each selected territory must receive at least one unit
    :post:  The sum of all the reinforcement units must be equal to `reinforcement_units(s, a)`
    '''
    # PRE-CONDITION
    assert isinstance(s, State)
    assert isinstance(a, Army)
    assert a in s.world.armies, "unknown army to reinforce"
    assert not is_defeated_army(s, a), "defeated army cannot be reinforced"
    assert reinforcement_units(s, a) > 0, "reinforcement units have to be strictly positive"

    result = DO_SOME_STUFF

    # POST-CONDITION
    assert all(territory_occupant(s, t) == a for t in result), "The selected territories belong to the army"
    assert all(result[t] > 0 for t in result), "Each selected territory must receive at least one unit"
    assert sum(result[t] for t in result) == reinforcement_units(s, a), "The sum of all the reinforcement units must be equal to {reinforcement_units(s, a)}"

    return result
"""


__all__ = [
    'random_reinforcement',
    'random_uniform_reinforcement',
    ]


from typing import Dict, List, Optional, Tuple
from random import shuffle, randint, sample, choice
from model.region import Unit
from model.territory import Territory
from model.state import State
from model.army import Army
from model.state_informations import (
    is_defeated_army,
    territories_occupied_by_army,
    reinforcement_units,
    territory_occupant,
    territory_units,
)


def random_reinforcement (s: State,
                          a: Army) -> Dict[Territory, Unit]:
    """
    A random sub-set of the territories receives a random number of units.
    """
    assert isinstance(s, State)
    assert isinstance(a, Army)
    assert a in s.world.armies, "unknown army to reinforce"
    assert not is_defeated_army(s, a), "defeated army cannot be reinforced"
    assert reinforcement_units(s, a) > 0, "reinforcement units have to be strictly positive"

    T_a = territories_occupied_by_army(s, a)
    card_T_a = len(T_a)
    k = randint(1, min(reinforcement_units(s, a), card_T_a)) # type: Unit
    (q, r) = divmod(reinforcement_units(s, a), k)
    result = { t: q
               for t in sample(list(T_a), k) }
    for _ in range(r):
        result[choice(list(result))] += 1

    assert all(territory_occupant(s, t) == a for t in result), "The selected territories belong to the army"
    assert all(result[t] > 0 for t in result), "Each selected territory must receive at least one unit"
    assert sum(result[t] for t in result) == reinforcement_units(s, a), "The sum of all the reinforcement units must be equal to {reinforcement_units(s, a)}"

    return result


def random_uniform_reinforcement (s: State,
                                  a: Army) -> Dict[Territory, Unit]:
    """
    The new units are as evenly as possible distributed over the territories, plus a random distribution of bonuses.

    :post:  uniform distribution of the units
    """
    assert isinstance(s, State)
    assert isinstance(a, Army)
    assert a in s.world.armies, "unknown army to reinforce"
    assert not is_defeated_army(s, a), "defeated army cannot be reinforced"
    assert reinforcement_units(s, a) > 0, "reinforcement units have to be strictly positive"

    u_a = reinforcement_units(s, a)
    T_a = list(territories_occupied_by_army(s, a))
    shuffle(T_a)
    N_a = [ n_a
            for i in range(len(T_a))
            if (n_a := u_a // len(T_a) + (1 if i < u_a % len(T_a) else 0)) > 0 ]
    result = dict(zip(T_a, N_a))

    assert all(territory_occupant(s, t) == a for t in result), "The selected territories belong to the army"
    assert all(result[t] > 0 for t in result), "Each selected territory must receive at least one unit"
    assert sum(result[t] for t in result) == reinforcement_units(s, a), "The sum of all the reinforcement units must be equal to {reinforcement_units(s, a)}"
    assert all(abs(result[t_0] - result[t_1]) <= 1 for t_0 in result for t_1 in result), "uniform distribution of the units"

    return result

# ============================== Shared helpers ==============================




MAX_FOCUSED_CAP = 8
MIN_WEIGHT = 0.25
REGION_LOCK_THRESHOLD = 2
REGION_LOCK_WEIGHT = 1.4
ELIMINATION_FOCUS_THRESHOLD = 6
ELIMINATION_BORDER_BONUS = 2.4


def _neighbors(s: State, t: Territory) -> List[Territory]:
    return [t2 for (src, t2) in s.world.adjacencies if src == t]


def _enemy_neighbors(s: State, a: Army, t: Territory) -> List[Territory]:
    return [n for n in _neighbors(s, t) if territory_occupant(s, n) != a]


def _friendly_neighbors(s: State, a: Army, t: Territory) -> List[Territory]:
    return [n for n in _neighbors(s, t) if territory_occupant(s, n) == a]


def _region_missing_count(s: State, a: Army, t: Territory) -> int:
    region = t.region
    return sum(1 for terr in s.world.region_territories[region] if territory_occupant(s, terr) != a)


def _region_completion_bonus(s: State, a: Army, t: Territory) -> float:
    region = t.region
    region_territories = s.world.region_territories[region]
    missing = [terr for terr in region_territories if territory_occupant(s, terr) != a]
    if not missing or t not in missing:
        return 0.0
    remaining = len(missing) - 1
    if remaining == 0:
        return region.value * 2.5
    if remaining == 1:
        return region.value * 1.2
    if remaining == 2:
        return region.value * 0.4
    return 0.0


def _balanced_region_completion_value(s: State, a: Army, t: Territory) -> float:
    missing = _region_missing_count(s, a, t)
    if missing == 0:
        return 0.0
    if missing == 1:
        return t.region.value * 2.5
    if missing == 2:
        return t.region.value * 1.2
    if missing == 3:
        return t.region.value * 0.5
    return 0.0


def _focus_enemy(s: State, a: Army) -> Optional[Tuple[Army, int]]:
    opponents = [opp for opp in s.world.armies if opp != a and territories_occupied_by_army(s, opp)]
    if not opponents:
        return None
    weakest = min(opponents, key=lambda opp: len(territories_occupied_by_army(s, opp)))
    return weakest, len(territories_occupied_by_army(s, weakest))


def _dominance_gap(s: State, a: Army) -> int:
    mine = len(territories_occupied_by_army(s, a))
    opponents = [
        len(territories_occupied_by_army(s, opp))
        for opp in s.world.armies
        if opp != a and territories_occupied_by_army(s, opp)
    ]
    if not opponents:
        return 0
    return mine - max(opponents)


def _frontline_pressure(s: State, a: Army, t: Territory) -> float:
    enemies = _enemy_neighbors(s, a, t)
    if not enemies:
        return 0.2
    enemy_units = sum(territory_units(s, e) for e in enemies)
    friendly_support = sum(territory_units(s, f) for f in _friendly_neighbors(s, a, t))
    choke_bonus = 1.35 if len(enemies) == 1 else 1.0
    region_bonus = 0.5 * _region_completion_bonus(s, a, t)
    missing = _region_missing_count(s, a, t)
    lock_bonus = 0.0
    if 0 < missing <= REGION_LOCK_THRESHOLD:
        lock_bonus = (REGION_LOCK_THRESHOLD - missing + 1) * REGION_LOCK_WEIGHT
    return (
        max(MIN_WEIGHT, enemy_units - 0.5 * friendly_support + 1.4 * len(enemies))
        * choke_bonus
        + region_bonus
        + lock_bonus
    )


def _border_territories_balanced(s: State, a: Army) -> List[Territory]:
    return [t for t in territories_occupied_by_army(s, a) if _enemy_neighbors(s, a, t)]


# ======================== Probabilistic strategy ============================

def _prob_border_territories(s: State, a: Army) -> List[Territory]:
    territories = territories_occupied_by_army(s, a)
    return [t for t in territories if _enemy_neighbors(s, a, t)]


def reinforce_border_first(s: State,
                           a: Army) -> Dict[Territory, Unit]:
    """
    Probabilistic helper: cycle reinforcements over border territories, stacking
    the largest front first when no borders are present.
    """
    units = reinforcement_units(s, a)
    assert units > 0, "reinforcements units have to be strictly positive"

    borders = _prob_border_territories(s, a)
    if not borders:
        owned = sorted(
            territories_occupied_by_army(s, a),
            key=lambda t: territory_units(s, t),
            reverse=True
        )
        return {owned[0]: units} if owned else {}

    allocation: Dict[Territory, Unit] = {}
    order = sorted(borders, key=lambda t: -territory_units(s, t))
    for i in range(units):
        territory = order[i % len(order)]
        allocation[territory] = allocation.get(territory, 0) + 1
    return allocation


# ======================== Heuristic Probabilistic Attacker strategy ======================

def _focus_slots(border_count: int) -> int:
    return min(MAX_FOCUSED_CAP, max(1, border_count))


def reinforce_borders_heavy(s: State,
                            a: Army) -> Dict[Territory, Unit]:
    """
    Distribute most reinforcements across hot borders while locking fragile regions.
    """
    assert isinstance(s, State)
    assert isinstance(a, Army)
    assert a in s.world.armies, "unknown army to reinforce"
    assert not is_defeated_army(s, a), "defeated army cannot be reinforced"
    assert reinforcement_units(s, a) > 0, "reinforcements units have to be strictly positive"

    units = reinforcement_units(s, a)
    territories = list(territories_occupied_by_army(s, a))
    if not territories:
        return {}

    borders = _border_territories_balanced(s, a)
    if not borders:
        per_territory = max(1, units // len(territories))
        result = {t: per_territory for t in territories[:units]}
        remaining = units - sum(result.values())
        if remaining > 0 and territories:
            result[territories[0]] = result.get(territories[0], 0) + remaining
        return result

    border_scores = [(t, _frontline_pressure(s, a, t)) for t in borders]
    interior_candidates = [
        (t, _balanced_region_completion_value(s, a, t))
        for t in territories
        if t not in borders
    ]
    interior_candidates = [(t, value) for t, value in interior_candidates if value > 0]

    interior_budget = 0
    if interior_candidates:
        interior_budget = max(1, units // 5)
    if interior_budget > units:
        interior_budget = units // 2

    units_for_borders = units - interior_budget
    result: Dict[Territory, Unit] = {}
    assigned = 0

    weights: List[Tuple[Territory, float]] = []
    for territory, score in border_scores:
        weight = max(0.1, score) if score > 0 else 1.0
        weights.append((territory, weight))
    total_weight = sum(weight for _, weight in weights) or 1.0
    for territory, weight in weights:
        allocation = int(weight / total_weight * units_for_borders)
        if allocation > 0:
            result[territory] = allocation
            assigned += allocation

    remainder = units_for_borders - assigned
    if remainder > 0:
        for territory, _ in sorted(weights, key=lambda item: item[1], reverse=True):
            if remainder == 0:
                break
            result[territory] = result.get(territory, 0) + 1
            remainder -= 1

    if interior_budget > 0 and interior_candidates:
        interior_candidates.sort(key=lambda item: item[1], reverse=True)
        for territory, _ in interior_candidates:
            if interior_budget == 0:
                break
            result[territory] = result.get(territory, 0) + 1
            interior_budget -= 1

    distributed = sum(result.values())
    if distributed < units:
        frontier = max(border_scores, key=lambda item: item[1])[0]
        result[frontier] = result.get(frontier, 0) + (units - distributed)

    focus_info = _focus_enemy(s, a)
    focus_army = focus_info[0] if focus_info else None
    focus_size = focus_info[1] if focus_info else None
    targets = list(result.keys())
    weights = []
    focus_fronts: List[Territory] = []
    for t in targets:
        base = max(MIN_WEIGHT, _frontline_pressure(s, a, t))
        if focus_army:
            touches_focus = any(territory_occupant(s, e) == focus_army for e in _enemy_neighbors(s, a, t))
            if touches_focus:
                focus_fronts.append(t)
                missing = max(0, (ELIMINATION_FOCUS_THRESHOLD - (focus_size or ELIMINATION_FOCUS_THRESHOLD)))
                base += ELIMINATION_BORDER_BONUS * (1 + missing * 0.5)
        weights.append(base)

    dominance_gap = _dominance_gap(s, a)
    priority = []
    seen = set()
    for group in (focus_fronts, targets):
        for t in group:
            if t not in seen:
                priority.append(t)
                seen.add(t)

    for critical in focus_fronts:
        donors = [t for t in targets if t != critical and result.get(t, 0) > 1]
        while result[critical] < 2 and donors:
            donor = max(donors, key=lambda d: result[d])
            if result[donor] <= 1:
                break
            result[donor] -= 1
            result[critical] += 1

    if dominance_gap >= 4 and targets:
        spearhead = max(
            targets,
            key=lambda t: (result.get(t, 0), _frontline_pressure(s, a, t))
        )
        donors = [t for t in targets if t != spearhead and result.get(t, 0) > 1]
        donors.sort(key=lambda t: result[t], reverse=True)
        need = max(0, 3 - result[spearhead]) + dominance_gap // 2
        while donors and need > 0:
            donor = donors[0]
            result[donor] -= 1
            result[spearhead] += 1
            need -= 1
            if result[donor] <= 1:
                donors.pop(0)
            else:
                donors.sort(key=lambda t: result[t], reverse=True)

    assert all(territory_occupant(s, t) == a for t in result), "The selected territories belong to the army"
    assert sum(result.values()) == units, "The sum of all the reinforcement units must be equal to reinforcement_units(s, a)"

    return result
