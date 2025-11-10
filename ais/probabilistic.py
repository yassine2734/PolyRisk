# ais/probabilistic.py
from functools import lru_cache
from typing import Dict, Optional, Tuple
from strategy.strategy import Strategy
from strategy.defenses import maximum_defense
from strategy.invasions import uniform_invasion
from model.region import Unit
from model.territory import Territory
from model.army import Army
from model.state import State
from model.state_informations import (
    territories_occupied_by_army, territory_units, territory_occupant,
    reinforcement_units
)

# --- Petite table des issues d’un lancer (approximations standards Risk) ---
def _prob_table(a:int, d:int) -> Dict[Tuple[int,int], float]:
    # Retourne { (pertes_att, pertes_def): proba } pour un lancer
    if (a, d) == (1,1): return {(1,0):0.583, (0,1):0.417}
    if (a, d) == (2,1): return {(1,0):0.421, (0,1):0.579}
    if (a, d) == (3,1): return {(1,0):0.340, (0,1):0.660}
    if (a, d) == (2,2): return {(2,0):0.448, (1,1):0.324, (0,2):0.228}
    if (a, d) == (3,2): return {(2,0):0.292, (1,1):0.336, (0,2):0.372}
    # cas (1,2) très défavorable à l’attaquant (on évite en pratique)
    if (a, d) == (1,2): return {(1,0):0.745, (0,1):0.255}  # approx.
    return {}

@lru_cache(maxsize=None)
def conquest_prob(A:int, D:int) -> float:
    """Proba que l’attaquant prenne le territoire en partant de (A attaquants sur case, D défenseurs)."""
    if D == 0: return 1.0
    if A <= 1: return 0.0
    a = min(3, A-1)
    d = min(2, D)
    p = 0.0
    for (dA, dD), q in _prob_table(a, d).items():
        p += q * conquest_prob(A - dA, D - dD)
    return p

def _enemy_neighbors(s: State, t: Territory):
    a = territory_occupant(s, t)
    neigh = [t2 for t2 in s.world.territories if (t, t2) in s.world.adjacencies]
    return [t2 for t2 in neigh if territory_occupant(s, t2) != a]

def _my_border_territories(s: State, a: Army):
    Ts = territories_occupied_by_army(s, a)
    return [t for t in Ts if any(territory_occupant(s, n) != a
                                 for n in s.world.territories
                                 if (t, n) in s.world.adjacencies)]

# --- 1) Reinforcements (bordure d’abord, simple) ---
def reinforce_border_first(s: State, a: Army) -> Dict[Territory, Unit]:
    U = reinforcement_units(s, a)
    if U <= 0: return {}
    borders = _my_border_territories(s, a)
    if not borders:
        # Pas de frontière => tout sur le plus gros territoire
        Ts = sorted(territories_occupied_by_army(s, a),
                    key=lambda t: territory_units(s, t), reverse=True)
        return {Ts[0]: U} if Ts else {}
    # Distribution ronde-robin sur les frontières
    R: Dict[Territory, Unit] = {t:0 for t in borders}
    i = 0
    order = sorted(borders, key=lambda t: -territory_units(s, t))
    while U > 0:
        R[order[i % len(order)]] += 1
        U -= 1
        i += 1
    return R

# --- 2) Attack (choisit une seule attaque si p >= tau) ---
TAU_ATTACK = 0.65
def attack_if_favorable(s: State, a: Army) -> Optional[Tuple[Territory, Territory, Unit]]:
    best = None
    best_p = -1.0
    for t_a in _my_border_territories(s, a):
        A = territory_units(s, t_a)
        if A <= 1:
            continue
        for t_d in _enemy_neighbors(s, t_a):
            D = territory_units(s, t_d)
            p = conquest_prob(A, D)
            if p > best_p:
                best_p = p
                best = (t_a, t_d)
    if best is None or best_p < TAU_ATTACK:
        return None
    t_a, t_d = best
    n = min(3, territory_units(s, t_a)-1)  # dés attaquants
    return (t_a, t_d, n)

# --- 3) Defense (on réutilise la stratégie existante robuste) ---
def defend_max(s: State, t_a: Territory, t_d: Territory, n: Unit) -> Unit:
    return maximum_defense(s, t_a, t_d, n)

# --- 4) Invasion (simple: split uniforme) ---
def invade_uniform(s: State, t_a: Territory, t_d: Territory, n_a: Unit, p_a: Unit) -> Unit:
    return uniform_invasion(s, t_a, t_d, n_a, p_a)

# --- 5) Maneuver (facultatif au V0 : pas de manœuvre) ---
def no_maneuver(_: State, __: Army) -> Optional[Tuple[Territory, Territory, Unit]]:
    return None

# ---- Strategy object à exposer ----
strategy_probabilistic_v0 = Strategy(
    "probabilistic V0 (Markov local)",
    reinforce_border_first,
    attack_if_favorable,
    defend_max,
    invade_uniform,
    no_maneuver
)
