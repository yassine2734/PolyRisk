import matplotlib.pyplot as plt
import numpy as np
from model.world import World
from model.state_informations import army_units, territories_occupied_by_army
from boards.polytech_board import risk_map
from game.game import game
from game.setup import random_initial_state

# Importation de vos stratégies
from ais.neutrals import strategy_neutral_fully_random
from ais.randoms import strategy_fully_random
from ais.probabilistic import strategy_probabilistic
from ais.HeuristicProbabilisticAttacker import strategy_heuristic_probabilistic_attacker

# Configuration
GAMES_TO_PLAY = 5000
MAX_TURNS = 1000

def get_army_growth(target_strat, label, strategies_pool):
    """Joue des parties et enregistre l'évolution des unités pour une IA cible"""
    curves = []
    
    print(f"Simulation de {GAMES_TO_PLAY} parties pour {label}...")
    
    for _ in range(GAMES_TO_PLAY):
        # 1 IA cible vs 3 Randoms
        strategies = [target_strat] + [strategy_fully_random] * 3
        world = World(strategies, *risk_map())
        
        # Identification de l'armée cible (celle qui a la stratégie target_strat)
        target_army = next(a for a in world.armies if a.strategy == target_strat)
        
        s0 = random_initial_state(world)
        history = game(s0, MAX_TURNS)
        
        # Extraction de la courbe d'unités
        units_history = []
        # Ajout de l'état initial
        units_history.append(army_units(s0, target_army))
        
        for _, state in history:
            if len(territories_occupied_by_army(state, target_army)) > 0:
                units_history.append(army_units(state, target_army))
            else:
                units_history.append(0) # Éliminé
                
        curves.append(units_history)
        
    return curves

# Exécution des simulations
random_curves = get_army_growth(strategy_fully_random, "Random", [])
prob_curves = get_army_growth(strategy_probabilistic, "Probabilistic", [])
hpa_curves = get_army_growth(strategy_heuristic_probabilistic_attacker, "HPA", [])

# --- Visualisation ---
fig, ax = plt.subplots(figsize=(12, 7))

def plot_avg_curve(curves, color, label):
    # Normalisation de la longueur des courbes
    max_len = max(len(c) for c in curves)
    matrix = np.zeros((len(curves), max_len))
    for i, c in enumerate(curves):
        matrix[i, :len(c)] = c
        # Si la partie s'arrête (victoire), on maintient le score final pour la moyenne
        if len(c) < max_len:
            matrix[i, len(c):] = c[-1] 
            
    mean_curve = np.mean(matrix, axis=0)
    std_curve = np.std(matrix, axis=0)
    x = np.arange(max_len)
    
    ax.plot(x, mean_curve, color=color, label=label, linewidth=2.5)
    ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)

plot_avg_curve(random_curves, 'gray', 'Fully Random')
plot_avg_curve(prob_curves, '#3498db', 'Probabilistic')
plot_avg_curve(hpa_curves, '#e74c3c', 'HPA')

ax.set_title("Dynamique de 'Snowball' : Évolution de la Puissance Militaire\n(Moyenne sur 20 parties vs 3 Randoms)", fontsize=14, fontweight='bold')
ax.set_xlabel("Tours de jeu", fontsize=12)
ax.set_ylabel("Nombre Total d'Unités", fontsize=12)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 60) # On zoome sur le début/milieu de partie où tout se joue

plt.savefig('snowball_dynamics.png', dpi=300)
print("Graphique généré : snowball_dynamics.png")