import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time

# Imports du jeu
from model.world import World
from model.state_informations import undefeated_armies
from boards.polytech_board import risk_map
from game.game import game
from game.setup import random_initial_state

# Importation de vos stratégies
from ais.randoms import strategy_fully_random
from ais.probabilistic import strategy_probabilistic
from ais.HeuristicProbabilisticAttacker import strategy_heuristic_probabilistic_attacker

# Configuration
GAMES_TO_PLAY = 10000
MAX_TURNS = 200 
NUM_CORES = 6

def run_simulation_batch(num_games, predator_strat, max_turns):
    """
    Fonction exécutée par un processus ouvrier (worker).
    Elle joue un paquet de 'num_games' et retourne la matrice de survie locale.
    """
    # On recrée les imports locaux si nécessaire ou on utilise les globaux.
    # En multiprocessing 'spawn' (Windows), les objets doivent être picklables.
    
    local_survival_counts = np.zeros((num_games, max_turns))
    
    for i in range(num_games):
        # Setup : 1 Prédateur vs 3 Proies (Random)
        strategies = [predator_strat] + [strategy_fully_random] * 3
        world = World(strategies, *risk_map())
        
        # Identification des armées
        # Note: On suppose que predator_strat est une fonction unique comparée par référence
        predator_army = next(a for a in world.armies if a.strategy == predator_strat)
        prey_armies = [a for a in world.armies if a.strategy != predator_strat]
        
        s0 = random_initial_state(world)
        history = game(s0, max_turns)
        
        # Analyse de l'historique
        for t, (play, state) in enumerate(history):
            if t >= max_turns: break
            
            alive_armies = list(undefeated_armies(state))
            alive_prey_count = sum(1 for prey in prey_armies if prey in alive_armies)
            
            local_survival_counts[i, t] = alive_prey_count / 3.0
            
        # Remplissage fin de partie
        final_len = len(history)
        if final_len < max_turns:
            local_survival_counts[i, final_len:] = local_survival_counts[i, final_len-1]

    return local_survival_counts

def get_prey_survival_multiprocessing(predator_strat, label):
    """
    Orchestre la simulation en parallèle sur tous les cœurs CPU disponibles.
    """
    print(f"Simulation parallèle (Prédation) avec : {label}...")
    start_time = time.time()
    
    # Détection du nombre de cœurs
    num_cpus = multiprocessing.cpu_count()
    
    # Découpage du travail (Batching)
    # Si 10000 jeux et 8 cœurs -> ~1250 jeux par cœur
    batch_size = GAMES_TO_PLAY // num_cpus
    remainder = GAMES_TO_PLAY % num_cpus
    
    # Création de la liste des tâches (chaque tâche est un nombre de jeux à jouer)
    tasks = [batch_size + (1 if i < remainder else 0) for i in range(num_cpus)]
    
    results = []
    
    # Lancement des processus
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # On map la fonction worker sur notre liste de tâches
        # On passe predator_strat et MAX_TURNS comme arguments fixes
        futures = [executor.submit(run_simulation_batch, n, predator_strat, MAX_TURNS) for n in tasks]
        
        # Récupération des résultats
        for future in futures:
            results.append(future.result())
            
    # Fusion des résultats (empilement des matrices numpy)
    full_matrix = np.vstack(results)
    
    duration = time.time() - start_time
    print(f"   -> Terminé en {duration:.2f} secondes sur {num_cpus} cœurs.")

    # Moyenne sur toutes les parties
    return np.mean(full_matrix, axis=0) * 100

if __name__ == "__main__":
    # --- Exécution ---
    
    print(f"Lancement de l'analyse sur {GAMES_TO_PLAY} parties par stratégie.")

    # 1. Baseline
    curve_baseline = get_prey_survival_multiprocessing(strategy_fully_random, "Fully Random")

    # 2. Prédation par Probabilistic
    curve_prob = get_prey_survival_multiprocessing(strategy_probabilistic, "Probabilistic")

    # 3. Prédation par HPA
    curve_hpa = get_prey_survival_multiprocessing(strategy_heuristic_probabilistic_attacker, "HPA")

    # --- Visualisation ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(MAX_TURNS)

    ax.step(x, curve_baseline, where='post', label='Survie face à Random', color='gray', linestyle=':', linewidth=2)
    ax.step(x, curve_prob, where='post', label='Survie face à Probabilistic', color='#3498db', linewidth=2.5)
    ax.step(x, curve_hpa, where='post', label='Survie face à HPA', color='#e74c3c', linewidth=3)

    ax.fill_between(x, curve_hpa, curve_prob, color='#e74c3c', alpha=0.1, label="Gain d'agressivité HPA")

    ax.set_title("Analyse de Prédation : Vitesse d'élimination des adversaires", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tours de jeu", fontsize=12)
    ax.set_ylabel("% d'Adversaires (Randoms) encore en vie", fontsize=12)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Annotation clé
    idx_50_hpa = np.argmax(curve_hpa <= 50)
    idx_50_prob = np.argmax(curve_prob <= 50)

    # Protection contre le cas où l'array est vide ou condition jamais atteinte
    if np.any(curve_hpa <= 50) and np.any(curve_prob <= 50):
        if idx_50_hpa > 0 and idx_50_prob > 0:
            ax.annotate(f'HPA élimine 50% des cibles\n{idx_50_prob - idx_50_hpa} tours plus tôt !',
                        xy=(idx_50_hpa, 50), xytext=(idx_50_hpa + 20, 60),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        fontsize=10, fontweight='bold', color='#c0392b')

    plt.tight_layout()
    filename = 'predator_survival_analysis_multicore.png'
    plt.savefig(filename, dpi=300)
    print(f"Graphique généré : {filename}")
    plt.show()