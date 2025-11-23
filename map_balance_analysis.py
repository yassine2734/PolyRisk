import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

# Imports du jeu
from model.world import World
from model.state_informations import territories_occupied_by_army
from boards.polytech_board import risk_map
from game.game import game
from game.setup import random_initial_state
from ais.probabilistic import strategy_probabilistic

# Configuration
GAMES_TO_PLAY = 5000
MAX_TURNS = 300
NUM_CORES = 6  

def run_balance_batch(num_games, max_turns):
    """
    Worker : Simule un lot de parties et renvoie les comptes locaux
    (wins par territoire et starts par territoire).
    """
    local_wins = defaultdict(int)
    local_starts = defaultdict(int)
    
    # On recrée l'environnement pour ce processus
    strategies = [strategy_probabilistic] * 4
    
    for _ in range(num_games):
        world = World(strategies, *risk_map())
        s0 = random_initial_state(world)
        
        # 1. Enregistrer les positions de départ pour CETTE partie
        # Dict temporaire : Army -> List[TerritoryName]
        current_game_starts = {} 
        
        for army in world.armies:
            terrs = territories_occupied_by_army(s0, army)
            t_names = [t.name for t in terrs]
            current_game_starts[army] = t_names
            
            # On incrémente le compteur global d'apparitions
            for name in t_names:
                local_starts[name] += 1
                
        # 2. Jouer
        history = game(s0, max_turns)
        
        if not history: continue
        final_state = history[-1][1]
        
        # 3. Identifier le vainqueur
        best_army = max(world.armies, key=lambda a: len(territories_occupied_by_army(final_state, a)))
        
        # 4. Créditer la victoire aux territoires de départ du vainqueur
        for t_name in current_game_starts[best_army]:
            local_wins[t_name] += 1
            
    return local_starts, local_wins

def analyze_map_balance_multiprocessing():
    print(f"Analyse parallèle de l'équilibre ({GAMES_TO_PLAY} parties, {NUM_CORES} cœurs)...")
    start_time = time.time()
    
    # --- 1. Préparation des Batchs ---
    batch_size = GAMES_TO_PLAY // NUM_CORES
    remainder = GAMES_TO_PLAY % NUM_CORES
    tasks = [batch_size + (1 if i < remainder else 0) for i in range(NUM_CORES)]
    
    # --- 2. Exécution Parallèle ---
    results = []
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [executor.submit(run_balance_batch, n, MAX_TURNS) for n in tasks]
        for future in futures:
            results.append(future.result())
            
    # --- 3. Agrégation des résultats (Map-Reduce) ---
    total_starts = defaultdict(int)
    total_wins = defaultdict(int)
    
    # On fusionne les dictionnaires de chaque worker
    for batch_starts, batch_wins in results:
        for terr, count in batch_starts.items():
            total_starts[terr] += count
        for terr, count in batch_wins.items():
            total_wins[terr] += count

    duration = time.time() - start_time
    print(f"Simulations terminées en {duration:.2f} secondes.")
    print("Calcul des statistiques...")

    # --- 4. Construction du DataFrame ---
    # On a besoin d'une instance du monde juste pour récupérer les noms et régions
    temp_world = World([strategy_probabilistic]*4, *risk_map())
    all_territory_names = [t.name for t in temp_world.territories]
    
    data = []
    for name in all_territory_names:
        starts = total_starts[name]
        wins = total_wins[name]
        
        # Protection division par zéro (si map très grande ou peu de jeux)
        win_rate = (wins / starts * 100) if starts > 0 else 0
        
        region = next(t.region.name for t in temp_world.territories if t.name == name)
        data.append({'Territoire': name, 'Région': region, 'Win Equity (%)': win_rate})
        
    df = pd.DataFrame(data)
    df = df.sort_values('Win Equity (%)', ascending=False)
    
    return df

if __name__ == "__main__":
    # --- Visualisation ---
    df_results = analyze_map_balance_multiprocessing()

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    # Barplot
    # Note: palette="viridis" colore selon la catégorie Hue (Region)
    chart = sns.barplot(x="Win Equity (%)", y="Territoire", hue="Région", data=df_results, dodge=False, palette="viridis")

    # Ligne de référence
    plt.axvline(x=25, color='red', linestyle='--', label='Équilibre Théorique (25%)')

    plt.title(f"Biais de la Carte ({GAMES_TO_PLAY} simu / {NUM_CORES} cœurs)", fontsize=15, fontweight='bold')
    plt.xlabel("Taux de Victoire quand possédé au tour 0 (%)", fontsize=12)
    plt.ylabel("")
    plt.legend(title="Région", loc='lower right')
    plt.tight_layout()

    filename = 'map_balance_analysis_multicore.png'
    plt.savefig(filename, dpi=300)
    print(f"\nGraphique généré : {filename}")
    
    print("\nTop 3 Territoires 'OP' (OverPowered) :")
    print(df_results.head(3)[['Territoire', 'Win Equity (%)']])
    
    print("\nFlop 3 Territoires 'Maudits' :")
    print(df_results.tail(3)[['Territoire', 'Win Equity (%)']])