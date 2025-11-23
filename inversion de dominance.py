import matplotlib.pyplot as plt
import numpy as np

# Données
players = [2, 3, 4, 5]
prob_winrate = [59.0, 48.7, 41.5, 36.0]
hpa_winrate = [41.0, 50.9, 57.9, 63.0]

plt.figure(figsize=(10, 6))
plt.plot(players, prob_winrate, 'o-', linewidth=2, markersize=10, 
         label='Probabilistic', color='#3498db')
plt.plot(players, hpa_winrate, 's-', linewidth=2, markersize=10, 
         label='HPA', color='#e74c3c')

# Point d'intersection
plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=2.5, color='green', linestyle='--', alpha=0.5, 
            label='Point d\'inversion (~2.5J)')

plt.xlabel('Nombre de Joueurs', fontsize=12)
plt.ylabel('Taux de Victoire (%)', fontsize=12)
plt.title('Inversion de Dominance selon Contexte\n(10 000 parties par point)', 
          fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(players)
plt.ylim([35, 65])

plt.savefig('inversion_dominance.png', dpi=300, bbox_inches='tight')
plt.show()