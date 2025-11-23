import matplotlib.pyplot as plt
import numpy as np

# Données extraites de tournament_results.txt (Sections Head-to-Head)
labels = ['Test 1:\nBorders Heavy', 'Test 2:\nBorder First']
hpa_scores = [41.0, 49.8]        # Heuristic Probabilistic Attacker
prob_scores = [59.0, 50.2]       # Probabilistic (Baseline)

x = np.arange(len(labels))  # positions des labels
width = 0.35  # largeur des barres

fig, ax = plt.subplots(figsize=(10, 6))

# Création des barres
rects1 = ax.bar(x - width/2, hpa_scores, width, label='HPA', color='#e74c3c', alpha=0.9, edgecolor='black')
rects2 = ax.bar(x + width/2, prob_scores, width, label='Probabilistic', color='#3498db', alpha=0.9, edgecolor='black')

# Titres et Labels
ax.set_ylabel('Taux de Victoire (%)', fontsize=12, fontweight='bold')
ax.set_title('Impact de l\'Heuristique de Renforcement en Duel (H2H)\n(Comparatif sur 10 000 parties)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim([0, 75])  # Un peu d'espace pour les annotations
ax.legend(loc='upper center', ncol=2)

# Ligne des 50%
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Fonction pour ajouter les étiquettes de valeur
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points de décalage vertical
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Annotation de l'amélioration (La flèche verte du script original)
# On pointe du haut de la barre HPA du Test 1 vers le haut de la barre HPA du Test 2
x_start = x[0] - width/2
y_start = hpa_scores[0]
x_end = x[1] - width/2
y_end = hpa_scores[1]

ax.annotate('', xy=(x_end, y_end + 2), xytext=(x_start, y_start + 2),
            arrowprops=dict(arrowstyle='->', lw=2, color='green', connectionstyle="arc3,rad=-0.2"))

# Texte de l'amélioration au milieu de la flèche
mid_x = (x_start + x_end) / 2
mid_y = (y_start + y_end) / 2 + 5
ax.text(mid_x, mid_y, '+8.8% Victoires', ha='center', color='green', fontsize=11, fontweight='bold', backgroundcolor='white')

# Sauvegarde et affichage
plt.tight_layout()
plt.savefig('impact_reinforce_heuristic_corrected.png', dpi=300)
plt.show()