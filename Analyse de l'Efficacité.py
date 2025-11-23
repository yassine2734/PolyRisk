import matplotlib.pyplot as plt

# Données issues de votre fichier "Analyse de l'Efficacité.txt"
# Format: (Nom, Durée Moyenne (s), Win%, Tours Moyens)
data = [
    ("Random (Baseline)", 0.21, 25.9, 233.1),
    ("Probabilistic", 0.08, 87.7, 89.0),
    ("HPA (Notre IA)", 0.07, 94.9, 61.7)
]

names = [x[0] for x in data]
times = [x[1] for x in data]
wins = [x[2] for x in data]
turns = [x[3] for x in data]

fig, ax = plt.subplots(figsize=(8, 5))

# Couleurs sobres + bulles moins grandes
colors = ['#9ba3ad', '#5dade2', '#e05b5b']
sizes = [t * 4 for t in turns]
ax.scatter(times, wins, s=sizes, c=colors, alpha=0.65, linewidth=0.8, edgecolors="#2f2f2f")

ax.set_title("Performance vs temps de calcul (taille = tours)", fontsize=13)
ax.set_xlabel("Temps moyen par partie (s)", fontsize=11)
ax.set_ylabel("Taux de victoire (%)", fontsize=11)
ax.invert_xaxis()  # moins de temps = mieux

# Libellés simples pour chaque point
for i, name in enumerate(names):
    ax.annotate(
        f"{name}\n{wins[i]}% | {times[i]}s | {int(turns[i])} tours",
        (times[i], wins[i]),
        xytext=(0, 12),
        textcoords='offset points',
        ha='center',
        fontsize=9,
        color="#202020"
    )

ax.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('efficacite_algorithmique.png', dpi=300)
plt.show()
