import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Contexte': ['Duel (2J)', '+ 1 Random (3J)', '+ 2 Randoms (4J)', '+ 3 Randoms (5J)'],
    'Prob': [59.0, 48.7, 41.5, 36.0],
    'HPA': [41.0, 50.9, 57.9, 63.0],
    'Écart': [-18.0, +2.2, +16.4, +27.0]
}

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Couleurs conditionnelles
cell_colors = []
for i, row in df.iterrows():
    row_colors = ['white']  # Contexte
    
    # Prob
    if row['Prob'] > 50:
        row_colors.append('#ccffcc')  # Vert clair
    else:
        row_colors.append('#ffcccc')  # Rouge clair
    
    # HPA
    if row['HPA'] > 50:
        row_colors.append('#ccffcc')
    else:
        row_colors.append('#ffcccc')
    
    # Écart
    if row['Écart'] > 0:
        row_colors.append('#ccffcc')
    else:
        row_colors.append('#ffcccc')
    
    cell_colors.append(row_colors)

table = ax.table(cellText=df.values,
                colLabels=df.columns,
                cellLoc='center',
                loc='center',
                cellColours=cell_colors,
                colColours=['lightgray']*4)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 3)

# Bold pour l'en-tête
for i in range(4):
    table[(0, i)].set_text_props(weight='bold', fontsize=13)

ax.set_title('Évolution de la Dominance selon le Contexte\n(10 000 parties par configuration)', 
             fontsize=16, fontweight='bold', pad=20)

plt.savefig('tableau_comparatif_contexte.png', dpi=300, bbox_inches='tight')
plt.show()