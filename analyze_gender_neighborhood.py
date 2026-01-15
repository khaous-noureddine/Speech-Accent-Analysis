import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Charger tes résultats
df = pd.read_pickle("word_distances_xlsr53_layer_19.pkl")

# 2. CHARGER TES MÉTADONNÉES (À ADAPTER)
# Supposons que tu as un dictionnaire ou un DF avec 'speaker' et 'gender'
# meta = pd.read_csv("metadata_arabic.csv") 

# 3. Ajouter le genre pour locuteur_x et locuteur_y
# df = df.merge(meta, left_on='speaker_x', right_on='speaker').rename(columns={'gender': 'gender_x'})
# df = df.merge(meta, left_on='speaker_y', right_on='speaker').rename(columns={'gender': 'gender_y'})

# 4. Créer une colonne pour identifier le type de paire
def get_pair_type(row):
    if row['gender_x'] == row['gender_y']:
        return f"Même Genre ({row['gender_x']})"
    else:
        return "Genres Différents"

# df['type_paire'] = df.apply(get_pair_type, axis=1)

# 5. Visualiser le résultat
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df, x='type_paire', y=0) # '0' est la colonne de distance
# plt.title("Distribution des distances DTW par genre (Layer 19)")
# plt.show()