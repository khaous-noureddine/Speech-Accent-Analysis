import pandas as pd

sa="/home/dist/noureddine.khaous/napster/accented-speech-recognition/data/processed/speech_accents/corpus.parquet"
l2_arctic="/home/dist/noureddine.khaous/napster/accented-speech-recognition/data/processed/l2_arctic/corpus.parquet"
arctic="/home/dist/noureddine.khaous/napster/accented-speech-recognition/data/processed/arctic/corpus.parquet"


 
df = pd.read_parquet(arctic) 
# 1. Aperçu du dataset
print("=== Aperçu du dataset ===")
print(df.head(), "\n")

# 3. Colonnes disponibles
print("=== Colonnes ===")
print(df.columns.tolist(), "\n")

# # 2. Informations générales
# print("=== Infos générales ===")
# print(f"Nombre de lignes : {len(df)}")
# print(f"Nombre de colonnes : {df.shape[1]}\n")



# # 4. Exemple de chemin audio
# print("=== Exemple audio_path ===")
# print(df["audio_path"].iloc[0], "\n")

# # 5. Statistiques rapides
# print("=== Statistiques ===")
# print(df.describe(include="all"))