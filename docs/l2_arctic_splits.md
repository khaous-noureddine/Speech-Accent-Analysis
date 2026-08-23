# Splits L2-ARCTIC leave-one-accent-out

## Objectif

Pour les expériences de resoumission, L2-ARCTIC est reconstruit directement
depuis les données brutes situées dans `data/raw/l2_arctic/speakers`. Les
anciens fichiers Parquet de `l2_arctic_cv` ne sont pas utilisés.

Le script associe les WAV et les transcriptions des 24 locuteurs, conserve les
954 prompts disponibles pour tous les locuteurs, puis copie les WAV une seule
fois dans un dossier partagé. Il produit ensuite six folds : Arabic, Chinese,
Hindi, Korean, Spanish et Vietnamese.

Dans chaque fold :

- l’accent indiqué par le nom du dossier est complètement absent de train/dev ;
- ses quatre locuteurs constituent le test ;
- les cinq autres accents fournissent train et dev ;
- les locuteurs et les prompts sont disjoints entre train, dev et test.

Ce protocole mesure donc conjointement la généralisation à un accent, à des
locuteurs et à des prompts non vus pendant l’entraînement.

## Sorties

Les fichiers sont générés dans :

```text
data/processed/l2_arctic_leave_one_accent_out/
├── wavs/                    # copie partagée des WAV
├── inventory.parquet       # inventaire canonique des 22 896 exemples
├── inventory_report.json
├── fold_summary.csv
├── arabic/
├── chinese/
├── hindi/
├── korean/
├── spanish/
└── vietnamese/
```

Chaque dossier d’accent contient :

- `corpus.parquet` : exemples et colonne `split` ;
- `split_stats.csv` : nombres d’exemples, locuteurs, prompts et durées ;
- `manifest.json` : définition exacte du fold ;
- `manifest.content.sha256` : empreinte du manifest ;
- `validation_report.json` : contrôle automatique des recouvrements.

## Commande de génération

À lancer depuis la racine du dépôt :

```bash
PYTHONPATH=src python -m accented_asr.data.prepare_l2_arctic \
  --corpus-dir data/raw/l2_arctic/speakers \
  --output-dir data/processed/l2_arctic_leave_one_accent_out \
  --repository-root . \
  --split-seed 20260817
```

La seed `20260817` concerne uniquement la construction des splits. Elle doit
rester fixe pour toutes les conditions et ne doit pas être confondue avec les
seeds utilisées pour entraîner les modèles.
