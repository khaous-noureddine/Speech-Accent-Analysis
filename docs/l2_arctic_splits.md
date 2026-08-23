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

## Détails d’implémentation

### Construction de l’inventaire

Le script parcourt les dossiers bruts des 24 locuteurs. Pour chaque WAV, il
recherche une transcription portant le même identifiant de prompt. Une entrée
est rejetée si la transcription est absente ou vide, si l’audio est vide ou si
la paire `(speaker_id, prompt_id)` est dupliquée.

Les transcriptions sont normalisées de la manière suivante : suppression des
espaces en début et fin, passage en minuscules, suppression de la ponctuation
sauf les apostrophes, puis réduction des espaces consécutifs à un seul espace.

Les fichiers audio ne sont ni rééchantillonnés ni transformés pendant cette
préparation. Les 26 867 WAV bruts sont mono, PCM 16 bits, à 44,1 kHz. Ils sont
copiés une seule fois dans le dossier partagé `wavs/`; les six folds stockent
uniquement leur chemin relatif. Le rééchantillonnage vers la fréquence attendue
par le modèle devra être effectué au chargement pendant l’entraînement.

Pour garantir un corpus parallèle équilibré, seuls les prompts disposant d’un
audio et d’une transcription pour chacun des 24 locuteurs sont retenus. Cette
intersection globale contient 954 prompts, soit 22 896 exemples
(`954 × 24`). Les données brutes contiennent 26 867 paires audio-transcription :
3 971 paires sont donc exclues de l’inventaire canonique parce que leur prompt
n’est pas disponible pour les 24 locuteurs.

Chaque ligne de `inventory.parquet` contient notamment le corpus, le locuteur,
le genre, la langue maternelle/accent, les identifiants d’utterance et de
prompt, la transcription normalisée, le chemin audio, la durée, la fréquence
d’échantillonnage et le nombre de canaux.

| Accent/L1 | Locuteurs |
|---|---|
| Arabic | ABA, SKA, YBAA, ZHAA |
| Chinese | BWC, LXC, NCC, TXHC |
| Hindi | ASI, RRBI, SVBI, TNI |
| Korean | HJK, HKK, YDCK, YKWK |
| Spanish | EBVS, ERMS, MBMPS, NJS |
| Vietnamese | HQTV, PNV, THV, TLV |

### Construction déterministe des folds

Les 954 prompts sont mélangés de façon déterministe avec la seed de split
`20260817`, puis répartis globalement selon un ratio cible 80/10/10. La règle du
plus grand reste donne exactement 763 prompts de train, 96 de dev et 95 de
test. Cette même partition de prompts est utilisée dans les six folds.

Pour chaque accent tenu à l’écart :

- ses quatre locuteurs sont assignés au test ;
- pour chacun des cinq accents vus, trois locuteurs sont assignés au train et
  le quatrième au dev ;
- une entrée n’est conservée dans un split que si le rôle de son locuteur et le
  rôle de son prompt correspondent au même split.

Cette dernière règle explique pourquoi le ratio 80/10/10 s’applique aux
prompts, mais pas directement aux nombres d’exemples. Les assignations des
locuteurs et des prompts délibérément croisées entre deux splits ne sont pas
utilisées dans le fold concerné.

La seed de split est distincte des seeds d’entraînement. Une nouvelle seed de
modèle ne doit jamais modifier les prompts, locuteurs ou accents assignés aux
splits.

## Distribution dans chaque fold

| Split | Exemples | Prompts | Locuteurs | Accents | Répartition par accent |
|---|---:|---:|---:|---:|---|
| Train | 11 445 | 763 | 15 | 5 | 3 locuteurs et 2 289 exemples par accent vu |
| Dev | 480 | 96 | 5 | 5 | 1 locuteur et 96 exemples par accent vu |
| Test | 380 | 95 | 4 | 1 | 4 locuteurs et 380 exemples pour l’accent tenu à l’écart |

Train et dev contiennent les cinq mêmes accents vus. Le test contient
uniquement l’accent indiqué par le nom du fold : Arabic, Chinese, Hindi,
Korean, Spanish ou Vietnamese.

## Contrôles et reproductibilité

La génération vérifie automatiquement :

- la présence des six accents attendus et de quatre locuteurs par accent ;
- l’unicité des paires locuteur-prompt ;
- l’absence de prompts communs entre train, dev et test ;
- l’absence de locuteurs communs entre train, dev et test ;
- l’absence totale de l’accent tenu à l’écart dans train et dev ;
- la présence exclusive de cet accent dans test ;
- la non-vacuité des trois splits.

Chaque manifest enregistre la seed, l’accent tenu à l’écart, les listes de
prompts, les rôles des locuteurs, l’empreinte de l’inventaire source et sa propre
empreinte SHA-256. Le rapport `validation_report.json` doit avoir le statut
`passed` et indiquer zéro recouvrement de prompts et de locuteurs.

Les manifests restent avec les données sous `data/processed/` et ne sont pas
versionnés séparément. Ils sont reproductibles en relançant la commande avec les
mêmes données brutes, le même code et la seed `20260817`. Chaque expérience
devra enregistrer le champ `split_manifest_sha256` afin d’identifier exactement
le fold utilisé.

## Interprétation pour le papier

Le protocole doit être décrit comme une validation croisée stricte à six folds
*leave-one-accent-out*. Pour chaque fold, Stage 2 et toute sélection de
checkpoint utilisent uniquement train et dev ; le test ne doit être évalué
qu’après sélection du modèle.

Le test combine simultanément un accent non vu, des locuteurs non vus et des
prompts non vus. Les résultats démontreront donc une généralisation conjointe à
ces trois facteurs ; ils ne devront pas être présentés comme isolant uniquement
l’effet de l’accent. Il faudra rapporter le WER de chaque accent ainsi que la
macro-moyenne sur les six accents, puis quantifier la variabilité avec les seeds
de modèle prévues.

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
