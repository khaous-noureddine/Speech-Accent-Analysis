# Pipeline Supervised Contrastive Learning — Documentation

## Vue d'ensemble

Le but est d'entraîner XLSR-53 à produire des embeddings **invariants à l'accent** :
la même phrase prononcée par des locuteurs d'accents différents doit être proche dans l'espace vectoriel,
indépendamment de qui la prononce.

---

## 1. Les données

### Corpus utilisés

| Corpus | Speakers | Rôle |
|--------|----------|------|
| CMU Arctic | 6 speakers natifs américains | Anchor natifs |
| L2-Arctic | 24 speakers L2 (6 L1 différents) | Accents non-natifs |

**Total** : ~28,271 exemples, 1131 utterances valides (avec ≥ 2 speakers différents).

### Structure d'un exemple

Chaque exemple dans le dataset est un triplet :
```
(utterance_id, speaker_id, audio)

ex: ("arctic_a0001", "US_AWB", tensor[160000])
```

Le `label` associé à chaque exemple = index entier de l'`utterance_id` :
```python
utt2label = {"arctic_a0001": 0, "arctic_a0002": 1, ...}
```

---

## 2. Construction des batchs — SupConBatchSampler

### Principe

Le sampler ne choisit **pas** des anchors individuels. Il construit des batchs structurés :

```
K utterances × S speakers = B exemples par batch

K = 15 utterances  (choisies aléatoirement parmi les 1131 éligibles)
S = 20 speakers    (choisis aléatoirement parmi les 25 disponibles)
B = 15 × 20 = 300 exemples par batch
```

### Ce que retourne le sampler

```python
yield [idx_0, idx_1, ..., idx_299]  # 300 indices dans le dataset
```

Le DataLoader charge ensuite `dataset[idx]` pour chaque indice et le collate_fn assemble le batch final.

### Contenu d'un batch

```
exemples 0-19   : utterance_A, 20 speakers différents → label=0
exemples 20-39  : utterance_B, 20 speakers différents → label=1
...
exemples 280-299: utterance_O, 20 speakers différents → label=14
```

### Pourquoi 100 batchs suffit

Le nombre de combinaisons possibles est astronomique :
```
C(1131, 15) × C(25, 20)^15 >> 1 milliard
```

Avec 100 batchs par epoch, on ne verra jamais deux batchs identiques.
Chaque batch génère ~45,000 paires uniques → 100 batchs = ~4.5M paires par epoch.

---

## 3. Forward pass

```
batch["audio"]  [300, T_audio]
      ↓
CNN feature extractor          → [300, T_frames, 1024]
      ↓
Transformer XLSR-53
  layers 0-17  : gelés
  layers 18-23 : fine-tunés
      ↓
hidden_states                  → [300, T_frames, 1024]
      ↓
CTC head (auxiliaire)          → [300, T_frames, 32]
      ↓
Mean pooling                   → [300, 1024]
      ↓
Projection MLP                 → [300, 256]   L2-normalisés
      ↓
embeddings                     → [300, 256]   ← entrée de SupCon loss
```

---

## 4. SupCon Loss — comment ça marche

### Les labels

```python
batch["labels"] = [0, 0, ..., 0,   # 20 fois → utterance_A
                   1, 1, ..., 1,   # 20 fois → utterance_B
                   ...
                   14, 14,...,14]  # 20 fois → utterance_O
# shape : [300]  — un label par exemple, pas un par batch
```

### La matrice de similarité

```python
sim = embeddings @ embeddings.T / temperature  # [300, 300]
```

Chaque case `sim[i, j]` = similarité cosine entre l'exemple `i` et l'exemple `j`.

### Le masque positifs/négatifs

```python
labels = labels.unsqueeze(1)              # [300, 1]
pos_mask = (labels == labels.T).float()  # [300, 300]
```

Ce qui donne une matrice **bloc-diagonale** :

```
      utt_A          utt_B        ...    utt_O
utt_A [ T T T ... T | F F F ... F | ... | F F F ]
utt_A [ T T T ... T | F F F ... F | ... | F F F ]
utt_B [ F F F ... F | T T T ... T | ... | F F F ]
utt_B [ F F F ... F | T T T ... T | ... | F F F ]
...
```

`T` = positif (même utterance, speaker différent)  
`F` = négatif (utterance différente)

### Calcul de la loss

**Tous les 300 exemples sont anchor simultanément** — chaque ligne de la matrice correspond à un anchor.

Pour chaque anchor `i` :
```
loss_i = -1/|positifs| · Σ_{p ∈ positifs} log( exp(sim[i,p]) / Σ_{j≠i} exp(sim[i,j]) )
```

Loss finale :
```python
loss = mean(loss_i for all i with at least one positive)
```

### Intuition géométrique

- Le **numérateur** pousse l'anchor vers ses positifs (même utterance, accent différent)
- Le **dénominateur** pousse l'anchor loin de ses négatifs (utterances différentes)
- Résultat : les embeddings de la même phrase se regroupent, quelle que soit l'accent

---

## 5. Loss combinée

```python
loss = L_supcon + λ · L_ctc

λ = 0.1  (hyperparamètre)
```

La **CTC loss** est auxiliaire — elle empêche le modèle de "jeter" l'information linguistique
en optimisant uniquement l'invariance à l'accent (ce qu'on appelle le phonemic collapse).

---

## 6. Hyperparamètres clés

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `k_utterances` | 15 | Utterances par batch |
| `s_speakers` | 20 | Speakers par utterance |
| `n_batches` | 100 | Batchs par epoch |
| `temperature τ` | 0.1 | Dureté des négatifs dans SupCon |
| `λ_ctc` | 0.1 | Poids de la loss CTC auxiliaire |
| `freeze_layers` | 0-17 | Layers gelés (18-23 fine-tunés) |
| `lr` | 2e-5 | Learning rate |
| `epochs` | 50 | Nombre d'epochs |
| `proj_dim` | 1024→512→256 | Dimensions du MLP de projection |

---

## 7. Conditions expérimentales

| Condition | Stage 2 | Stage 3 data | Objectif |
|-----------|---------|--------------|---------|
| **A** (proposée) | SupCon fine-tuning | LibriSpeech clean | Test de l'invariantisation seule |
| **B** (upper bound) | SupCon fine-tuning | AESRC2020 accented | Combien reste-t-il d'accent sensitivity ? |
| **C** (baseline) | Aucun | LibriSpeech clean | Transfert standard sans invariantisation |
| **D** (oracle) | Aucun | AESRC2020 accented | Meilleur WER possible in-distribution |

**Gap A→C** = contribution de Stage 2 seul  
**Gap A→B** = accent sensitivity résiduelle après invariantisation  
**Gap C→D** = coût de ne pas avoir de données accentuées sans invariantisation

---

## 8. Métriques d'évaluation Stage 2

| Métrique | Direction | Signification |
|----------|-----------|---------------|
| `alignment_pos` | ↓ | Distance L2² entre paires positives |
| `alignment_neg` | ↑ | Distance L2² entre paires négatives |
| `alignment_ratio` | ↓ | pos/neg — signal principal d'entraînement |
| `alignment_cos` | ↑ | Similarité cosine entre positifs |
| `uniformity` | ↓ | Couverture de la sphère (pas de collapse) |
| `retrieval@K` | ↑ | Est-ce que les K plus proches voisins sont la même utterance ? |
| `accent_probe` | ↓ | Précision d'un classifieur linéaire d'accent (doit baisser) |
| `utterance_probe` | ↑ | Précision d'un classifieur de phrase (doit rester haut) |