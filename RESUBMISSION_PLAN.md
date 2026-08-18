# Plan de travail — Resoumission

Ce document est notre source de vérité pour préparer la prochaine soumission de
**Accent-Unified Representations for Robust ASR via Supervised Contrastive Learning**.

Il doit être mis à jour après chaque décision importante, expérience lancée ou
résultat obtenu.

## Objectif

Démontrer de manière causale et reproductible que l'adaptation contrastive
supervisée au niveau des prompts améliore la robustesse de l'ASR aux accents,
et distinguer son effet de celui de la supervision CTC et de la simple exposition
à de la parole accentuée.

## État actuel

- Branche de travail : `resubmission-experiments`
- Reviews et manuscrit ajoutés localement dans `6a032b3419d0c09188ff82d2/`
- Données locales : environ 44 Go
- Splits L2-ARCTIC de cross-validation : 8 folds présents
- Calcul local : 4 × NVIDIA RTX A6000 48 Go
- SLURM : indisponible dans l'environnement actuel
- Checkpoints/résultats de cross-validation locaux : aucun identifié

## Critiques prioritaires des reviewers

1. L'effet de SupCon est confondu avec celui de la perte CTC auxiliaire.
2. La généralisation zero-shot à des accents réellement absents de
   l'entraînement n'est pas démontrée clairement.
3. Les résultats reposent principalement sur une seed et un split fixe.
4. Les checkpoints ASR déjà fine-tunés affaiblissent la validité de la baseline
   dite « SSL-to-ASR ».
5. La méthode dépend de corpus parallèles et scriptés où plusieurs locuteurs
   lisent les mêmes phrases.
6. Les comparaisons avec les travaux antérieurs sont insuffisantes.
7. Plusieurs équations, affirmations et détails de reproductibilité doivent être
   corrigés ou clarifiés.

## Matrice expérimentale principale

| Condition | Stage 2 sur L2-ARCTIC | Stage 3 | Question testée |
|---|---|---|---|
| C | aucune | LibriSpeech | Baseline ASR |
| A | SupCon + CTC | LibriSpeech | Méthode complète |
| E | SupCon seul | LibriSpeech | Effet sans supervision CTC |
| F | CTC seule | LibriSpeech | Effet de la supervision accentuée |
| G | aucune | LibriSpeech + données ARCTIC/L2-ARCTIC | Une simple mixture ASR suffit-elle ? |

Comparaisons centrales :

- A vs C : bénéfice global de Stage 2 ;
- A vs F : contribution propre de SupCon ;
- A vs E : contribution de la CTC auxiliaire ;
- A vs G : intérêt de Stage 2 face à une exposition ASR classique aux mêmes données.

## Protocole envisagé

### Initialisation

- Utiliser un checkpoint SSL non fine-tuné pour l'ASR.
- Employer exactement la même initialisation dans toutes les conditions comparées.
- Ne pas mélanger les nouveaux résultats avec les anciens résultats issus de
  checkpoints déjà fine-tunés pour l'ASR.
- Choix exact du checkpoint : **à valider**.

### Répétitions et statistiques

- Backbone principal : Wav2Vec2.
- Conditions prioritaires : C, A, E et F.
- Exécuter au moins 3 seeds par condition.
- Rapporter moyenne, écart-type et intervalles de confiance.
- Effectuer un bootstrap apparié au niveau des utterances pour les différences
  de WER.
- Réserver HuBERT aux expériences essentielles selon le budget disponible.

### Généralisation zero-shot

- Les 8 folds existants ne constituent **pas** une évaluation zero-shot accent :
  chaque L1 est présente dans train, dev et test.
- Le générateur existant partitionne les prompts indépendamment pour chaque L1
  et ne vérifie les fuites qu'à l'intérieur d'une L1. Il doit être remplacé :
  un même prompt ne doit jamais être dans plusieurs splits, même via deux L1
  différentes.
- Construire une partition globale et déterministe des prompts, partagée par
  toutes les L1.
- Séparer strictement les locuteurs de train, dev et test.
- Construire 6 folds leave-one-L1-out, un pour chacune des L1 de L2-ARCTIC.
- Documenter séparément : prompts inconnus, locuteurs inconnus, accents inconnus
  et corpus externes.
- Clarifier la composition exacte des splits AESRC train/dev/test.

#### Protocole L2-ARCTIC proposé

Deux protocoles complémentaires seront conservés :

1. **Split principal speaker-and-prompt-disjoint** : toutes les L1 sont vues
   pendant l'entraînement, mais les locuteurs et prompts de test sont inconnus.
   Les prompts sont répartis globalement une seule fois entre train/dev/test.
2. **Zero-shot accent leave-one-L1-out** : une L1 entière et tous ses locuteurs
   sont absents de Stage 2. Les données de développement proviennent uniquement
   des cinq L1 vues, tandis que le test provient de la L1 tenue à l'écart.

Pour le protocole zero-shot strict, les prompts de test sont également absents
de train et dev. Il mesure donc conjointement la généralisation à une L1, à des
locuteurs et à des prompts inconnus. Une éventuelle évaluation sur prompts vus
devra être rapportée séparément et explicitement comme telle.

Chaque génération doit produire :

- un manifeste versionné contenant la seed de split et les listes de prompts,
  locuteurs et L1 par split ;
- les statistiques en nombre d'exemples, locuteurs, prompts, L1 et heures ;
- un rapport automatique de tous les recouvrements ;
- une empreinte SHA-256 des manifests pour identifier exactement les données
  utilisées par chaque expérience.

### Politique de seeds

- La seed de construction des données est fixe et indépendante des seeds modèle.
- Les splits sont générés une seule fois puis réutilisés par toutes les conditions.
- Les conditions comparées utilisent les mêmes seeds modèle (paired seeds).
- Minimum prévu : seeds modèle `13`, `42` et `77`.
- Chaque checkpoint enregistre la seed, le hash du manifeste, la configuration,
  le checkpoint initial et les versions logicielles.

### Jeux d'évaluation

- LibriSpeech test-clean : contrôle de la performance sur parole standard.
- L2-ARCTIC : robustesse sur parole accentuée parallèle.
- AESRC : robustesse accentuée et, si le split le permet, accents inconnus.
- Speech Accent Archive : généralisation cross-corpus.
- EDACC/AfriSpeech : à évaluer selon disponibilité et adéquation scientifique.

## Plan d'exécution

### Organisation cible du dépôt

La migration doit être incrémentale : les anciens scripts restent disponibles
dans `legacy/` jusqu'à ce que leurs remplaçants aient des tests et produisent un
smoke test valide.

```text
accented-speech-recognition/
├── src/accented_asr/
│   ├── data/                 # schémas, datasets, collators et splits
│   ├── models/               # backbone, projection, CTC et losses
│   ├── training/             # boucles Stage 2 et Stage 3
│   ├── evaluation/           # transcription, WER, bootstrap et agrégation
│   └── reproducibility/      # seeds, manifests et métadonnées de runs
├── scripts/                  # points d'entrée CLI fins
├── workflows/                # workflow Snakemake et règles modulaires
├── configs/
│   ├── base/                 # paramètres communs
│   ├── conditions/           # A, C, E, F et G
│   └── protocols/            # main et leave-one-L1-out
├── manifests/l2_arctic/      # listes d'identifiants et empreintes des splits
├── tests/                    # tests unitaires et d'intégration courts
├── experiments/              # définitions versionnées des études
├── outputs/                  # checkpoints/logs/résultats, ignorés par Git
├── legacy/                   # anciennes implémentations non actives
├── docs/                     # documentation scientifique et technique
└── RESUBMISSION_PLAN.md
```

Les sorties suivent une convention unique :

```text
outputs/<study>/<protocol>/<condition>/<backbone>/fold=<fold>/seed=<seed>/
```

Chaque dossier de run contient au minimum :

- `config.resolved.yaml` ;
- `run_metadata.json` ;
- `split_manifest.sha256` ;
- `metrics.jsonl` ;
- `checkpoints/` ;
- `predictions/` ;
- `scores/`.

### Dette technique identifiée

- `stage2/supcon_train_meanpool.py` contient 2 296 lignes et réimplémente à la
  fois données, modèle, entraînement et évaluation.
- `stage2/supcon_train_l2cv.py` conserve environ 600 lignes d'une ancienne
  implémentation commentée avant son implémentation active.
- `supcon_model.py` et `supcon_xlsr.py` définissent deux versions concurrentes
  des mêmes composants.
- `supcon_data.py` et `supcon_data_l2cv.py` se recouvrent fortement.
- `transcribe.py`/`transcribe_ngram_lm.py` et
  `compute_wer.py`/`wer-hf.py` dupliquent une partie de l'évaluation.
- Le `Snakefile` mélange orchestration locale, paramètres scientifiques et
  commandes SLURM spécifiques au cluster.
- Les configurations historiques utilisent des schémas, chemins de checkpoints
  et modèles initiaux incohérents ; la condition D active ne correspond pas à
  sa définition scientifique.
- `requirements.txt`, `pixi.toml` et `pixi.lock` ne décrivent pas actuellement
  le même environnement complet.
- Aucun répertoire de tests n'est présent.
- Des artefacts `.snakemake/` sont suivis par Git malgré leur présence dans
  `.gitignore`.
- Le manuscrit est un dépôt Git imbriqué non suivi : décider explicitement s'il
  doit rester séparé, devenir un submodule ou être intégré sous `paper/`.
- Le README ne documente ni l'installation complète, ni la préparation des
  données, ni un run reproductible de bout en bout.

### Stratégie de migration du code

1. Construire d'abord le nouveau module de splits et ses tests sans déplacer le
   reste du code.
2. Extraire ensuite loss, modèle, dataset et sampler de Stage 2 dans des modules
   testables, puis faire pointer un unique script d'entraînement vers eux.
3. Ajouter une configuration validée et résolue pour éviter les combinaisons
   scientifiques invalides.
4. Unifier l'évaluation et ajouter l'agrégation multi-seed/multi-fold.
5. Modulariser Snakemake après validation des commandes Python locales.
6. Déplacer les scripts remplacés vers `legacy/` seulement après comparaison
   sur un smoke test identique.

### Phase 1 — Audit et fiabilisation du code

- [ ] Corriger la clé `poolxed` en `pooled` dans `stage2/supcon_model.py`.
- [ ] Unifier ou clarifier les rôles de `supcon_model.py` et `supcon_xlsr.py`.
- [ ] Ajouter un mode explicite de loss : `supcon_ctc`, `supcon_only`, `ctc_only`.
- [ ] Ajouter des validations empêchant une condition mal configurée.
- [ ] Vérifier l'initialisation de la tête CTC et la sauvegarde dans les checkpoints.
- [ ] Vérifier la reproductibilité complète des seeds.
- [ ] Ajouter des tests unitaires pour les pertes et le masked mean pooling.
- [ ] Faire un smoke test GPU court pour chaque mode de loss.
- [ ] Créer un package Python installable sous `src/accented_asr/`.
- [ ] Définir un schéma de configuration unique et validé.
- [ ] Définir la convention de nommage des runs et leurs métadonnées obligatoires.

### Phase 2 — Données et protocole

- [x] Auditer conceptuellement les 8 folds L2-ARCTIC existants.
- [x] Établir qu'ils ne testent pas des L1 inconnues et que leur validation des
  prompts n'est pas globale.
- [x] Ajouter un splitter global, déterministe et
  accompagné d'un manifeste.
- [x] Ajouter des tests automatiques d'absence de fuite globale de prompts,
  locuteurs et L1 selon le protocole.
- [x] Générer un split principal strict speaker-and-prompt-disjoint.
- [x] Générer les 6 folds leave-one-L1-out.
- [x] Auditer statistiquement les nouveaux splits L2-ARCTIC.
- [ ] Auditer les splits AESRC et leurs accents.
- [ ] Choisir et figer le checkpoint SSL de départ.
- [ ] Définir les budgets identiques : steps, durée audio vue et batch effectif.
- [ ] Générer les configurations versionnées A/C/E/F/G.
- [x] Enregistrer les manifests sous `manifests/l2_arctic/` sans chemins absolus.

### Phase 3 — Expériences pilotes

- [ ] Lancer des runs courts A/C/E/F sur un seul fold et une seule seed.
- [ ] Vérifier les courbes SupCon, CTC, WER et l'utilisation mémoire.
- [ ] Confirmer que les checkpoints se rechargent correctement en Stage 3.
- [ ] Choisir les hyperparamètres sans regarder les jeux de test.

### Phase 4 — Expériences principales

- [ ] Lancer C, A, E et F avec 3 seeds.
- [ ] Lancer la condition G.
- [ ] Lancer la validation leave-one-accent/L1-out.
- [ ] Répliquer les comparaisons essentielles avec HuBERT si le budget le permet.
- [ ] Évaluer tous les checkpoints sur les mêmes jeux et avec la même normalisation.

### Phase 5 — Analyse

- [ ] Agréger WER moyen, écart-type et intervalles de confiance.
- [ ] Calculer les tests bootstrap appariés.
- [ ] Produire les résultats par accent et par locuteur.
- [ ] Réviser les métriques d'alignement et préciser l'espace où elles sont calculées.
- [ ] Régénérer les figures uniquement sur des splits held-out clairement identifiés.
- [ ] Comparer le coût de calcul et le nombre de paramètres.

### Phase 6 — Révision du papier

- [ ] Corriger l'équation SupCon pour montrer la moyenne réellement implémentée.
- [ ] Décrire le masked pooling, GELU, vocabulaire et initialisation de la tête CTC.
- [ ] Expliquer comment le batch contrastif tient en mémoire.
- [ ] Remplacer « naturally repeated content » par « scripted parallel prompts ».
- [ ] Reconnaître explicitement la dépendance aux transcriptions de Stage 2.
- [ ] Reformuler les claims « without accented ASR data ».
- [ ] Corriger les contradictions entre le texte et les tableaux HuBERT.
- [ ] Corriger l'interprétation des distances Wav2Vec2.
- [ ] Clarifier les splits utilisés pour les figures et la sélection des checkpoints.
- [ ] Expliquer les différences entre le nombre d'epochs annoncé et les figures.
- [ ] Ajouter Han et al., SCaLa et les baselines d'adaptation pertinentes.
- [ ] Nuancer la comparaison avec Whisper selon la taille et les données d'entraînement.
- [ ] Ajouter le budget GPU total et les détails de reproductibilité.

## Décisions à prendre

- [ ] Checkpoint SSL principal exact.
- [ ] Définition finale de la condition G et quantité de données accentuées utilisée.
- [x] Six folds zero-shot leave-one-L1-out, avec montée progressive de 1 à 3 seeds.
- [ ] Étendue de la réplication HuBERT.
- [ ] Prochaine conférence/cycle et date limite.

## Journal de travail

### 2026-08-17

- Création de la branche `resubmission-experiments` depuis `main`.
- Lecture des reviews et audit initial du manuscrit, des configurations et du code.
- Identification des ablations E/F/G et de la validation zero-shot comme priorités.
- Création de ce document de suivi.
- Décision de figer les splits avant de lancer l'ablation SupCon-only.
- Audit du splitter L2-ARCTIC existant : il tient des locuteurs et prompts à
  l'écart au sein de chaque L1, mais toutes les L1 restent vues et la séparation
  des prompts n'est pas garantie globalement entre L1.
- Décision d'utiliser une partition globale des prompts, un split principal
  strict et six folds leave-one-L1-out.
- Décision de dissocier la seed du split des trois seeds d'entraînement appariées.
- Audit d'architecture du dépôt et définition d'une organisation cible.
- Décision d'effectuer une migration incrémentale, en commençant par les splits
  et les tests avant de consolider Stage 2.
- Implémentation du nouveau cœur de split L2-ARCTIC, d'un CLI Parquet et de
  manifests auto-vérifiés par SHA-256.
- Ajout de 13 tests couvrant déterminisme, séparation globale, leave-one-L1-out
  et rejet des inventaires/manifests invalides.

### 2026-08-18

- L'audit canonique des 26 867 WAV a retenu 953 prompts disposant d'un audio et
  d'une transcription par locuteur, soit 22 872 exemples. `arctic_b0115` a été
  exclu car la transcription de RRBI manque dans les métadonnées processed.
- Génération et versionnement du split principal et des six folds
  leave-one-L1-out. Le split principal contient 9 156/570/570 exemples
  train/dev/test ; chaque fold zero-shot contient 11 445/475/380 exemples.
- Vérification exhaustive des 22 872 en-têtes audio et de la conformité entre
  inventaire, manifests, Parquet, hashes et rapports anti-fuite.
