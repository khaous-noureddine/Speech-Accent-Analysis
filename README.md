# Accented Speech Recognition

This project aims to fine-tune pretrained speech models (Wav2Vec2, HuBERT, XLS-R, Whisper) to learn embeddings that capture **linguistic content** while reducing information related to **accent** and **speaker identity**.

Using contrastive / triplet learning, utterances with the same transcript are pulled closer in embedding space, while different transcripts are pushed apart.

## Goals

- Improve robustness to unseen accents  
- Improve cross-speaker generalization  
- Learn content-focused speech representations  

## Datasets

- Speech Accent Archive  
- CMU ARCTIC  
- L2-ARCTIC


## Installation & Usage 