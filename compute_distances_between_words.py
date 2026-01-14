import pickle
import argparse

from math import sqrt
from pathlib import Path

import pandas as pd
import numpy as np

# from dtw import dtw
from librosa.sequence import dtw

from loguru import logger
from tqdm.auto import tqdm

tqdm.pandas()

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", required=True, type=Path)
parser.add_argument("--repr_column", required=True)
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--output", required=True)

args = parser.parse_args()


print("loading data...")
df = pd.concat(
    [
        pickle.load(open(filename, "rb"))
        for filename in tqdm(list(args.input_dir.glob("*.pkl")))
    ],
    axis=0,
)

# remove empty parts (silences)
df["annotation"] = df["annotation"].str.replace("\t", "")
df = df[df["annotation"] != ""]




# compute normalized representations
# ----------------------------------
#
# XXX representations are normalized at the corpus-level, it would be interesting to see if it makes sense to normalize them at the speaker-level
logger.info("normalize representations")
normalized_col = f"normalized_{args.repr_column}"
X = np.vstack(df[args.repr_column].values) # vstack to create a big matrix of all representations (num_lines x embedding_size (1024 for xlsr53))

mu = X.mean(axis=0)
sigma = X.std(axis=0)

df[normalized_col] = df[args.repr_column].apply(lambda x: (x - mu) / sigma)

if args.normalize:
    repr_column = normalized_col
else:
    repr_column = args.repr_column

logger.warning(f"using representations in {repr_column}")





# compute cross-product between words
# -----------------------------------

logger.info("compute cross product")
# for the moment with have one embedding per row... collect all embeddings corresponding to the same word and stack them together. At the end, each word in each file is associated to a matrix n_frames ✕ embedding_representation
merged_representations = (
    df.groupby(["filename", "speaker", "annotation"])[repr_column]
    .apply(lambda x: np.vstack(x))
    .reset_index()
)

# For each annotation (e.g. word — this is the groupby part): build the cross product of all the embeddings corresponding to that annotation (i.e. all possible combinaisons — the merge part) and stack all resulting dataframes together (the concat part)
pairs = pd.concat(
    [
        group.merge(group, how="cross")
        for _, group in tqdm(merged_representations.groupby("annotation"))
    ],
    axis=0,
)

sample = merged_representations.sample(int(sqrt(pairs.shape[0])))
sample = sample.merge(sample, how="cross").query("annotation_x != annotation_y")

pairs = pd.concat([pairs, sample], axis=0)






def align(x, y):
    from scipy.spatial.distance import cdist

    dist_mat = cdist(x, y, metric="cosine")
    
    # 'step_pattern' définit comment on peut se déplacer dans la matrice
    D, wp = dtw(C=dist_mat)

    # On renvoie la distance finale (le dernier élément de la matrice de coût cumulé)
    return D[-1, -1]


logger.info("compute similarity")
sim = pairs.progress_apply(
    lambda row: align(row[repr_column + "_x"], row[repr_column + "_y"]),
    axis=1,
    result_type="expand",
)

final = pd.concat([sim, pairs], axis=1)

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
final.to_pickle(args.output)