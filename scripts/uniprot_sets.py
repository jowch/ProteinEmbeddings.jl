import os

from numpy.random import choice
from Bio import SeqIO


SEED = 1111
DATA_DIR = "/data/databases/uniprot"
SETS_DIR = os.path.join(DATA_DIR, "sets")

SPROT_COUNT = 565_928
TREMBL_COUNT = 225_013_025


sets = ['train', 'val', 'test']

for name in sets:
    if not os.path.isdir(os.path.join(SETS_DIR, name)):
        os.makedirs(os.path.join(SETS_DIR, name), exist_ok=True)


weights = [0.7, 0.1, 0.2]
assignments = choice([0, 1, 2], size=(SPROT_COUNT + TREMBL_COUNT),
                     replace=False, p=weights)
