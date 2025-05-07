#!/usr/bin/env python3
"""
Convert the Google Quick‑Draw! Sketch‑RNN archives (.npz) into two LMDB
databases + a key‑list pickle expected by Bhunia et al.’s repo.

  raw_npz/airplane.npz   →  QuickDraw/QuickDraw_TrainData  (lmdb)
                          →  QuickDraw/QuickDraw_TestData   (lmdb)
                          →  QuickDraw/QuickDraw_Keys.pickle

Author: you 🖋️
"""

import os, glob, lmdb, pickle, tqdm, numpy as np
from pathlib import Path

RAW_DIR   = Path("raw_npz")               # where 345 *.npz live
OUT_DIR   = Path("QuickDraw")             # output folder
SPLIT     = 0.90                          # 90 % train, 10 % test
MAP_SIZE  = 1 << 40                       # 1 TB – plenty of headroom

OUT_DIR.mkdir(exist_ok=True)
env_tr = lmdb.open(str(OUT_DIR / "QuickDraw_TrainData"), map_size=MAP_SIZE)
env_te = lmdb.open(str(OUT_DIR / "QuickDraw_TestData"),  map_size=MAP_SIZE)

train_keys, test_keys = [], []

for npz_file in tqdm.tqdm(sorted(RAW_DIR.glob("*.npz")),
                          desc="classes", unit="cls"):
    cls = npz_file.stem                           # e.g. "airplane"
    npz = np.load(npz_file, allow_pickle=True, encoding="latin1")
    # Merge Google’s original splits so we can reshuffle
    samples = np.concatenate([npz["train"],
                               npz["valid"],
                               npz["test"]])

    # Deterministic shuffle per class for reproducibility
    rng = np.random.default_rng(seed=hash(cls) & 0xFFFFFFFF)
    rng.shuffle(samples)

    split_idx = int(SPLIT * len(samples))

    for i, stroke in enumerate(samples):
        key_str = f"{cls}_{i:05d}"
        blob    = pickle.dumps(stroke, protocol=4)

        if i < split_idx:
            with env_tr.begin(write=True) as txn:
                txn.put(key_str.encode(), blob)
            train_keys.append(key_str)
        else:
            with env_te.begin(write=True) as txn:
                txn.put(key_str.encode(), blob)
            test_keys.append(key_str)

# Save key lists (validation list left empty to match repo format)
with open(OUT_DIR / "QuickDraw_Keys.pickle", "wb") as f:
    pickle.dump((train_keys, [], test_keys), f, protocol=4)

print(f"✅  Done.  Train {len(train_keys)}  |  Test {len(test_keys)}")
