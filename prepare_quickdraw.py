#!/usr/bin/env python
"""prepare_quickdraw.py (rev-13)
Generate QuickDraw LMDBs for Bhunia et al.'s repo, storing raw float32 vectors (no PNGs) so that
dataset code (`Dataset.py`) can read raw buffers directly.
Outputs:
  QuickDraw_TrainData/
  QuickDraw_TestData/
  QuickDraw_Keys.pickle

Usage:
  python prepare_quickdraw.py --out_dir QuickDraw [--num_categories N] [--limit_per_cat N]
"""
import argparse
import os
import random
import lmdb
import pickle
from tqdm import tqdm
from quickdraw import QuickDrawData, QuickDrawDataGroup
import numpy as np

# Config
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.1, 0.2
random.seed(42)


def get_category_names(n=None):
    """Return first n QuickDraw category names, or all if n is None."""
    qd = QuickDrawData()
    try:
        names = qd.drawing_names
    except AttributeError:
        names = qd.all_categories
    return names if n is None else names[:n]


def strokes_to_array(strokes):
    """Convert strokes to an (M,3) numpy array of float32 [x,y,flag]."""
    pts = []
    for stroke in strokes:
        if isinstance(stroke, (list, tuple)) and len(stroke) == 2:
            xs, ys = stroke
        elif hasattr(stroke, 'x') and hasattr(stroke, 'y'):
            xs, ys = stroke.x, stroke.y
        else:
            continue
        for j, (x, y) in enumerate(zip(xs, ys)):
            flag = 1.0 if j == 0 else 0.0
            pts.append([x, y, flag])
    if not pts:
        return np.zeros((0,3), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def open_env(path):
    os.makedirs(path, exist_ok=True)
    return lmdb.open(path, map_size=30 * 1024 ** 3, subdir=True)


def main():
    p = argparse.ArgumentParser(description="Prepare QuickDraw LMDBs (raw float32 vectors)")
    p.add_argument('--out_dir', default='QuickDraw', help='Output directory')
    p.add_argument('--num_categories', type=int, default=None,
                   help='Number of categories (max 345); default=None -> all')
    p.add_argument('--limit_per_cat', type=int, default=None,
                   help='Limit drawings per category for debugging')
    args = p.parse_args()

    train_env = open_env(os.path.join(args.out_dir, 'QuickDraw_TrainData'))
    test_env = open_env(os.path.join(args.out_dir, 'QuickDraw_TestData'))

    categories = get_category_names(args.num_categories)
    total = len(categories)
    train_keys, val_keys, test_keys = [], [], []

    done_file = os.path.join(args.out_dir, '.done_categories')
    processed = set()
    if os.path.exists(done_file):
        processed = set(line.strip() for line in open(done_file))

    for idx_cat, cat in enumerate(tqdm(categories, desc='Categories', unit='cat'), 1):
        if cat in processed:
            tqdm.write(f"[{idx_cat}/{total}] Skipping {cat}")
            continue
        tqdm.write(f"[{idx_cat}/{total}] Processing {cat}")
        draws = QuickDrawDataGroup(cat).drawings
        if args.limit_per_cat:
            draws = draws[:args.limit_per_cat]

        per_cat_count = 0
        for d in tqdm(draws, desc=f" {cat}", leave=False):
            arr = strokes_to_array(d.strokes)
            if arr.size == 0:
                continue
            key_str = f"{idx_cat:03d}_{per_cat_count:08d}"
            per_cat_count += 1
            # choose split
            r = random.random()
            env, kl = (train_env, train_keys) if r < TRAIN_RATIO else \
                      (train_env, val_keys) if r < TRAIN_RATIO+VAL_RATIO else \
                      (test_env, test_keys)
            # write raw bytes
            with env.begin(write=True) as txn:
                txn.put(key_str.encode(), arr.tobytes())
            kl.append(key_str)

        # mark done
        with open(done_file, 'a') as df:
            df.write(cat + '\n')
        processed.add(cat)
        tqdm.write(f" -> Marked {cat}")

    # save keys
    with open(os.path.join(args.out_dir, 'QuickDraw_Keys.pickle'), 'wb') as fk:
        pickle.dump([train_keys, val_keys, test_keys], fk)
    tqdm.write(f"Done: {len(train_keys)} train, {len(val_keys)} val, {len(test_keys)} test")

if __name__ == '__main__':
    main()
