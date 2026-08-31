#!/usr/bin/env python3
import argparse, os, pickle, numpy as np
from pathlib import Path

# split actors into folds so no actor is in more than one set
def make_folds(actors_per_clip, n_folds=5, val_frac=0.15, seed=42):
    rng = np.random.RandomState(seed)
    uniq = np.unique(actors_per_clip)
    # shuffle actors then chop into n_folds test groups
    perm = rng.permutation(uniq)
    test_groups = [list(g) for g in np.array_split(perm, n_folds)]
    folds = []
    for k in range(n_folds):
        test_actors = list(test_groups[k])
        non_test = [a for a in perm if a not in test_actors]
        # take some of the leftover actors for validation
        n_val = max(1, int(round(len(non_test) * val_frac)))
        val_actors = list(non_test[:n_val])
        train_actors = [a for a in non_test if a not in val_actors]
        train_set, val_set, test_set = set(train_actors), set(val_actors), set(test_actors)
        train_idx = np.where(np.isin(actors_per_clip, list(train_set)))[0]
        val_idx   = np.where(np.isin(actors_per_clip, list(val_set)))[0]
        test_idx  = np.where(np.isin(actors_per_clip, list(test_set)))[0]
        assert not (train_set & val_set) and not (train_set & test_set) and not (val_set & test_set)
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(actors_per_clip)
        folds.append({'fold': k,
            'train_idx': train_idx.astype(np.int64), 'val_idx': val_idx.astype(np.int64),
            'test_idx': test_idx.astype(np.int64),
            'train_actors': sorted(int(a) for a in train_actors),
            'val_actors': sorted(int(a) for a in val_actors),
            'test_actors': sorted(int(a) for a in test_actors)})
    return folds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', required=True, choices=['mead8', 'mead6'])
    ap.add_argument('--cache_dir', default=os.environ.get('CACHE_DIR', './cache'))
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    npz = np.load(cache / f'{args.which}_pretrained.npz', allow_pickle=True)
    actors = np.asarray(npz['actors']); labels = np.asarray(npz['labels'])
    print(f'{args.which}: {len(actors)} clips, {len(np.unique(actors))} actors, {len(np.unique(labels))} classes')
    folds = make_folds(actors, n_folds=args.n_folds, val_frac=args.val_frac, seed=args.seed)
    all_test = []
    for f in folds:
        all_test += f['test_actors']
        print(f"  fold {f['fold']}: train={len(f['train_actors'])} val={len(f['val_actors'])} test={len(f['test_actors'])} actors | clips {len(f['train_idx'])}/{len(f['val_idx'])}/{len(f['test_idx'])}")
    assert sorted(all_test) == sorted(int(a) for a in np.unique(actors)), 'every actor test exactly once'
    print('  OK: actor-disjoint, each actor tested once')
    out = cache / f'{args.which}_folds.pkl'
    with open(out, 'wb') as fh: pickle.dump(folds, fh)
    print(f'  wrote {out}')

if __name__ == '__main__':
    main()
