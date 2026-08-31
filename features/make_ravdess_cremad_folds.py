#!/usr/bin/env python3
import argparse, pickle, numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold


# same idea as mead but using sklearn GroupKFold by actor
def build_si_folds(labels, actors, n_folds=5, seed=42):
    X = np.arange(len(labels))
    gkf = GroupKFold(n_splits=n_folds)
    folds = []
    for fi, (tv, te) in enumerate(gkf.split(X, labels, actors)):
        tv_actors = np.unique(actors[tv])
        rng = np.random.RandomState(seed + fi)
        n_val = max(1, len(tv_actors) // 5)
        # hold out a fifth of the train actors for validation
        val_actors = rng.choice(tv_actors, size=n_val, replace=False)
        val_mask = np.isin(actors[tv], val_actors)
        val_idx, train_idx = tv[val_mask], tv[~val_mask]
        folds.append({
            'fold': fi,
            'train_idx': train_idx.astype(np.int64),
            'val_idx': val_idx.astype(np.int64),
            'test_idx': te.astype(np.int64),
            'train_actors': sorted(int(a) for a in set(actors[train_idx])),
            'val_actors':   sorted(int(a) for a in set(actors[val_idx])),
            'test_actors':  sorted(int(a) for a in set(actors[te])),
        })
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', required=True, choices=['ravdess', 'cremad'])
    ap.add_argument('--cache_dir', default='/parallel_scratch/nz00326/Dissertation/cache')
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    npz = np.load(cache / f'{args.which}_pretrained.npz', allow_pickle=True)
    actors = np.asarray(npz['actors']); labels = np.asarray(npz['labels'])
    print(f'{args.which}: {len(actors)} clips, {len(np.unique(actors))} actors, {len(np.unique(labels))} classes')
    folds = build_si_folds(labels, actors, n_folds=args.n_folds, seed=args.seed)
    all_test = []
    for f in folds:
        tr, va, te = set(f['train_actors']), set(f['val_actors']), set(f['test_actors'])
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), f'leak in fold {f["fold"]}'
        assert len(f['train_idx']) + len(f['val_idx']) + len(f['test_idx']) == len(actors)
        all_test += f['test_actors']
        print(f"  fold {f['fold']}: train={len(f['train_actors'])} val={len(f['val_actors'])} test={len(f['test_actors'])} actors | clips {len(f['train_idx'])}/{len(f['val_idx'])}/{len(f['test_idx'])}")
    assert sorted(all_test) == sorted(int(a) for a in np.unique(actors)), 'every actor test exactly once'
    print('  OK: actor-disjoint, each actor tested once')
    out = cache / f'{args.which}_folds.pkl'
    with open(out, 'wb') as fh: pickle.dump(folds, fh)
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
