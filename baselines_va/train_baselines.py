import os
import csv
import json
import time
import platform
import argparse
import subprocess
import numpy as np
import torch

from .config import build_args, NUM_CLASSES, TRAINED_METHODS, ALL_METHODS, DISPLAY_NAME
from .dataset import (load_npz, load_folds, encode_labels, make_dataloaders,
                      load_label_classes)
from .model import build_model
from .trainer import Trainer
from .utils.metrics import (setup_seed, configure_tf32, count_parameters,
                            count_parameters_total, count_macs,
                            classification_metrics)


# save per fold and averaged results for one method
def _write_method_csvs(fold_rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'per_fold_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        w.writeheader()
        w.writerows(fold_rows)
    metrics = [m for m in fold_rows[0].keys() if m != 'fold']
    with open(os.path.join(out_dir, 'aggregate_results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'mean', 'std'])
        for m in metrics:
            vals = [r[m] for r in fold_rows if r.get(m) is not None]
            if vals:
                w.writerow([m, round(float(np.mean(vals)), 5), round(float(np.std(vals)), 5)])
            else:
                w.writerow([m, '', ''])


# save the per clip predictions of the score average
def _write_score_avg_samples(diag_dir, fold, idx, y_true, avg_probs, class_names):
    os.makedirs(diag_dir, exist_ok=True)
    pred = avg_probs.argmax(1)
    conf = avg_probs.max(1)
    K = avg_probs.shape[1]
    rows = []
    for b in range(len(idx)):
        t, p = int(y_true[b]), int(pred[b])
        nll = -np.log(max(float(avg_probs[b, t]), 1e-12))
        row = {'orig_index': int(idx[b]), 'true_idx': t, 'pred_idx': p,
               'true_name': class_names[t] if class_names else t,
               'pred_name': class_names[p] if class_names else p,
               'correct': int(t == p), 'confidence': round(float(conf[b]), 5),
               'ce': round(float(nll), 5)}
        for k in range(K):
            row[f'p{k}'] = round(float(avg_probs[b, k]), 5)
        rows.append(row)
    with open(os.path.join(diag_dir, f'fold{fold}_test_samples.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    Trainer.write_confusion(y_true, pred, K, diag_dir, fold)


# average a value over the folds
def _aggr_mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.mean(vals)), 4) if vals else ''


# save timing and parameter counts
def _write_efficiency(efficiency, per_method_folds, ds_root):
    cols = ['method', 'params_trainable', 'params_total', 'macs_per_sample',
            'latency_bs1_ms_median', 'latency_bs1_ms_p95', 'throughput_bs64_sps',
            'peak_infer_mem_mb', 'train_time_s_mean', 'time_to_best_s_mean',
            'epochs_run_mean', 'peak_train_mem_mb_mean']
    rows = []
    for m in TRAINED_METHODS:
        if m not in efficiency:
            continue
        e, pf = efficiency[m], per_method_folds[m]
        rows.append({'method': m,
                     'params_trainable': e['params_trainable'], 'params_total': e['params_total'],
                     'macs_per_sample': e['macs_per_sample'],
                     'latency_bs1_ms_median': e['latency_bs1_ms_median'],
                     'latency_bs1_ms_p95': e['latency_bs1_ms_p95'],
                     'throughput_bs64_sps': e.get('throughput_bs64_sps'),
                     'peak_infer_mem_mb': e['peak_infer_mem_mb'],
                     'train_time_s_mean': _aggr_mean(pf, 'train_time_s'),
                     'time_to_best_s_mean': _aggr_mean(pf, 'time_to_best_s'),
                     'epochs_run_mean': _aggr_mean(pf, 'epochs_run'),
                     'peak_train_mem_mb_mean': _aggr_mean(pf, 'peak_train_mem_mb')})
    if 'audio' in efficiency and 'video' in efficiency:
        ea, ev = efficiency['audio'], efficiency['video']
        ta, tv = ea.get('throughput_bs64_sps'), ev.get('throughput_bs64_sps')
        thr = round(1.0 / (1.0 / ta + 1.0 / tv), 2) if ta and tv else ''
        pa, pv = ea['peak_infer_mem_mb'], ev['peak_infer_mem_mb']
        rows.append({'method': 'score_avg',
                     'params_trainable': ea['params_trainable'] + ev['params_trainable'],
                     'params_total': ea['params_total'] + ev['params_total'],
                     'macs_per_sample': ea['macs_per_sample'] + ev['macs_per_sample'],
                     'latency_bs1_ms_median': round(ea['latency_bs1_ms_median'] + ev['latency_bs1_ms_median'], 4),
                     'latency_bs1_ms_p95': round(ea['latency_bs1_ms_p95'] + ev['latency_bs1_ms_p95'], 4),
                     'throughput_bs64_sps': thr,
                     'peak_infer_mem_mb': (max(pa, pv) if pa is not None and pv is not None else ''),
                     'train_time_s_mean': '', 'time_to_best_s_mean': '',
                     'epochs_run_mean': '', 'peak_train_mem_mb_mean': ''})
    path = os.path.join(ds_root, 'efficiency.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})
    return path


# record the current git commit for reproducibility
def _git_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# save a json describing how the run was set up
def _write_metadata(args, data, folds, labels_int, class_names, efficiency,
                    ds_root, seed, allow_tf32, device, num_workers):
    cuda = str(device).startswith('cuda')
    classes, counts = np.unique(data['labels'], return_counts=True)
    fold_sizes = [{'fold': int(f.get('fold', k)), 'train': len(f['train_idx']),
                   'val': len(f['val_idx']), 'test': len(f['test_idx'])}
                  for k, f in enumerate(folds)]
    meta = {
        'dataset': args.dataset, 'num_classes': int(args.num_classes),
        'n_samples': int(len(labels_int)), 'n_folds': len(folds),
        'class_names': [str(c) for c in class_names],
        'class_distribution': {str(c): int(n) for c, n in zip(classes, counts)},
        'fold_sizes': fold_sizes,
        'seed': int(seed), 'device': str(device), 'num_workers': num_workers,
        'allow_tf32': bool(allow_tf32),
        'tf32': {'matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
                 'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32)},
        'config': {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                   for k, v in vars(args).items() if k != 'device'},
        'environment': {
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'cudnn': (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
            'gpu': (torch.cuda.get_device_name(0) if cuda else None),
            'python': platform.python_version(),
            'platform': platform.platform()},
        'code_version': _git_hash(),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'efficiency': efficiency,
    }
    path = os.path.join(ds_root, 'run_metadata.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    return path


# train the baselines for all folds and write the results
def run(dataset, cache_dir, seed=42, device=None, out_root='./experiments/baselines',
        num_workers=2, overrides=None, allow_tf32=False):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')


    configure_tf32(allow_tf32)
    args = build_args(dataset, cache_dir, overrides)
    args.device = device

    data = load_npz(cache_dir, dataset)
    folds = load_folds(cache_dir, dataset)
    labels_int, enc_classes = encode_labels(data['labels'])
    enc_classes = [str(c) for c in enc_classes]
    joblib_classes = load_label_classes(cache_dir, dataset)
    class_names = joblib_classes if (joblib_classes is not None
                                     and len(joblib_classes) == len(enc_classes)) else enc_classes
    assert len(class_names) == NUM_CLASSES[dataset], \
        f"{dataset}: found {len(class_names)} classes, expected {NUM_CLASSES[dataset]}"
    K = args.num_classes
    ds_root = os.path.join(out_root, dataset)

    per_method_folds = {m: [] for m in ALL_METHODS}
    efficiency = {}
    sample_batch = None

    for k, fold in enumerate(folds):
        loaders = make_dataloaders(
            data, fold['train_idx'], fold['val_idx'], fold['test_idx'],
            labels_int, batch_size=args.batch_size, num_workers=num_workers)

        fold_test, fold_val = {}, {}
        for m in TRAINED_METHODS:
            setup_seed(seed + k)
            model = build_model(m, feature_dim=args.feature_dim,
                                hidden_dim=args.hidden_dim, num_classes=K,
                                dropout=args.dropout)
            if k == 0:
                print(f"[{dataset}] {m:<7} params: {count_parameters(model):,}")
            diag = os.path.join(ds_root, m, 'diagnostics')
            os.makedirs(diag, exist_ok=True)

            trainer = Trainer(args)
            trainer.do_train(model, loaders, diag_dir=diag, fold=k, class_names=class_names)
            model.load_state_dict(torch.load(os.path.join(diag, f'fold{k}_best.pth'),
                                             map_location=device))
            test = trainer.do_test(model, loaders['test'], collect_samples=True,
                                   class_names=class_names)


            val = trainer.do_test(model, loaders['valid'], collect_samples=True,
                                  class_names=class_names)
            trainer.write_test_samples(test['sample_rows'], diag, k)
            trainer.write_val_samples(val['sample_rows'], diag, k)
            trainer.write_confusion(test['y_true'], test['y_pred'], K, diag, k)

            per_method_folds[m].append({
                'fold': k, 'Val_acc': val['Acc'], 'Acc': test['Acc'],
                'F1_macro': test['F1_macro'], 'F1_weighted': test['F1_weighted'],
                'test_loss': test['Loss'],
                'train_time_s': trainer.train_time_s, 'epochs_run': trainer.epochs_run,
                'time_to_best_s': trainer.time_to_best_s,
                'peak_train_mem_mb': trainer.peak_train_mem_mb})


            if k == 0:
                if sample_batch is None:
                    sample_batch = next(iter(loaders['test']))
                v1 = sample_batch['vision'][:1].to(device)
                a1 = sample_batch['audio'][:1].to(device)
                l1 = sample_batch['audio_len'][:1].to(device)
                bench = trainer.benchmark(model, sample_batch)
                efficiency[m] = {
                    'params_trainable': count_parameters(model),
                    'params_total': count_parameters_total(model),
                    'macs_per_sample': count_macs(model, (v1, a1, l1)),
                    **bench}
            fold_test[m] = {'orig_index': test['orig_index'], 'probs': test['probs'],
                            'y_true': test['y_true']}
            fold_val[m] = {'orig_index': val['orig_index'], 'probs': val['probs'],
                           'y_true': val['y_true']}
            print(f"  fold {k}  {m:<7} val={val['Acc']:.4f}  test={test['Acc']:.4f}"
                  f"  F1_macro={test['F1_macro']:.4f}")


        a, v = fold_test['audio'], fold_test['video']
        assert np.array_equal(a['orig_index'], v['orig_index']), \
            "audio/video test order differ -- cannot average per-sample probabilities"
        avg = 0.5 * a['probs'] + 0.5 * v['probs']
        y_true = a['y_true']
        sa = classification_metrics(avg, y_true)
        nll = float(-np.log(np.clip(avg[np.arange(len(y_true)), y_true], 1e-12, 1.0)).mean())

        av, vv = fold_val['audio'], fold_val['video']
        assert np.array_equal(av['orig_index'], vv['orig_index']), \
            "audio/video val order differ -- cannot average per-sample probabilities"
        avg_val = 0.5 * av['probs'] + 0.5 * vv['probs']
        val_acc = classification_metrics(avg_val, av['y_true'])['Acc']
        per_method_folds['score_avg'].append({
            'fold': k, 'Val_acc': val_acc, 'Acc': sa['Acc'], 'F1_macro': sa['F1_macro'],
            'F1_weighted': sa['F1_weighted'], 'test_loss': round(nll, 4)})
        _write_score_avg_samples(os.path.join(ds_root, 'score_avg', 'diagnostics'),
                                 k, a['orig_index'], y_true, avg, class_names)
        print(f"  fold {k}  {'score_avg':<7} val={val_acc:.4f}  test={sa['Acc']:.4f}"
              f"  F1_macro={sa['F1_macro']:.4f}")


    for m in ALL_METHODS:
        _write_method_csvs(per_method_folds[m], os.path.join(ds_root, m))

    os.makedirs(ds_root, exist_ok=True)
    summary_path = os.path.join(ds_root, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'display_name', 'Val_acc_mean', 'Val_acc_std',
                    'Acc_mean', 'Acc_std', 'F1_macro_mean', 'F1_macro_std',
                    'F1_weighted_mean'])
        for m in ALL_METHODS:
            rows = per_method_folds[m]
            va = [r['Val_acc'] for r in rows]
            acc = [r['Acc'] for r in rows]
            fm = [r['F1_macro'] for r in rows]
            fw = [r['F1_weighted'] for r in rows]
            w.writerow([m, DISPLAY_NAME[m],
                        round(float(np.mean(va)), 5), round(float(np.std(va)), 5),
                        round(float(np.mean(acc)), 5), round(float(np.std(acc)), 5),
                        round(float(np.mean(fm)), 5), round(float(np.std(fm)), 5),
                        round(float(np.mean(fw)), 5)])

    print(f"\n[{dataset}] baseline summary ({len(folds)}-fold):")
    for m in ALL_METHODS:
        va = [r['Val_acc'] for r in per_method_folds[m]]
        acc = [r['Acc'] for r in per_method_folds[m]]
        print(f"  {DISPLAY_NAME[m]:<32} val = {np.mean(va):.4f}   "
              f"test = {np.mean(acc):.4f} +/- {np.std(acc):.4f}")
    print(f"  -> {summary_path}")

    eff_path = _write_efficiency(efficiency, per_method_folds, ds_root)
    meta_path = _write_metadata(args, data, folds, labels_int, class_names, efficiency,
                                ds_root, seed, allow_tf32, device, num_workers)
    print(f"  -> {eff_path}")
    print(f"  -> {meta_path}")
    return per_method_folds


# read the command line and start the run
def main():
    p = argparse.ArgumentParser(description="Train V/A fusion baselines on RAVDESS / CREMA-D")
    p.add_argument('--dataset', required=True,
                   choices=['ravdess', 'cremad', 'mead8', 'mead6'])
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_root', default='./experiments/baselines')
    p.add_argument('--num_workers', type=int, default=2)

    p.add_argument('--learning_rate', type=float, default=None)
    p.add_argument('--hidden_dim', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--max_epochs', type=int, default=None)
    p.add_argument('--early_stop', type=int, default=None)
    p.add_argument('--allow_tf32', action='store_true',
                   help="allow TF32 matmul/cuDNN (faster, lower precision); off "
                        "by default so results are full FP32 and identical across "
                        "GPUs such as A100 vs A4000")
    args = p.parse_args()

    overrides = {k: v for k, v in dict(
        learning_rate=args.learning_rate, hidden_dim=args.hidden_dim,
        dropout=args.dropout, weight_decay=args.weight_decay,
        batch_size=args.batch_size, max_epochs=args.max_epochs,
        early_stop=args.early_stop).items() if v is not None}

    run(args.dataset, args.cache_dir, seed=args.seed, out_root=args.out_root,
        num_workers=args.num_workers, overrides=overrides,
        allow_tf32=args.allow_tf32)


if __name__ == '__main__':
    main()
