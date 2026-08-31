import os
import sys
import argparse
import json
import logging
import pickle
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch

# emoe code uses top level imports so add its folder to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.emoe_repo import EMOE as EMOERepo
from utils.emoe_trainer import EMOE as EMOETrainer
from utils.functions import setup_seed
from dataset import load_npz, encode_labels, make_dataloaders



try:
    from utils import plots as P
    HAS_PLOTS = True
except Exception as _e:
    HAS_PLOTS = False
    _PLOT_ERR = str(_e)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   choices=['ravdess', 'cremad', 'mead8', 'mead6'])
    p.add_argument('--cache_dir',
                   default=os.environ.get('CACHE_DIR', './cache'),
                   help='dir holding {dataset}_pretrained.npz and {dataset}_folds.pkl')
    p.add_argument('--exp_root', default='./experiments')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--folds', default='all',
                   help="comma-separated fold indices, e.g. '0,1' (default 'all')")
    p.add_argument('--subsample', type=int, default=0,
                   help='if >0, cap each split at this many examples (smoke test)')


    p.add_argument('--exp', default='',
                   help='experiment folder name; auto-generated from the '
                        'config when empty')
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--update_epochs', type=int, default=10)
    p.add_argument('--embed_dim', type=int, default=256)
    p.add_argument('--output_dropout', type=float, default=0.5)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--early_stop', type=int, default=10)
    p.add_argument('--conv_kernel', type=int, default=5, choices=[1, 5])
    p.add_argument('--no_plots', action='store_true', default=False,
                   help='skip diagnostics figures (used during the sweep)')
    args = p.parse_args()


    args.variant = 'repo'
    args.batch_size = 16
    args.num_heads = 8
    args.nlevels = 4
    args.fusion_method = 'sum'
    args.attn_mask = True

    args.attn_dropout = 0.3
    args.attn_dropout_a = 0.2
    args.attn_dropout_v = 0.0
    args.relu_dropout = 0.0
    args.embed_dropout = 0.2
    args.res_dropout = 0.0


    args.len_v = 30
    args.len_a = 250
    args.audio_dim = 768
    args.video_dim = 768

    args.patience = 5
    args.max_epochs = 50
    args.label_smoothing = 0.0
    args.use_class_weights = False
    args.eval_test_in_train = False
    args.KeyEval = 'Loss'
    args.train_mode = 'classification'
    args.model_name = 'EMOE'

    if not args.exp:
        is_official = (args.learning_rate == 1e-4 and args.update_epochs == 10
                       and args.embed_dim == 256 and args.output_dropout == 0.5
                       and args.temperature == 0.1 and args.early_stop == 10
                       and args.conv_kernel == 5 and args.seed == 42)
        args.exp = ('run3_official' if is_official else
                    f'r3_lr{args.learning_rate:g}_ue{args.update_epochs}'
                    f'_d{args.embed_dim}_do{args.output_dropout:g}'
                    f'_t{args.temperature:g}_es{args.early_stop}'
                    f'_k{args.conv_kernel}_s{args.seed}')
    return args


def build_args_for_model(args, num_classes, fold_idx):
    class A: pass
    a = A()
    a.dst_feature_dim_nheads = (args.embed_dim, args.num_heads)
    a.feature_dims = (args.audio_dim, args.video_dim)
    a.num_classes = num_classes
    a.len_v = args.len_v
    a.len_a = args.len_a
    a.need_data_aligned = False
    a.nlevels = args.nlevels
    a.attn_dropout = args.attn_dropout
    a.attn_dropout_a = args.attn_dropout_a
    a.attn_dropout_v = args.attn_dropout_v
    a.relu_dropout = args.relu_dropout
    a.embed_dropout = args.embed_dropout
    a.res_dropout = args.res_dropout
    a.output_dropout = args.output_dropout
    a.attn_mask = args.attn_mask
    a.fusion_method = args.fusion_method
    a.conv1d_kernel_size_a = args.conv_kernel
    a.conv1d_kernel_size_v = args.conv_kernel
    a.temperature = args.temperature
    a.variant = args.variant

    a.learning_rate = args.learning_rate
    a.patience = args.patience
    a.early_stop = args.early_stop
    a.max_epochs = args.max_epochs
    a.update_epochs = args.update_epochs
    a.label_smoothing = args.label_smoothing
    a.class_weights = None
    a.eval_test_in_train = args.eval_test_in_train
    a.KeyEval = args.KeyEval
    a.train_mode = args.train_mode
    a.dataset_name = 'classification'
    a.model_name = args.model_name
    a.cur_seed = args.seed + fold_idx
    a.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a.checkpoint_dir = os.path.join(args.out_dir, 'ckpt', args.dataset,
                                    f'fold{fold_idx}')
    return a


def main():
    args = get_args()
    args.out_dir = os.path.join(args.exp_root, args.exp)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump({k: v for k, v in vars(args).items()
                   if isinstance(v, (str, int, float, bool, type(None)))},
                  f, indent=2, sort_keys=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler()])
    logger = logging.getLogger('EMOE')


    npz_path = os.path.join(args.cache_dir, f'{args.dataset}_pretrained.npz')
    fold_path = os.path.join(args.cache_dir, f'{args.dataset}_folds.pkl')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    if not os.path.exists(fold_path):
        raise FileNotFoundError(fold_path)

    logger.info(f'Loading {npz_path}')
    data = load_npz(npz_path)
    logger.info(f'  N={len(data["labels"])}, '
                f'video {data["video_feats"].shape}, '
                f'audio {data["audio_feats"].shape}')

    with open(fold_path, 'rb') as f:
        folds = pickle.load(f)
    logger.info(f'  {len(folds)} folds')

    le_path = os.path.join(args.cache_dir, f'{args.dataset}_label_encoder.joblib')
    if os.path.exists(le_path):
        le = joblib.load(le_path)
        classes = [str(c) for c in le.classes_]
        labels_int = le.transform(data['labels']).astype(np.int64)
        logger.info(f'  using saved label encoder: {le_path}')
    else:
        classes = sorted(np.unique(data['labels']).tolist())
        labels_int, le = encode_labels(data['labels'], classes=classes)
        logger.info(f'  no saved encoder at {le_path}, fitting fresh')
    num_classes = len(classes)
    assert classes == sorted(classes), f'label classes are not alphabetical: {classes}'
    logger.info(f'  classes ({num_classes}): {classes}')


    if args.folds == 'all':
        folds_to_run = set(range(len(folds)))
    else:
        folds_to_run = set(int(x) for x in args.folds.split(','))
        logger.info(f'  running ONLY folds: {sorted(folds_to_run)}')

    all_fold_results = []
    fold_histories, fold_samples, fold_gates = {}, {}, {}
    for fold in folds:
        fold_idx = fold['fold']
        if fold_idx not in folds_to_run:
            continue
        train_idx = fold['train_idx']
        val_idx = fold['val_idx']
        test_idx = fold['test_idx']

        if args.subsample > 0:
            rng = np.random.default_rng(args.seed + fold_idx)
            def _sub(idx):
                if len(idx) <= args.subsample:
                    return idx
                return rng.choice(idx, size=args.subsample, replace=False)
            train_idx = _sub(train_idx)
            val_idx = _sub(val_idx)
            test_idx = _sub(test_idx)
            logger.info(f'  SUBSAMPLE active: train={len(train_idx)} '
                        f'val={len(val_idx)} test={len(test_idx)}')
        logger.info('=' * 70)
        logger.info(f'FOLD {fold_idx}: train={len(train_idx)} '
                    f'val={len(val_idx)} test={len(test_idx)}')

        seed_f = args.seed + fold_idx
        setup_seed(seed_f)
        random.seed(seed_f); np.random.seed(seed_f); torch.manual_seed(seed_f)

        loaders = make_dataloaders(data, train_idx, val_idx, test_idx,
                                   labels_int, batch_size=args.batch_size,
                                   num_workers=args.num_workers)

        model_args = build_args_for_model(args, num_classes, fold_idx)
        model = EMOERepo(model_args).to(model_args.device)
        logger.info(f'  variant: repo ({EMOERepo.__module__}) '
                    f'[run3_official: repo architecture + official hparams]')
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'  model trainable params: {n_params:,}')

        trainer = EMOETrainer(model_args)

        t0 = time.time()
        trainer.do_train(model, loaders, return_epoch_results=False)
        train_secs = time.time() - t0


        diag_dir = os.path.join(args.out_dir, 'diagnostics', args.dataset)
        os.makedirs(diag_dir, exist_ok=True)
        hist = getattr(trainer, 'history', [])
        if hist:
            pd.DataFrame(hist).to_csv(
                os.path.join(diag_dir, f'fold{fold_idx}_history.csv'),
                index=False)
        fold_histories[fold_idx] = hist

        best_path = os.path.join(model_args.checkpoint_dir, 'emoe_best.pth')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path,
                                             map_location=model_args.device))

        test_results = trainer.do_test(model, loaders['test'],
                                       mode='TEST_FINAL',
                                       return_sample_results=True)
        samples = test_results.pop('samples')
        feats = samples.pop('features')
        samples['class_true'] = np.array(classes)[samples['true']]
        samples['class_pred'] = np.array(classes)[samples['pred']]
        samples['correct'] = (samples['true'] == samples['pred']).astype(int)
        samples['dataset_index'] = np.asarray(test_idx)
        pd.DataFrame(samples).to_csv(
            os.path.join(diag_dir, f'fold{fold_idx}_test_samples.csv'),
            index=False)
        samples_plot = dict(samples); samples_plot['features'] = feats
        fold_samples[fold_idx] = samples_plot


        gates = trainer.do_test_gates(model, loaders['test'])
        logger.info(f'  gates: {gates}  (learned={test_results["Acc"]:.4f})')
        fold_gates[fold_idx] = gates




        best_rows = [h for h in hist if h.get('is_best')]
        sel = best_rows[-1] if best_rows else {}
        row = {
            'experiment': args.exp,
            'variant': args.variant,
            'fold': fold_idx,
            'best_epoch': sel.get('epoch', -1),
            'epochs_ran': hist[-1]['epoch'] if hist else 0,
            'val_acc_best': sel.get('val_acc', float('nan')),
            'val_loss_best': sel.get('val_loss', float('nan')),
            'val_f1_best': sel.get('val_f1_macro', float('nan')),
            **gates,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'train_secs': round(train_secs, 1),
            **{k: v for k, v in test_results.items() if k != 'Loss'},
            'test_loss': test_results['Loss'],
        }
        all_fold_results.append(row)

        del model, trainer, loaders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    df = pd.DataFrame(all_fold_results)
    out_csv = os.path.join(args.out_dir, f'{args.dataset}_5fold.csv')
    df.to_csv(out_csv, index=False)
    _metric_cols = [c for c in ['Acc', 'F1_macro', 'F1_weighted',
                                'W_video_mean', 'W_video_std', 'test_loss',
                                'Acc_gate_uniform', 'Acc_gate_oracle',
                                'Acc_gate_video_only', 'Acc_gate_audio_only']
                    if c in df.columns]
    _agg = pd.DataFrame({
        'metric': _metric_cols,
        'mean': [df[c].mean() for c in _metric_cols],
        'std': [df[c].std() for c in _metric_cols],
    })
    _agg_csv = os.path.join(args.out_dir, f'{args.dataset}_aggregate.csv')
    _agg.to_csv(_agg_csv, index=False)
    logger.info('=' * 70)
    logger.info(f'Aggregate (mean ± std) written to {_agg_csv}')
    logger.info(f'Per-fold results written to {out_csv}')
    logger.info(df.to_string(index=False))
    if 'Acc' in df.columns:
        logger.info(f'\nAcc      : {df["Acc"].mean():.4f} ± {df["Acc"].std():.4f}')
    if 'F1_macro' in df.columns:
        logger.info(f'F1_macro : {df["F1_macro"].mean():.4f} ± {df["F1_macro"].std():.4f}')


    if args.no_plots:
        logger.info('plots skipped (--no_plots)')
        return
    if not HAS_PLOTS:
        logger.warning(f'matplotlib unavailable — plots skipped ({_PLOT_ERR}). '
                       f'pip install matplotlib to enable.')
        return
    try:
        learned_accs = {r['fold']: r['Acc'] for r in all_fold_results}
        if fold_histories:
            P.plot_training_curves(fold_histories, args.out_dir, args.dataset)
            P.plot_routing_trajectory(fold_histories, args.out_dir, args.dataset)
        if fold_samples:
            P.plot_weight_distributions(fold_samples, args.out_dir, args.dataset)
            r_pooled = P.plot_w_vs_reliability(fold_samples, args.out_dir,
                                               args.dataset)
            P.plot_w_correct_vs_wrong(fold_samples, args.out_dir, args.dataset)
            pooled_cm = P.plot_confusion(fold_samples, classes, args.out_dir,
                                         args.dataset)
            P.plot_per_class_accuracy(pooled_cm, classes, args.out_dir,
                                      args.dataset)
            P.plot_confidence(fold_samples, args.out_dir, args.dataset)
            P.plot_tsne(fold_samples, classes, args.out_dir, args.dataset)
            P.plot_importance_vs_weights(fold_samples, args.out_dir,
                                         args.dataset)
            np.savetxt(os.path.join(args.out_dir, 'diagnostics', args.dataset,
                                    'confusion_pooled.csv'),
                       pooled_cm, fmt='%d', delimiter=',',
                       header=','.join(classes), comments='')
            logger.info(f'  pooled corr(w_video, reliability gap) = {r_pooled:+.3f}')
        if fold_gates:
            P.plot_gate_comparison(fold_gates, learned_accs, args.out_dir,
                                   args.dataset)
        P.plot_fold_accuracy(learned_accs, args.out_dir, args.dataset)
        logger.info(f'Diagnostics plots written to '
                    f'{os.path.join(args.out_dir, "plots", args.dataset)}')
    except Exception as e:
        logger.warning(f'plotting failed (results unaffected): {e}')


if __name__ == '__main__':
    main()
