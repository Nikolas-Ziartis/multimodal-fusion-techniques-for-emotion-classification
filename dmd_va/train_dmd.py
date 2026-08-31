import os
import csv
import argparse
import numpy as np
import torch



# keep full fp32 so results match across runs
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

from .config import build_args, NUM_CLASSES
from .dataset import load_npz, load_folds, encode_labels, make_dataloaders, load_label_classes
from .model.dmd_repo import DMD as DMDModel
from .dmd_trainer import DMD as DMDTrainer
from .distillnets.get_distillation_kernel import DistillationKernel as HeteroKernel
from .distillnets.get_distillation_kernel_homo import DistillationKernel as HomoKernel
from .utils.metrics import setup_seed, count_parameters


# turn the graph prior into weights that sum to one
def _softmax(x, t=0.25):
    x = np.asarray(x, dtype=np.float64) / t
    e = np.exp(x - x.max())
    return (e / e.sum()).tolist()


def build_model(args, device):
    d = args.dst_feature_dim_nheads[0]
    K = args.num_classes
    model = DMDModel(args).to(device)
    homo = HomoKernel(K, d, args.gd_size_homo, [0, 1], [0, 1],
                      _softmax(args.gd_prior_homo), args.gd_reg, args.w_losses,
                      args.gd_metric, args.gd_alpha, args).to(device)
    hetero = HeteroKernel(K, d, args.gd_size_hetero, [0, 1], [0, 1],
                          _softmax(args.gd_prior_hetero), args.gd_reg, args.w_losses,
                          args.gd_metric, args.gd_alpha, args).to(device)
    return [model, homo, hetero]


def run(dataset, cache_dir, seed=1111, device=None, out_root='./experiments/dmd_va',
        num_workers=2, overrides=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
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

    diag_dir = os.path.join(out_root, dataset, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)

    per_fold = []
    for k, fold in enumerate(folds):
        setup_seed(seed)
        loaders = make_dataloaders(
            data, fold['train_idx'], fold['val_idx'], fold['test_idx'],
            labels_int, batch_size=args.batch_size, num_workers=num_workers)

        model = build_model(args, device)
        if k == 0:
            print(f"[{dataset}] DMD-VA params: {count_parameters(model[0]):,} "
                  f"(+homo {count_parameters(model[1]):,}, +hetero {count_parameters(model[2]):,})")

        trainer = DMDTrainer(args)
        trainer.do_train(model, loaders, diag_dir=diag_dir, fold=k, class_names=class_names)


        model[0].load_state_dict(torch.load(
            os.path.join(diag_dir, f'fold{k}_best.pth'), map_location=device))
        test = trainer.do_test(model[0], loaders['test'], mode='TEST',
                               collect_samples=True, class_names=class_names)
        trainer.write_test_samples(test['sample_rows'], diag_dir, k)
        trainer.write_confusion(test['y_true'], test['y_pred'], args.num_classes, diag_dir, k)

        row = {'fold': k, 'Acc': test['Acc'], 'F1_macro': test['F1_macro'],
               'F1_weighted': test['F1_weighted'], 'test_loss': test['Loss']}
        row.update({f'acc_{h}': test['head_acc'][h] for h in test['head_acc']})
        per_fold.append(row)
        print(f"  fold {k}: Acc={test['Acc']:.4f}  F1_macro={test['F1_macro']:.4f}  "
              f"F1_w={test['F1_weighted']:.4f}")


    with open(os.path.join(out_root, dataset, 'per_fold_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_fold[0].keys()))
        w.writeheader()
        w.writerows(per_fold)

    metrics = [m for m in per_fold[0].keys() if m != 'fold']
    with open(os.path.join(out_root, dataset, 'aggregate_results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'mean', 'std'])
        for m in metrics:
            vals = [r[m] for r in per_fold]
            w.writerow([m, round(float(np.mean(vals)), 5), round(float(np.std(vals)), 5)])

    accs = [r['Acc'] for r in per_fold]
    print(f"[{dataset}] 5-fold Acc = {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    return per_fold


def main():
    p = argparse.ArgumentParser(description="Train DMD-VA (2-modality DMD) on RAVDESS / CREMA-D")
    p.add_argument('--dataset', required=True,
                   choices=['ravdess', 'cremad', 'mead8', 'mead6'])
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--seed', type=int, default=1111)
    p.add_argument('--out_root', default='./experiments/dmd_va')
    p.add_argument('--num_workers', type=int, default=2)

    p.add_argument('--learning_rate', type=float, default=None)
    p.add_argument('--update_epochs', type=int, default=None)
    p.add_argument('--output_dropout', type=float, default=None)
    p.add_argument('--gd_metric', type=str, default=None, choices=['l1', 'l2', 'cosine', 'kl'])
    p.add_argument('--gd_reg', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--early_stop', type=int, default=None)
    args = p.parse_args()

    overrides = {k: v for k, v in dict(
        learning_rate=args.learning_rate, update_epochs=args.update_epochs,
        output_dropout=args.output_dropout, gd_metric=args.gd_metric,
        gd_reg=args.gd_reg, weight_decay=args.weight_decay, batch_size=args.batch_size,
        early_stop=args.early_stop).items() if v is not None}

    run(args.dataset, args.cache_dir, seed=args.seed, out_root=args.out_root,
        num_workers=args.num_workers, overrides=overrides)


if __name__ == '__main__':
    main()
