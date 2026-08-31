import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils.metrics import classification_metrics


# trains one model and runs the tests for a fold
class Trainer:
    def __init__(self, args):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.key_eval = getattr(args, 'KeyEval', 'Loss')

    def _to_device(self, batch):
        dev = self.args.device
        return (batch['vision'].to(dev), batch['audio'].to(dev),
                batch['audio_len'].to(dev), batch['label'].to(dev).long())

    # train loop with early stopping on the validation set
    def do_train(self, model, loaders, diag_dir=None, fold=0, class_names=None):
        model.to(self.args.device)
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate,
                               weight_decay=getattr(self.args, 'weight_decay', 0.0))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                      patience=self.args.patience)

        min_or_max = 'min' if self.key_eval == 'Loss' else 'max'
        best_valid = 1e8 if min_or_max == 'min' else -1.0
        best_epoch, history = 0, []

        cuda = str(self.args.device).startswith('cuda')
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        best_wall = t0

        for epoch in range(1, self.args.max_epochs + 1):
            model.train()
            train_loss, y_pred, y_true = 0.0, [], []
            for batch in loaders['train']:
                vision, audio, alen, labels = self._to_device(batch)
                optimizer.zero_grad()
                logits = model(vision, audio, alen)
                loss = self.criterion(logits, labels)
                loss.backward()
                if self.args.grad_clip != -1.0:
                    nn.utils.clip_grad_value_(model.parameters(), self.args.grad_clip)
                optimizer.step()
                train_loss += loss.item()
                y_pred.append(logits.detach().cpu())
                y_true.append(labels.cpu())

            train_loss /= max(len(loaders['train']), 1)
            tr = classification_metrics(torch.cat(y_pred), torch.cat(y_true))
            val = self.do_test(model, loaders['valid'])
            scheduler.step(val['Loss'])

            history.append({
                'epoch': epoch,
                'train_loss': round(train_loss, 5), 'train_acc': tr['Acc'],
                'val_acc': val['Acc'], 'val_f1_macro': val['F1_macro'],
                'val_f1_weighted': val['F1_weighted'], 'val_loss': val['Loss'],
                'lr': optimizer.param_groups[0]['lr'], 'is_best': 0,
            })

            cur = val[self.key_eval]
            better = cur <= best_valid - 1e-6 if min_or_max == 'min' \
                else cur >= best_valid + 1e-6
            if better:
                best_valid, best_epoch = cur, epoch
                best_wall = time.perf_counter()
                if diag_dir is not None:
                    torch.save(model.state_dict(),
                               os.path.join(diag_dir, f'fold{fold}_best.pth'))

            if epoch - best_epoch >= self.args.early_stop:
                break

        if cuda:
            torch.cuda.synchronize()

        self.train_time_s = round(time.perf_counter() - t0, 4)
        self.epochs_run = len(history)
        self.time_to_best_s = round(best_wall - t0, 4)
        self.peak_train_mem_mb = (round(torch.cuda.max_memory_allocated() / 1e6, 2)
                                  if cuda else None)

        for row in history:
            if row['epoch'] == best_epoch:
                row['is_best'] = 1
        if diag_dir is not None and history:
            with open(os.path.join(diag_dir, f'fold{fold}_history.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                w.writeheader()
                w.writerows(history)
        return history

    # evaluate the model on a split
    def do_test(self, model, loader, collect_samples=False, class_names=None):
        model.eval()
        eval_loss = 0.0
        logits_all, true_all, idx_all = [], [], []
        with torch.no_grad():
            for batch in loader:
                vision, audio, alen, labels = self._to_device(batch)
                logits = model(vision, audio, alen)
                eval_loss += self.criterion(logits, labels).item()
                logits_all.append(logits.cpu())
                true_all.append(labels.cpu())
                idx_all.append(batch['orig_index'])

        eval_loss /= max(len(loader), 1)
        logits = torch.cat(logits_all)
        true = torch.cat(true_all)
        probs = F.softmax(logits, dim=1).numpy()
        res = classification_metrics(logits, true)
        res['Loss'] = round(eval_loss, 4)
        if collect_samples:
            idx = torch.cat(idx_all).numpy()
            res['y_true'] = true.numpy()
            res['y_pred'] = logits.argmax(1).numpy()
            res['probs'] = probs
            res['orig_index'] = idx
            res['sample_rows'] = self._sample_rows(idx, true.numpy(), logits,
                                                   probs, class_names)
        return res

    # test with one modality zeroed to see how much it matters
    def do_test_zeroed(self, model, loader, zero):
        assert zero in ('audio', 'vision')
        model.eval()
        logits_all, true_all = [], []
        with torch.no_grad():
            for batch in loader:
                vision, audio, alen, labels = self._to_device(batch)
                if zero == 'audio':
                    audio = torch.zeros_like(audio)
                else:
                    vision = torch.zeros_like(vision)
                logits_all.append(model(vision, audio, alen).cpu())
                true_all.append(labels.cpu())
        return classification_metrics(torch.cat(logits_all), torch.cat(true_all))

    # measure forward pass speed
    def benchmark(self, model, sample_batch, n_warmup=10, n_iter=50, bs_throughput=64):
        dev = self.args.device
        cuda = str(dev).startswith('cuda')
        model.eval()

        def fwd(v, a, l):
            with torch.no_grad():
                model(v, a, l)

        v1 = sample_batch['vision'][:1].to(dev)
        a1 = sample_batch['audio'][:1].to(dev)
        l1 = sample_batch['audio_len'][:1].to(dev)
        for _ in range(n_warmup):
            fwd(v1, a1, l1)
        if cuda:
            torch.cuda.synchronize()
        lat = []
        for _ in range(n_iter):
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            fwd(v1, a1, l1)
            if cuda:
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t) * 1000.0)
        lat = np.array(lat)

        B = bs_throughput
        vb = sample_batch['vision'][:1].repeat(B, 1, 1).to(dev)
        ab = sample_batch['audio'][:1].repeat(B, 1, 1).to(dev)
        lb = sample_batch['audio_len'][:1].repeat(B).to(dev)
        for _ in range(max(3, n_warmup // 2)):
            fwd(vb, ab, lb)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(n_iter):
            fwd(vb, ab, lb)
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        throughput = (B * n_iter) / elapsed if elapsed > 0 else float('nan')
        peak_mb = round(torch.cuda.max_memory_allocated() / 1e6, 2) if cuda else None

        return {
            'latency_bs1_ms_median': round(float(np.median(lat)), 4),
            'latency_bs1_ms_p95': round(float(np.percentile(lat, 95)), 4),
            f'throughput_bs{B}_sps': round(float(throughput), 2),
            'peak_infer_mem_mb': peak_mb,
        }

    @staticmethod
    def _sample_rows(idx, y_true, logits, probs, class_names):
        ce = F.cross_entropy(logits, torch.as_tensor(y_true).long(),
                             reduction='none').numpy()
        pred = probs.argmax(1)
        conf = probs.max(1)
        K = probs.shape[1]
        rows = []
        for b in range(len(idx)):
            t, p = int(y_true[b]), int(pred[b])
            row = {
                'orig_index': int(idx[b]), 'true_idx': t, 'pred_idx': p,
                'true_name': class_names[t] if class_names else t,
                'pred_name': class_names[p] if class_names else p,
                'correct': int(t == p), 'confidence': round(float(conf[b]), 5),
                'ce': round(float(ce[b]), 5),
            }
            for k in range(K):
                row[f'p{k}'] = round(float(probs[b, k]), 5)
            rows.append(row)
        return rows

    @staticmethod
    def _write_samples(rows, diag_dir, fold, split):
        if not rows:
            return
        path = os.path.join(diag_dir, f'fold{fold}_{split}_samples.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    @staticmethod
    def write_test_samples(rows, diag_dir, fold):
        Trainer._write_samples(rows, diag_dir, fold, 'test')

    @staticmethod
    def write_val_samples(rows, diag_dir, fold):
        Trainer._write_samples(rows, diag_dir, fold, 'val')

    @staticmethod
    # save the confusion matrix
    def write_confusion(y_true, y_pred, n_classes, diag_dir, fold):
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        path = os.path.join(diag_dir, f'fold{fold}_confusion.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['true\\pred'] + list(range(n_classes)))
            for i in range(n_classes):
                w.writerow([i] + cm[i].tolist())
