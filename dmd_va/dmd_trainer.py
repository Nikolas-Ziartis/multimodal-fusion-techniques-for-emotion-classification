import os
import csv
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# DMD-VA: local imports; classification metrics replace MMSA's MetricsTop.
from .losses.HingeLoss import HingeLoss
from .utils.metrics import classification_metrics

logger = logging.getLogger('MMSA')


def dict_to_str(d):  # DMD-VA: small local helper (was imported from ..utils)
    return ' '.join(f"{k}:{round(float(v), 4)}" for k, v in d.items() if not isinstance(v, dict))


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class DMD():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()   # DMD-VA: classification (was nn.L1Loss)
        self.cosine = nn.CosineEmbeddingLoss()
        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        # DMD-VA: 'Loss' (min) or 'Acc' (max) for checkpoint selection.
        self.key_eval = getattr(args, 'KeyEval', 'Loss')

    def _ort_term(self, s, c):
        # DMD-VA: soft orthogonality between modality-specific (s) and shared (c)

        s2 = s.reshape(-1, s.size(-1))
        c2 = c.reshape(-1, c.size(-1))
        tgt = torch.full((s2.size(0),), -1.0, device=s2.device)
        return self.cosine(s2, c2, tgt)

    def do_train(self, model, dataloader, return_epoch_results=False,
                 diag_dir=None, fold=0, class_names=None):


        params = list(model[0].parameters()) + \
                 list(model[1].parameters()) + \
                 list(model[2].parameters())

        # DMD-VA: weight_decay exposed (defaults to 0.0, matching DMD's released

        optimizer = optim.Adam(params, lr=self.args.learning_rate,
                               weight_decay=getattr(self.args, 'weight_decay', 0.0))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.key_eval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        history = []   # DMD-VA: per-epoch diagnostics rows

        net = []
        net_dmd = model[0]
        net_distill_homo = model[1]
        net_distill_hetero = model[2]
        net.append(net_dmd)
        net.append(net_distill_homo)
        net.append(net_distill_hetero)
        model = net

        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            train_loss = 0.0
            left_epochs = self.args.update_epochs
            edge_homo_sum, edge_hetero_sum, n_batches = None, None, 0
            # DMD-VA: plain iteration (was tqdm).
            if True:
                for batch_data in dataloader['train']:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    # DMD-VA: integer class labels (was a regression target view(-1,1)).
                    labels = batch_data['labels']['M'].to(self.args.device).long()

                    logits_homo, reprs_homo, logits_hetero, reprs_hetero = [], [], [], []

                    output = model[0](text, audio, vision, is_distill=True)


                    logits_homo.append(output['logits_v_homo'])
                    logits_homo.append(output['logits_a_homo'])
                    reprs_homo.append(output['repr_v_homo'])
                    reprs_homo.append(output['repr_a_homo'])


                    logits_hetero.append(output['logits_v_hetero'])
                    logits_hetero.append(output['logits_a_hetero'])
                    reprs_hetero.append(output['repr_v_hetero'])
                    reprs_hetero.append(output['repr_a_hetero'])

                    logits_homo = torch.stack(logits_homo)
                    reprs_homo = torch.stack(reprs_homo)
                    logits_hetero = torch.stack(logits_hetero)
                    reprs_hetero = torch.stack(reprs_hetero)


                    edges_homo, edges_origin_homo = model[1](logits_homo, reprs_homo)
                    edges_hetero, edges_origin_hetero = model[2](logits_hetero, reprs_hetero)


                    loss_task_all = self.criterion(output['output_logit'], labels)
                    loss_task_v_homo = self.criterion(output['logits_v_homo'], labels)
                    loss_task_a_homo = self.criterion(output['logits_a_homo'], labels)
                    loss_task_v_hetero = self.criterion(output['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(output['logits_a_hetero'], labels)
                    loss_task_c = self.criterion(output['logits_c'], labels)
                    loss_task = loss_task_all + loss_task_v_homo + loss_task_a_homo \
                        + loss_task_v_hetero + loss_task_a_hetero + loss_task_c


                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    loss_recon = loss_recon_v + loss_recon_a


                    loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                    loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                    loss_s_sr = loss_sv_slv + loss_sa_sla






                    cosine_similarity_s_c_v = self._ort_term(output['s_v'], output['c_v'])
                    cosine_similarity_s_c_a = self._ort_term(output['s_a'], output['c_a'])
                    loss_ort = cosine_similarity_s_c_v + cosine_similarity_s_c_a


                    c_v, c_a = output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)


                    loss_reg_homo, loss_logit_homo, loss_repr_homo = \
                        model[1].distillation_loss(logits_homo, reprs_homo, edges_homo)
                    graph_distill_loss_homo = 0.05 * (loss_logit_homo + loss_reg_homo)


                    loss_reg_hetero, loss_logit_hetero, loss_repr_hetero = \
                        model[2].distillation_loss(logits_hetero, reprs_hetero, edges_hetero)
                    graph_distill_loss_hetero = 0.05 * (loss_logit_hetero + loss_repr_hetero + loss_reg_hetero)

                    combined_loss = loss_task + \
                                    graph_distill_loss_homo + graph_distill_loss_hetero + \
                                    (loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1) * 0.1

                    combined_loss.backward()


                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters()) + \
                                 list(model[1].parameters()) + \
                                 list(model[2].parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()

                    y_pred.append(output['output_logit'].detach().cpu())
                    y_true.append(labels.cpu())
                    # DMD-VA: accumulate mean GD edge weights for diagnostics.
                    eh = edges_homo.mean(1).detach().cpu().numpy()
                    et = edges_hetero.mean(1).detach().cpu().numpy()
                    edge_homo_sum = eh if edge_homo_sum is None else edge_homo_sum + eh
                    edge_hetero_sum = et if edge_hetero_sum is None else edge_hetero_sum + et
                    n_batches += 1
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:

                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            # DMD-VA: classification metrics (was MMSA regression self.metrics).
            train_results = classification_metrics(pred, true)
            logger.info(f">> Epoch {epochs} TRAIN total_loss: {round(train_loss, 4)} "
                        f"{dict_to_str(train_results)}")


            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.key_eval]
            scheduler.step(val_results['Loss'])

            # DMD-VA: mean GD edge weights this epoch (edges[0]=A->V, edges[1]=V->A).
            eh = (edge_homo_sum / max(n_batches, 1)).tolist()
            et = (edge_hetero_sum / max(n_batches, 1)).tolist()
            history.append({
                'epoch': epochs,
                'train_loss': round(train_loss, 5), 'train_acc': train_results['Acc'],
                'val_acc': val_results['Acc'], 'val_f1_macro': val_results['F1_macro'],
                'val_f1_weighted': val_results['F1_weighted'], 'val_loss': val_results['Loss'],
                'lr': optimizer.param_groups[0]['lr'],
                'edge_homo_AtoV': round(eh[0], 5), 'edge_homo_VtoA': round(eh[1], 5),
                'edge_hetero_AtoV': round(et[0], 5), 'edge_hetero_VtoA': round(et[1], 5),
                'is_best': 0,
            })

            # DMD-VA: fold-aware best checkpoint (was save-every-epoch + ./pt/dmd.pth).
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' \
                else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                if diag_dir is not None:
                    torch.save(model[0].state_dict(),
                               os.path.join(diag_dir, f'fold{fold}_best.pth'))


            if epochs - best_epoch >= self.args.early_stop:
                break

        # DMD-VA: mark the selected epoch and persist the per-epoch history.
        for row in history:
            if row['epoch'] == best_epoch:
                row['is_best'] = 1
        if diag_dir is not None:
            with open(os.path.join(diag_dir, f'fold{fold}_history.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                w.writeheader()
                w.writerows(history)
        return history

    # DMD-VA: the 6 classification heads (fusion + invariant + 2 homo + 2 hetero).
    HEADS = ['output_logit', 'logits_c', 'logits_v_homo', 'logits_a_homo',
             'logits_v_hetero', 'logits_a_hetero']

    def do_test(self, model, dataloader, mode="VAL", collect_samples=False, class_names=None):
        # DMD-VA: rewritten for categorical evaluation. Reports Acc / F1 on the



        model.eval()
        eval_loss = 0.0
        y_pred, y_true = [], []
        head_correct = {h: 0 for h in self.HEADS}
        n_total = 0
        sample_rows = []

        with torch.no_grad():
            for batch_data in dataloader:
                vision = batch_data['vision'].to(self.args.device)
                audio = batch_data['audio'].to(self.args.device)
                text = batch_data['text'].to(self.args.device)
                labels = batch_data['labels']['M'].to(self.args.device).long()
                output = model(text, audio, vision, is_distill=True)

                eval_loss += self.criterion(output['output_logit'], labels).item()
                logits = output['output_logit']
                y_pred.append(logits.cpu())
                y_true.append(labels.cpu())


                for h in self.HEADS:
                    head_correct[h] += (output[h].argmax(1) == labels).sum().item()
                n_total += labels.size(0)

                if collect_samples:
                    idx = batch_data['orig_index']
                    probs = F.softmax(logits, dim=1)
                    pred_idx = logits.argmax(1)
                    conf = probs.max(1).values

                    ce = {h: F.cross_entropy(output[h], labels, reduction='none').cpu()
                          for h in self.HEADS}
                    for b in range(labels.size(0)):
                        t, p = int(labels[b]), int(pred_idx[b])
                        sample_rows.append({
                            'orig_index': int(idx[b]),
                            'true_idx': t, 'pred_idx': p,
                            'true_name': class_names[t] if class_names else t,
                            'pred_name': class_names[p] if class_names else p,
                            'correct': int(t == p), 'confidence': round(float(conf[b]), 5),
                            'ce_fusion': round(float(ce['output_logit'][b]), 5),
                            'ce_c': round(float(ce['logits_c'][b]), 5),
                            'ce_v_homo': round(float(ce['logits_v_homo'][b]), 5),
                            'ce_a_homo': round(float(ce['logits_a_homo'][b]), 5),
                            'ce_v_hetero': round(float(ce['logits_v_hetero'][b]), 5),
                            'ce_a_hetero': round(float(ce['logits_a_hetero'][b]), 5),
                        })

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = classification_metrics(pred, true)
        eval_results['Loss'] = round(eval_loss, 4)
        eval_results['head_acc'] = {h: round(head_correct[h] / max(n_total, 1), 5)
                                    for h in self.HEADS}
        logger.info(f"{mode} >> {dict_to_str(eval_results)}")
        if collect_samples:
            eval_results['sample_rows'] = sample_rows
            eval_results['y_true'] = true.numpy()
            eval_results['y_pred'] = pred.argmax(1).numpy()
        return eval_results

    @staticmethod
    def write_test_samples(rows, diag_dir, fold):
        path = os.path.join(diag_dir, f'fold{fold}_test_samples.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    @staticmethod
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