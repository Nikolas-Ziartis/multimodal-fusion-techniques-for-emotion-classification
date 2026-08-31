import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm



from utils.functions import dict_to_str, uni_distill, entropy_balance
from utils.metrics import MetricsTop

logger = logging.getLogger('EMOE')


# runs training and testing for the emoe model
class EMOE():
    def __init__(self, args):
        self.args = args




        weight = getattr(args, 'class_weights', None)
        if weight is not None:
            weight = weight.to(args.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=getattr(args, 'label_smoothing', 0.0),
        )
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    # train loop with early stopping on the validation set
    def do_train(self, model, dataloader, return_epoch_results=False):
        params = list(model.parameters())
        optimizer = optim.Adam(params, lr=self.args.learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                      patience=self.args.patience)

        epochs, best_epoch = 0, 0


        self.history = []
        if return_epoch_results:
            epoch_results = {'train': [], 'valid': [], 'test': []}
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0


        ckpt_dir = getattr(self.args, 'checkpoint_dir', './pt')
        os.makedirs(ckpt_dir, exist_ok=True)


        max_epochs = getattr(self.args, 'max_epochs', 50)





        eval_test_in_train = getattr(self.args, 'eval_test_in_train', False)

        while True:
            epochs += 1
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0

            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)

                    labels = batch_data['labels']['M'].to(self.args.device).long()

                    lens = batch_data.get('audio_lens', None)
                    if lens is not None:
                        lens = lens.to(self.args.device)

                    output = model(audio, vision, audio_lens=lens)
                    w = output['channel_weight']

                    y_pred.append(output['logits_c'].detach().cpu())
                    y_true.append(labels.detach().cpu())


                    loss_task_v = self.criterion(output['logits_v'], labels)
                    loss_task_a = self.criterion(output['logits_a'], labels)
                    loss_task_m = self.criterion(output['logits_c'], labels)











                    a_dist = F.cross_entropy(output['logits_a'], labels,
                                             reduction='none')
                    v_dist = F.cross_entropy(output['logits_v'], labels,
                                             reduction='none')
                    c_diff = torch.stack([v_dist, a_dist], dim=1)




                    inv = 1.0 / (c_diff + 0.1)
                    dist = inv / inv.sum(dim=1, keepdim=True)
                    loss_sim = torch.mean((dist.detach() - w) ** 2)
                    loss_ety = entropy_balance(w)



                    if self.args.fusion_method == "sum":
                        loss_ud = uni_distill(
                            output['c_proj'],
                            (output['v_proj'] * w[:, 0].view(-1, 1)
                             + output['a_proj'] * w[:, 1].view(-1, 1)).detach(),
                        )
                    elif self.args.fusion_method == "concat":
                        loss_ud = uni_distill(
                            output['c_proj'],
                            torch.cat([
                                output['v_proj'] * w[:, 0].view(-1, 1),
                                output['a_proj'] * w[:, 1].view(-1, 1),
                            ], dim=1).detach(),
                        )




                    loss = loss_task_m + (loss_task_v + loss_task_a) / 2 \
                           + 0.1 * (loss_ety + 0.1 * loss_sim) + 0.1 * loss_ud

                    loss.backward()
                    train_loss += loss.item()

                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")

            self.history.append({
                'epoch': epochs,
                'train_loss': round(train_loss, 6),
                'train_acc': float(train_results.get('Acc', float('nan'))),
                'val_acc': float(val_results.get('Acc', float('nan'))),
                'val_f1_macro': float(val_results.get('F1_macro', float('nan'))),
                'val_loss': float(val_results.get('Loss', float('nan'))),
                'val_w_video_mean': float(val_results.get('W_video_mean', float('nan'))),
                'val_w_video_std': float(val_results.get('W_video_std', float('nan'))),
                'lr': optimizer.param_groups[0]['lr'],
                'is_best': 0,
            })
            if eval_test_in_train:
                _ = self.do_test(model, dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])

            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' \
                else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                model_save_path = os.path.join(ckpt_dir, 'emoe_best.pth')
                torch.save(model.state_dict(), model_save_path)
                if getattr(self, 'history', None):
                    self.history[-1]['is_best'] = 1

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                epoch_results['test'].append(
                    self.do_test(model, dataloader['test'], mode="TEST"))



            if epochs - best_epoch >= self.args.early_stop or epochs >= max_epochs:
                return epoch_results if return_epoch_results else None

    # evaluate the model on a split
    def do_test(self, model, dataloader, mode="VAL",
                return_sample_results=False, f=0):
        model.eval()
        y_pred, y_true = [], []



        w_all = []


        s_ce_c, s_ce_v, s_ce_a, s_conf, s_feat = [], [], [], [], []

        eval_loss = 0.0

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)

                    labels = batch_data['labels']['M'].to(self.args.device).long()
                    lens = batch_data.get('audio_lens', None)
                    if lens is not None:
                        lens = lens.to(self.args.device)
                    output = model(audio, vision, audio_lens=lens)

                    loss = self.criterion(output['logits_c'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['logits_c'].cpu())
                    y_true.append(labels.cpu())
                    w_all.append(output['channel_weight'].cpu())
                    if return_sample_results:
                        s_ce_c.append(F.cross_entropy(output['logits_c'],
                                      labels, reduction='none').cpu())
                        s_ce_v.append(F.cross_entropy(output['logits_v'],
                                      labels, reduction='none').cpu())
                        s_ce_a.append(F.cross_entropy(output['logits_a'],
                                      labels, reduction='none').cpu())
                        s_conf.append(torch.softmax(output['logits_c'], 1)
                                      .max(1).values.cpu())
                        s_feat.append(output['c_proj'].cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        w_all = torch.cat(w_all)
        eval_results["W_video_mean"] = float(w_all[:, 0].mean())
        eval_results["W_video_std"] = float(w_all[:, 0].std())
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results['samples'] = {
                'true': true.numpy(),
                'pred': pred.argmax(dim=1).numpy(),
                'w_video': w_all[:, 0].numpy(),
                'ce_fused': torch.cat(s_ce_c).numpy(),
                'ce_video': torch.cat(s_ce_v).numpy(),
                'ce_audio': torch.cat(s_ce_a).numpy(),
                'confidence': torch.cat(s_conf).numpy(),
                'features': torch.cat(s_feat).numpy(),
            }
        return eval_results






    # compare the learned router against fixed and oracle gates
    def do_test_gates(self, model, dataloader):
        model.eval()
        accs = {m: [0, 0] for m in
                ['uniform', 'video_only', 'audio_only', 'oracle']}
        with torch.no_grad():
            for batch_data in dataloader:
                vision = batch_data['vision'].to(self.args.device)
                audio = batch_data['audio'].to(self.args.device)
                labels = batch_data['labels']['M'].to(self.args.device).long()
                lens = batch_data.get('audio_lens', None)
                if lens is not None:
                    lens = lens.to(self.args.device)
                B = labels.shape[0]

                out = model(audio, vision, audio_lens=lens)
                ce_v = F.cross_entropy(out['logits_v'], labels, reduction='none')
                ce_a = F.cross_entropy(out['logits_a'], labels, reduction='none')
                oracle_w = torch.zeros(B, 2, device=labels.device)
                pick_v = (ce_v <= ce_a)
                oracle_w[pick_v, 0] = 1.0
                oracle_w[~pick_v, 1] = 1.0
                forced = {
                    'uniform': torch.full((B, 2), 0.5, device=labels.device),
                    'video_only': torch.tensor([[1.0, 0.0]],
                                  device=labels.device).repeat(B, 1),
                    'audio_only': torch.tensor([[0.0, 1.0]],
                                  device=labels.device).repeat(B, 1),
                    'oracle': oracle_w,
                }
                for mode_name, fw in forced.items():
                    o = model(audio, vision, audio_lens=lens, force_w=fw)
                    correct = (o['logits_c'].argmax(1) == labels).sum().item()
                    accs[mode_name][0] += correct
                    accs[mode_name][1] += B
        return {f'Acc_gate_{k}': round(v[0] / max(v[1], 1), 4)
                for k, v in accs.items()}
