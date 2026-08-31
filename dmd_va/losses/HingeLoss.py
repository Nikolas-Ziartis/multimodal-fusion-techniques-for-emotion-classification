import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def compute_cosine(self, x, y):


        x_norm = torch.sqrt(torch.sum(torch.pow(x, 2), 1)+1e-8)
        x_norm = torch.max(x_norm, 1e-8*torch.ones_like(x_norm))
        y_norm = torch.sqrt(torch.sum(torch.pow(y, 2), 1)+1e-8)
        y_norm = torch.max(y_norm, 1e-8*torch.ones_like(y_norm))
        cosine = torch.sum(x * y, 1) / (x_norm * y_norm)
        return cosine

    def forward(self, ids, feats, margin=0.1):
        B, F = feats.shape

        s = feats.repeat(1, B).view(-1, F)
        s_ids = ids.view(B, 1).repeat(1, B)
        
        t = feats.repeat(B, 1)
        t_ids = ids.view(1, B).repeat(B, 1)

        cosine = self.compute_cosine(s, t)
        equal_mask = torch.eye(B, dtype=torch.bool, device=feats.device) # DMD-VA: device-safe (was CPU)
        s_ids = s_ids[~equal_mask].view(B, B-1)
        t_ids = t_ids[~equal_mask].view(B, B-1)
        cosine = cosine.view(B, B)[~equal_mask].view(B, B-1)

        sim_mask = (s_ids == t_ids)
        # DMD-VA: constant margin for any different-class pair. Emotion classes


        margin = 0.15 * (s_ids != t_ids).float()

        loss = 0
        loss_num = 0
        
        for i in range(B):
            sim_num = sum(sim_mask[i])
            dif_num = B - 1 - sim_num
            if not sim_num or not dif_num:
                continue
            sim_cos = cosine[i, sim_mask[i]].reshape(-1, 1).repeat(1, dif_num)
            dif_cos = cosine[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)
            t_margin = margin[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)

            loss_i = torch.max(torch.zeros_like(sim_cos), t_margin - sim_cos + dif_cos).mean()
            loss += loss_i
            loss_num += 1

        if loss_num == 0:
            loss_num = 1

        loss = loss / loss_num
        return loss