import torch
import torch.nn as nn
import torch.nn.functional as F
# DMD-VA: BERT text encoder removed (no language modality); local import.
from .transformer import TransformerEncoder

class DMD(nn.Module):
    def __init__(self, args):
        super(DMD, self).__init__()
        # DMD-VA: language modality dropped. Sequence lengths come from the

        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.len_v, self.len_a = args.len_v, args.len_a
        self.orig_d_v, self.orig_d_a = args.feature_dims


        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        # DMD-VA: text_dropout removed (no language modality).
        self.attn_mask = args.attn_mask
        self.num_classes = args.num_classes
        combined_dim_low = self.d_a
        # DMD-VA: each modality is reinforced by the single other modality, so

        combined_dim_high = self.d_a
        # DMD-VA: fusion = [last_h_v (d), last_h_a (d), c_fusion (2d)] = 4d

        combined_dim = (self.d_v + self.d_a) + self.d_l * 2
        # DMD-VA: K-way classification logits (was 1 sentiment scalar).
        output_dim = args.num_classes


        # DMD-VA: language projection removed.
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)


        self.encoder_s_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.encoder_s_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)


        self.encoder_c = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)


        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)


        self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)


        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        # DMD-VA: homogeneous fusion over 2 invariant streams -> 2d (was 3d).
        self.proj1_c = nn.Linear(self.d_l * 2, self.d_l * 2)
        self.proj2_c = nn.Linear(self.d_l * 2, self.d_l * 2)
        self.out_layer_c = nn.Linear(self.d_l * 2, output_dim)


        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_a = self.get_network(self_type='va')
        # DMD-VA: single-source memory self-attention (embed dim d, was 2d).
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)


        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)



        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)


        self.weight_v = nn.Linear(self.d_v, self.d_v)
        self.weight_a = nn.Linear(self.d_a, self.d_a)
        self.weight_c = nn.Linear(2 * self.d_l, 2 * self.d_l)

        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='v', layers=-1):
        # DMD-VA: language network types removed; memory transformers are

        if self_type in ['a', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video, is_distill=False):
        # DMD-VA: `text` is an unused placeholder kept so the trainer call site

        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)

        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)
        c_list = [c_v, c_a]

        c_v_sim = self.align_c_v(c_v.contiguous().view(x_v.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_v.size(0), -1))

        recon_v = self.decoder_v(torch.cat([s_v, c_list[0]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[1]], dim=1))

        s_v_r = self.encoder_s_v(recon_v)
        s_a_r = self.encoder_s_a(recon_a)

        s_v = s_v.permute(2, 0, 1)
        s_a = s_a.permute(2, 0, 1)

        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        repr_v_low = self.proj1_v_low(hs_v_low)
        hs_proj_v_low = self.proj2_v_low(
            F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_low += hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)

        hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        repr_a_low = self.proj1_a_low(hs_a_low)
        hs_proj_a_low = self.proj2_a_low(
            F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_low += hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)

        proj_s_v = self.proj_cosine_v(s_v.transpose(0, 1).contiguous().view(x_v.size(0), -1))
        proj_s_a = self.proj_cosine_a(s_a.transpose(0, 1).contiguous().view(x_v.size(0), -1))

        c_v_att = self.self_attentions_c_v(c_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]
        c_a_att = self.self_attentions_c_a(c_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]
        c_fusion = torch.cat([c_v_att, c_a_att], dim=1)

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)



        h_v_with_as = self.trans_v_with_a(s_v, s_a, s_a)
        h_vs = self.trans_v_mem(h_v_with_as)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]


        h_a_with_vs = self.trans_a_with_v(s_a, s_v, s_v)
        h_as = self.trans_a_mem(h_a_with_vs)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]

        hs_proj_v_high = self.proj2_v_high(
            F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_high += last_h_v
        logits_v_high = self.out_layer_v_high(hs_proj_v_high)

        hs_proj_a_high = self.proj2_a_high(
            F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout,
                      training=self.training))
        hs_proj_a_high += last_h_a
        logits_a_high = self.out_layer_a_high(hs_proj_a_high)

        last_h_v = torch.sigmoid(self.weight_v(last_h_v))
        last_h_a = torch.sigmoid(self.weight_a(last_h_a))
        c_fusion = torch.sigmoid(self.weight_c(c_fusion))

        last_hs = torch.cat([last_h_v, last_h_a, c_fusion], dim=1)
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)

        res = {
            'logits_v_homo': logits_v_low,
            'logits_a_homo': logits_a_low,
            'repr_v_homo': repr_v_low,
            'repr_a_homo': repr_a_low,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_v': s_v,
            's_a': s_a,
            'proj_s_v': proj_s_v,
            'proj_s_a': proj_s_a,
            'c_v': c_v,
            'c_a': c_a,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_v': recon_v,
            'recon_a': recon_a,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            'logits_v_hetero': logits_v_high,
            'logits_a_hetero': logits_a_high,
            'repr_v_hetero': hs_proj_v_high,
            'repr_a_hetero': hs_proj_a_high,
            'last_h_v': h_vs[-1],
            'last_h_a': h_as[-1],
            'logits_c': logits_c,
            'output_logit': output
        }
        return res