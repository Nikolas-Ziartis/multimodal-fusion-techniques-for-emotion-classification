import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers_encoder.transformer import TransformerEncoder
from model.router_repo import router


# the emoe model with the modality router
class EMOE(nn.Module):

    def __init__(self, args):
        super(EMOE, self).__init__()
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.len_v = args.len_v
        self.len_a = args.len_a
        self.len_align = self.len_v
        self.aligned = args.need_data_aligned
        self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.attn_mask = args.attn_mask
        self.fusion_method = args.fusion_method
        output_dim = args.num_classes
        self.args = args

        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a,
                                kernel_size=args.conv1d_kernel_size_a,
                                padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v,
                                kernel_size=args.conv1d_kernel_size_v,
                                padding=0, bias=False)



        self.encoder_c = nn.Conv1d(self.d_a, self.d_a, kernel_size=1,
                                   padding=0, bias=False)

        self.self_attentions_v = self.get_network(self_type='v')
        self.self_attentions_a = self.get_network(self_type='a')

        self.proj1_v = nn.Linear(self.d_v, self.d_v)
        self.proj2_v = nn.Linear(self.d_v, self.d_v)
        self.out_layer_v = nn.Linear(self.d_v, output_dim)
        self.proj1_a = nn.Linear(self.d_a, self.d_a)
        self.proj2_a = nn.Linear(self.d_a, self.d_a)
        self.out_layer_a = nn.Linear(self.d_a, output_dim)

        if self.fusion_method == "sum":
            self.proj1_c = nn.Linear(self.d_a, self.d_a)
            self.proj2_c = nn.Linear(self.d_a, self.d_a)
            self.out_layer_c = nn.Linear(self.d_a, output_dim)
        elif self.fusion_method == "concat":
            self.proj1_c = nn.Linear(self.d_a * 2, self.d_a * 2)
            self.proj2_c = nn.Linear(self.d_a * 2, self.d_a * 2)
            self.out_layer_c = nn.Linear(self.d_a * 2, output_dim)



        self.Router = router(
            self.orig_d_a * self.len_align + self.orig_d_v * self.len_align,
            2,
            self.args.temperature,
        )
        self.transfer_a_ali = nn.Linear(self.len_a, self.len_align)

    # build one cross modal transformer branch
    def get_network(self, self_type='v', layers=-1):
        if self_type == 'a':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type == 'v':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
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

    def get_net(self, name):
        return getattr(self, name)

    # encode both modalities then fuse with the router
    def forward(self, audio, video, audio_lens=None, force_w=None):





        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)




        if force_w is not None:
            m_w = force_w.to(audio.device)
        elif not self.aligned:
            audio_ = self.transfer_a_ali(audio.permute(0, 2, 1)).permute(0, 2, 1)
            m_i = torch.cat((video, audio_), dim=2)
            m_w = self.Router(m_i)
        else:
            m_i = torch.cat((video, audio), dim=2)
            m_w = self.Router(m_i)

        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)


        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)

        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        c_v_att = self.self_attentions_v(c_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]
        c_a_seq = self.self_attentions_a(c_a)
        if type(c_a_seq) == tuple:
            c_a_seq = c_a_seq[0]
        c_a_att = c_a_seq[-1]

        v_proj = self.proj2_v(
            F.dropout(F.relu(self.proj1_v(c_v_att), inplace=True),
                      p=self.output_dropout, training=self.training))
        v_proj += c_v_att
        logits_v = self.out_layer_v(v_proj)
        a_proj = self.proj2_a(
            F.dropout(F.relu(self.proj1_a(c_a_att), inplace=True),
                      p=self.output_dropout, training=self.training))
        a_proj += c_a_att
        logits_a = self.out_layer_a(a_proj)



        if self.fusion_method == "sum":
            c_fusion = c_v_att * m_w[:, 0:1] + c_a_att * m_w[:, 1:2]
        elif self.fusion_method == "concat":
            c_fusion = torch.cat([c_v_att * m_w[:, 0:1],
                                  c_a_att * m_w[:, 1:2]], dim=1) * 2

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True),
                      p=self.output_dropout, training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)

        res = {
            'logits_c': logits_c,
            'logits_v': logits_v,
            'logits_a': logits_a,
            'channel_weight': m_w,
            'c_proj': c_proj,
            'v_proj': v_proj,
            'a_proj': a_proj,
            'c_fea': c_fusion,
        }
        return res
