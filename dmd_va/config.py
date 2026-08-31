import argparse

NUM_CLASSES = {'ravdess': 8, 'cremad': 6, 'mead8': 8, 'mead6': 6}


# default dmd settings shared by every run
def base_config():
    return dict(

        len_v=30, len_a=250,
        feature_dims=[768, 768],
        conv1d_kernel_size_v=5,
        conv1d_kernel_size_a=5,


        dst_feature_dim_nheads=[30, 6],
        nlevels=4,
        attn_mask=True,
        attn_dropout=0.4, attn_dropout_a=0.0, attn_dropout_v=0.0,
        relu_dropout=0.0, embed_dropout=0.0, res_dropout=0.0,
        output_dropout=0.5,


        learning_rate=1e-4,
        weight_decay=0.0,
        batch_size=16,
        update_epochs=10,
        grad_clip=0.6,
        patience=5,
        early_stop=10,
        KeyEval='Loss',
        margin=0.15,


        gd_size_homo=64, gd_size_hetero=32,
        gd_reg=10,
        w_losses=[1, 10],
        gd_metric='l1',
        gd_alpha=1 / 8,


        gd_prior_homo=[1.0, 1.0],
        gd_prior_hetero=[1.0, 1.0],
    )


# make the config for one dataset and apply any overrides
def build_args(dataset, cache_dir, overrides=None):
    cfg = base_config()
    cfg['dataset'] = dataset
    cfg['cache_dir'] = cache_dir
    cfg['num_classes'] = NUM_CLASSES[dataset]
    if overrides:
        cfg.update(overrides)
    return argparse.Namespace(**cfg)
