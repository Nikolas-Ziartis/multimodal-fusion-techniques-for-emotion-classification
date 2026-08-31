import argparse

NUM_CLASSES = {'ravdess': 8, 'cremad': 6, 'mead8': 8, 'mead6': 6}




TRAINED_METHODS = ['audio', 'video', 'concat']
ALL_METHODS = TRAINED_METHODS + ['score_avg']

DISPLAY_NAME = {
    'audio':     'Audio only (unimodal)',
    'video':     'Video only (unimodal)',
    'concat':    'Feature concat (early fusion)',
    'score_avg': 'Score average (late fusion)',
}


# default settings shared by every baseline run
def base_config():
    return dict(

        len_v=30, len_a=250,
        feature_dim=768,


        hidden_dim=256,
        dropout=0.5,


        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=64,
        max_epochs=50,
        patience=5,
        early_stop=10,
        grad_clip=-1.0,
        KeyEval='Loss',
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
