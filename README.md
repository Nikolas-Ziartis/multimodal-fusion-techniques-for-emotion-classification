# dmd_va — 2-modality (video + audio) DMD for emotion classification

A minimal, tagged adaptation of the official DMD code (Li, Wang & Cui,
*Decoupled Multimodal Distilling for Emotion Recognition*, CVPR 2023,
<https://github.com/mdswyz/DMD>) from 3-modality (L/V/A) sentiment **regression**
to 2-modality (V/A) categorical emotion **classification**.

It reads the project's shared pretrained-feature cache and actor-grouped folds,
so it can be compared head-to-head against the other methods on identical
samples, splits and label indices.

## Licence and provenance

The upstream code is MIT licensed; the licence is reproduced in
`THIRD_PARTY_LICENSE_DMD.txt` as that licence requires.

The official model / trainer / GD-kernel / loss files were copied in and edited
**in place**, every change tagged `# DMD-VA:`. `diffs/` holds the unified diff of
each against its pristine original, so the adaptation is fully auditable.
Changed-line counts (content only, EOL-normalised):

```
model/dmd_repo.py                            (official dmd.py)   +63  -132   net removal: language stripped
dmd_trainer.py                               (official DMD.py)   +173 -104   regression -> CE + diagnostics
distillnets/get_distillation_kernel.py       (hetero)            +4   -3     2 real edits
distillnets/get_distillation_kernel_homo.py  (homo)              +4   -3     2 real edits
distillnets/misc.py                          (utils/misc.py)     +5   -2     min_cosine weighting fix
losses/HingeLoss.py                          (HingeLoss.py)      +5   -2     device + categorical margin
```

The MulT encoder (`model/transformer.py`, `model/multihead_attention.py`,
`model/position_embedding.py`) is copied **verbatim** from the DMD repo. The
launcher, dataset, config and metrics are new files with no DMD-repo
counterpart.

## The three surgeries (documented deviations)

1. **3 → 2 modalities.** Language dropped everywhere; fusion 9d → 4d, c_fusion
   3d → 2d, HeteroGD memory transformers 2d → d (each modality reinforced by one
   other), GD graph 6 edges → 2 (`edges[0]` = A→V, `edges[1]` = V→A).
2. **Regression → classification.** Every head output_dim 1 → K; task loss L1 →
   CE over the 6 surviving heads; GD `W_logit` input 1 → K; ordinal margin
   `0.15·|y_i − y_j|` → constant 0.15 on different-class pairs.
3. **GD edge prior** symmetric over the 2 edges (both modalities strong here).

## Forced changes for modern PyTorch

- GD prior: `Variable(...).cuda()` → `register_buffer` (follows `.to(device)`).
- `min_cosine` / `ort`: `CosineEmbeddingLoss` is given 3-D tensors in the
  release (rejected by current torch); flattened to 2-D, cosine over the feature
  dimension.
- `min_cosine` uses `reduction='none'` so the per-edge distillation weights apply
  per sample (the release silently averaged them away).
- `'l1'` kept as the DMD-faithful distill metric; `l2` / `cosine` / `kl`
  selectable via `--gd_metric`. The `kl` path emits a benign torch deprecation
  about `kl_div` reduction; the default `l1` path is warning-free.

## Padding and pooling (faithful to the DMD release)

HeteroGD pools the last step (`h[-1]`) and MulT's `attn_mask` is causal, not a
padding mask, so DMD attends over zero-padded audio. This matches the released
DMD behaviour; if you compare against another method, apply the same masking
choice there too.

## Run

Run from the directory that *contains* this folder.

```bash
python -m dmd_va.train_dmd --dataset ravdess --cache_dir ./cache
python -m dmd_va.train_dmd --dataset cremad  --cache_dir ./cache
python -m dmd_va.train_dmd --dataset mead8   --cache_dir ./cache
```

Outputs go to `./experiments/dmd_va/<dataset>/` (gitignored):
`per_fold_results.csv`, `aggregate_results.csv`,
`diagnostics/fold{k}_{history,test_samples,confusion}.csv` and `fold{k}_best.pth`.

## The one data-contract touchpoint

If your `.npz` keys differ from `video_feats / audio_feats / audio_lens /
labels / actors`, adjust `dataset.load_npz` only. Audio may be a fixed
`(N, 250, 768)` array or a ragged object array of `(T_i, 768)`; both are handled.

## Not included

The plotting and hyper-parameter sweep scripts are not part of this repository.
`train_dmd.py` does not import them.
