# SpaDTA Tutorial

This directory provides the main entry points for reproducing SpaDTA training,
preprocessing, and evaluation experiments. The scripts are thin configuration
wrappers around the shared workflow in `model/` and are intended to be run from
the repository root.

## Environment

```bash
conda activate spatialmeta
cd /data/user/hesy/projects/SpatialMETA/SpaDTA_718
```

## Main training entry points

### Spatial metabolomics (SM)

```bash
python tutorial/run_single_sample_sm.py --sample-name X49_T
```

`run_single_sample_sm.py` calls
`SpaDTA_718.model.train_eval_workflow.run_train_eval_cli` with the SM
configuration. It supports the ccRCC and FMP samples listed in
`target_clusters_by_sample` and uses:

- model-ready input under
  `/bigdat2/user/hesy/spatialmeta/SpatialMETA/SpaDTA_718_model_input/SM`;
- 300 training epochs and batch size 512;
- training seed 42 and clustering seed 0;
- 20-component PCA for evaluation;
- epoch-300 evaluation and saved embeddings at epochs 120, 180, 240, and 300;
- output under `runs/sm_final/<sample_name>`.

The default sample is `248_T` and the default device is `cuda:7`. The command
line `--sample-name` changes the sample without changing the source script.

### ATAC

```bash
python tutorial/run_single_sample_atac.py --sample-name Mouse_Brain_E18_S1
```

`run_single_sample_atac.py` uses the same shared training workflow with the
ATAC-specific configuration. Its defaults are:

- model-ready input under
  `/bigdat2/user/hesy/spatialmeta/SpatialMETA/SpaDTA_718_model_input/ATAC`;
- 300 training epochs and batch size 256;
- training seed 42 and clustering seed 2020;
- 20-component PCA for evaluation;
- epoch-300 evaluation and saved embeddings at epochs 100, 200, and 300;
- output under `runs/atac_final/<sample_name>`.

The default sample is `Mouse_Brain_E18_S1` and the default device is `cuda:4`.
Four mouse-brain samples are supported by the script.

## Preprocessing

Preprocessing must be completed before training. The scripts write model-ready
`h5ad` files and preserve the spatial coordinates required by the downstream
workflow.

### ccRCC and FMP SM data

```bash
python tutorial/preprocess_sm_ccrcc.py --sample-name X49_T
python tutorial/preprocess_sm.py --sample-name m1_FMP
python tutorial/preprocess_sm_GBM.py --sample-name 248_T
```

The ccRCC preprocessor reads aligned raw joint files from
`ccRCC/adata_joint_<sample>_raw.h5ad`, selects 2,000 transcript features and
800 metabolite features, and builds the SpaDTA model input. The FMP/GBM
preprocessors implement the corresponding dataset-specific input resolution.

### RNA plus ATAC data

```bash
python tutorial/preprocess_atac.py --sample-name Mouse_Brain_E18_S1
```

The ATAC preprocessor matches RNA and ATAC spots by `obs_names`, verifies their
spatial coordinates, selects RNA HVGs, applies SMART-style TF-IDF and PCA to
ATAC, and writes a joint model input. ATAC preprocessing PCA uses random seed
42; this preprocessing seed is independent of the training seed.

## Evaluation of saved embeddings

To evaluate an existing epoch snapshot with the final clustering protocol:

```bash
python tutorial/evaluate_single_sample.py \
  --run-dir runs/sm_final/X49_T \
  --epoch 300
```

The evaluator reads the saved embedding, applies PCA20 and fixed-k mclust, and
writes `final_protocol/epoch_0300/metrics.csv`, `metrics.json`,
`mclust_labels.npy`, and `spot_labels.csv`. The default clustering seed is 0
for SM and 2020 for ATAC. Use the epoch-300 output for the reported benchmark
results.

`use_mclust.py` and `use_target_leiden.py` are standalone utilities for
re-clustering existing result files; they do not retrain the model.

## Output convention

Each training run records its effective configuration and produces model
checkpoints, training history, saved epoch embeddings, clustering outputs, and
summary metadata in its run directory. Downstream metric tables should be
computed from the epoch-300 snapshot and retained alongside the corresponding
run directory for reproducibility.

## Notes

- Paths in the scripts target the shared `/bigdat2` data installation; adjust
  only the dataset roots when running on another machine.
- The preprocessing representation and seed are kept fixed when comparing
  training seeds.
- Existing main results are not overwritten by the multi-seed wrappers, which
  use separate output roots.
