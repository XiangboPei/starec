# StaRec

Anonymous implementation of StaRec, a sequential recommender that combines a
fixed wavelet-packet spectral branch with causal self-attention. The repository
also includes a DuoRec baseline evaluated with the same data split and full-sort
protocol.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CUDA is used when available. Pass `--no_cuda` to run on CPU.

## Data

Place each processed dataset at `data/<Dataset>.txt`. Every line contains one
chronological user sequence:

```text
user_id item_1 item_2 item_3 ...
```

Item IDs must be positive consecutive integers; `0` is reserved for padding.
The last interaction is used for testing, the penultimate interaction for
validation, and earlier interactions for training. See `data/README.md` for the
dataset names used in the paper.

## Training

Train StaRec on ML-1M:

```bash
python main.py \
  --model_type starec \
  --data_name ML-1M \
  --train_name starec_ml1m \
  --lr 0.0003 \
  --hidden_dropout_prob 0.3 \
  --attention_probs_dropout_prob 0.3 \
  --wave db2 \
  --alpha 0.2
```

Train the shared-pipeline DuoRec reproduction:

```bash
python main.py \
  --model_type duorec \
  --data_name ML-1M \
  --train_name duorec_ml1m \
  --lr 0.0003 \
  --hidden_dropout_prob 0.3 \
  --attention_probs_dropout_prob 0.3 \
  --contrastive_weight 0.1 \
  --contrastive_temperature 1.0
```

Repeat the command with the dataset names listed in `data/README.md` to run the
full table. Outputs are written to `output/`.

Evaluate a checkpoint:

```bash
python main.py \
  --do_eval \
  --model_type starec \
  --data_name ML-1M \
  --load_model starec_ml1m \
  --train_name starec_ml1m_eval
```

## Evaluation

All models use leave-one-out splitting and full-sort ranking. Items already
observed in the applicable history are masked before ranking. The reported
metrics are HR@5, NDCG@5, HR@10, and NDCG@10; validation NDCG@10 selects the
checkpoint.

The complete paper table is available in both machine-readable and LaTeX form:

- `results/main_results.csv`
- `results/main_results.tex`

The added DuoRec values are shared-pipeline reproductions, not copied official
scores:

| Dataset | HR@5 | NDCG@5 | HR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| ML-1M | 0.2174 | 0.1490 | 0.3078 | 0.1781 |
| Digital Music | 0.0817 | 0.0625 | 0.1073 | 0.0708 |
| Musical Instruments | 0.0726 | 0.0594 | 0.0917 | 0.0656 |
| Office Products | 0.0893 | 0.0781 | 0.1010 | 0.0818 |

## Tests

Run `pytest -q` after installing pytest in the active environment.

## DuoRec Baseline

The DuoRec integration implements the `us_x` objective with a second dropout
view, same-target semantic positives, and dot-product InfoNCE.
