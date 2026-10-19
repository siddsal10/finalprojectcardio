# CardioMoE: four-client ECG classification with weighted gated aggregation

Four independent clients each train a different deep-learning architecture on their own
share of the PTB-XL training data. A gate learned on validation data combines their class
probabilities:

```
P_final(y|x) = w_A·P_A + w_B·P_B + w_C·P_C + w_D·P_D,    w_i ≥ 0,  Σ w_i = 1
```

| Client | Architecture | PTB-XL training folds |
|---|---|---|
| A | 1D CNN | 1, 2 |
| B | 1D ResNet | 3, 4 |
| C | BiLSTM | 5, 6 |
| D | CNN-LSTM | 7, 8 |

- **Task:** multi-label classification of the 5 PTB-XL diagnostic superclasses (NORM, MI, STTC, CD, HYP).
- **Splits:** official patient-separated folds. Fold 9 is used for validation (early stopping,
  calibration, thresholds and the aggregation weights). Fold 10 is the untouched test set.
- **Aggregators compared:** equal weights, static learned weights, and a dynamic per-patient gate.

The design is in [Claude.MD](Claude.MD) and the results are in [RESULTS.md](RESULTS.md).

**Per-finding triage:**
- Each finding is ruled in, ruled out or referred to a cardiologist, using bands fitted on fold 9.
- On the test set, auto-decided findings reach **0.937 precision, 0.908 recall and 0.967 accuracy**
  (macro over automated classes, 3 seeds). Without triage the figures are 0.698 / 0.725 / 0.869.
- This covers 54% of findings.

**CardioSentinel** ([web/](web/README.md)) is the clinician and patient portal built on these models:

```bash
python web/server.py     # http://127.0.0.1:8000, demo logins clinician@ / patient@cardiosentinel.demo, password demo1234
```

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

The trained models and their predictions for seeds 42–44 are committed under `outputs/fourclient/`,
so the portal and the triage step run without retraining. The data download (below) is only needed
for training, for the PTB-XL sample records in the portal, and for seeding the demo with real ECGs.

## Data

```bash
bash scripts/download.sh             # PTB-XL 1.0.3, 100 Hz (~0.6 GB, open access, CC BY 4.0)
python -m cardiomoe.data.ptbxl --prepare           # labels, splits, filtered signals -> data/processed/
```

Add `--debug` to either command to work with about 200 records.

## Run

```bash
python -m cardiomoe.fourclient.train --seeds 42      # train A-D (skips clients already trained)
python -m cardiomoe.fourclient.aggregate --seeds 42  # fit the aggregation weights on fold 9
python -m cardiomoe.fourclient.evaluate --seeds 42   # one-time evaluation on fold 10, tables + figures
python -m cardiomoe.fourclient.triage --seeds 42     # per-finding triage: fit on fold 9, apply once to fold 10

# or all three, detached from the terminal:
python scripts/launch_detached.py outputs/fourclient/run.log bash scripts/run_fourclient.sh 42
```

Reference experiments, after the steps above:

```bash
python -m cardiomoe.fourclient.train --clients ALL --seeds 42   # one CNN-LSTM on all training folds
python -m cardiomoe.fourclient.baseline --seeds 42              # baseline vs ensemble, client subsets
```

Every script accepts `--debug`, which runs end to end on CPU in a few minutes. Outputs go to
`outputs/fourclient/`:
- `seed*/`: checkpoints, histories and predictions
- `tables/`: comparison and per-class metrics
- `figures/`: training curves, model comparison, confusion matrices and weights

Run the tests with `pytest`.
