# CardioSentinel: a multi-model federated framework for ECG-based heart disease detection

Four clients ("hospitals") each train a **different deep-learning architecture** on their **own,
patient-separated share** of PTB-XL. They never share ECGs or model parameters: each one sends only
calibrated class probabilities, once, to a server that combines them with weights on the simplex:

```
P_final(y|x) = w_A·P_A + w_B·P_B + w_C·P_C + w_D·P_D,    w_i ≥ 0,  Σ w_i = 1
```

Findings the combined model is sure about are reported automatically; the rest are referred to a
cardiologist. The CardioSentinel web portal puts this in front of clinicians and patients.

| Client | Architecture | Params | PTB-XL training folds | Training ECGs |
|---|---|---|---|---|
| A | 1D CNN (4 conv blocks) | 0.44M | 1, 2 | 4,277 |
| B | 1D ResNet (16 residual blocks) | 6.73M | 3, 4 | 4,280 |
| C | BiLSTM (2 layers, 128 units) | 0.67M | 5, 6 | 4,264 |
| D | CNN-LSTM (3 conv + BiLSTM) | 0.47M | 7, 8 | 4,263 |

- **Task:** multi-label classification of the 5 PTB-XL diagnostic superclasses: NORM (normal), MI
  (myocardial infarction), STTC (ST/T change), CD (conduction disturbance), HYP (hypertrophy).
- **Data:** PTB-XL v1.0.3, 10-s 12-lead ECGs at 100 Hz; 21,799 → 21,388 labelled ECGs (18,617 patients).
- **Splits:** official patient-separated folds. Folds 1–8 train the clients (17,084 ECGs), fold 9
  (2,146) is used for early stopping, calibration, thresholds, aggregation weights and triage bands,
  and fold 10 (2,158) is the untouched test set, evaluated once.
- **Aggregators compared:** equal weights, static learned weights and a dynamic per-patient gate,
  by patient-grouped 5-fold CV on fold 9. A more complex one is used only if it is ≥ 0.002 AUROC
  better. Equal weights were selected in all 3 seeds.

B.Tech BCSE497J project, Vellore Institute of Technology: Sanvi Saket (23BDS0210) and Siddhi J Salunkhe
(23BDS0243), guided by Dr. Nithya K.

The design is in [Claude.MD](Claude.MD) and the full log of results in [RESULTS.md](RESULTS.md).

| Deliverable | Where |
|---|---|
| Paper (IEEE two-column, one-column PDF, Word) | [paper/main.pdf](paper/main.pdf), [paper/CardioSentinel_onecolumn.pdf](paper/CardioSentinel_onecolumn.pdf), [.docx](paper/CardioSentinel_onecolumn.docx) |
| Project-I report (VIT template, 58 pages) | [report/CardioSentinel_Project_Report.pdf](report/CardioSentinel_Project_Report.pdf), [.docx](report/CardioSentinel_Project_Report.docx) |
| Trained models, predictions, tables, figures | [outputs/fourclient/](outputs/fourclient/), [outputs/images/](outputs/images/) |
| Web portal | [web/](web/README.md) |
| Sample ECGs to try in the portal | [data/test_pdfs/](data/test_pdfs/) |

## Results (test fold 10, mean of seeds 42, 43, 44)

| Model | Macro AUROC | Macro F1 | Precision | Recall | Label-wise acc. | Exact match |
|---|---|---|---|---|---|---|
| Client A, 1D CNN | 0.888 | 0.680 | 0.646 | 0.726 | 0.848 | 0.509 |
| Client B, 1D ResNet | 0.882 | 0.672 | 0.630 | 0.734 | 0.840 | 0.506 |
| Client C, BiLSTM | 0.894 | 0.692 | 0.662 | 0.733 | 0.855 | 0.542 |
| Client D, CNN-LSTM | 0.896 | 0.690 | 0.670 | 0.722 | 0.857 | 0.547 |
| **Federated gate (equal weights)** | **0.905** [0.897, 0.913] | **0.707** | **0.698** | 0.725 | **0.869** | **0.581** |
| Pooled reference: CNN-LSTM on folds 1–8 | 0.909 | 0.707 | 0.685 | 0.740 | 0.864 | 0.566 |

- The gate beats the best single client in every seed: ΔAUROC +0.012, +0.011, +0.009, with the
  paired-bootstrap 95% CI above 0 each time.
- It comes within 0.004 AUROC of pooled training and recovers about 70% of the AUROC lost by
  splitting the data.
- Other federated methods (seed 42, all on Client D's CNN-LSTM, because they need one architecture):
  FedAvg 0.907, FedProx 0.906, FedNova 0.907 after 20 rounds, SCAFFOLD 0.842. The gate needs 1 round
  and any mix of architectures. Pooled traditional ML on 172 hand-crafted features: XGBoost 0.909,
  Random Forest 0.902, SVM 0.898, logistic regression 0.880.

**Overall confusion matrix of the gate** (all 5 classes, 2,158 ECGs × 5 = 10,790 decisions, mean of 3 seeds):

|  | Predicted yes | Predicted no |
|---|---|---|
| **Actually yes** | TP 2,177 | FN 615 |
| **Actually no** | FP 800 | TN 7,198 |

Micro precision 0.731, recall 0.780, accuracy 0.869. The table above uses macro averages, which
weight the 5 classes equally, so the weaker HYP class (AUROC 0.83) pulls them down. Per-class
matrices: `outputs/fourclient/figures/confusion_matrices.png`.

**Per-finding triage:**
- Each finding is ruled in (p ≥ h), ruled out (p ≤ l) or referred to a cardiologist, using bands fitted on fold 9.
- On the test set, auto-decided findings reach **0.937 precision, 0.908 recall and 0.967 accuracy**
  (macro over automated classes, 3 seeds). Without triage the figures are 0.698 / 0.725 / 0.869.
- This covers 53.7% of findings; 17.8% of ECGs are fully automatic. HYP is rule-out only and CD
  recall (0.880) is below the 0.90 target.

**Explanations:** Integrated Gradients per lead and time step (`paper/scripts/xai.py`). MI evidence
sits in V2–V4 on the QRS complexes and ST segments; removing the top-3 attributed leads lowers the
prediction about twice as much as removing 3 random leads.

**Photos and scans of paper ECGs** (`src/cardiomoe/images/`): a U-Net trace segmenter and a
digitizer turn the picture into a signal, which image-tuned copies of the four clients classify.
ECG-Image-Database (printouts of PTB-XL records) supplies 104 training, 60 validation and 100 test
photos on the same folds. On the test photos the image gate reaches AUROC 0.841, against 0.946 for
the recorded signals of the same ECGs. Photo triage is below the 0.90 target, so every photo finding
is referred to a doctor.

## CardioSentinel portal

[web/](web/README.md) is the clinician and patient portal built on these models (FastAPI + SQLite +
a static single-page app). Patients upload an ECG file (CSV, NPY, WFDB) or up to 4 photos / scans /
PDFs of a paper ECG, see their findings, keep a symptom journal, share results with their doctor and
ask a voice assistant about their own records. Clinicians review, sign off and release results.

```bash
python web/server.py     # http://127.0.0.1:8000, demo logins clinician@ / patient@cardiosentinel.demo, password demo1234
```

The voice assistant uses an LLM through OpenRouter. Put your key in `web/.env` (git-ignored):

```bash
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openrouter/auto     # optional
```

Without a key it answers offline from the matching records.

To try uploads, use the files in `data/test_pdfs/`. Six test-fold ECGs are provided: one per class,
plus an MI + STTC case. Each comes as a CSV signal, a rendered 3x4 PDF, a PNG and a phone-photo-like
JPG, and `answer_key.json` gives the true labels and the portal's output for each file. CSV uploads
give an automatic result. PDF, PNG and photo uploads are digitized, and every finding is referred to
a doctor.

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

# or all of them, detached from the terminal:
python scripts/launch_detached.py outputs/fourclient/run.log bash scripts/run_fourclient.sh 42
```

Reference experiments, after the steps above:

```bash
python -m cardiomoe.fourclient.train --clients ALL --seeds 42   # one CNN-LSTM on all training folds
python -m cardiomoe.fourclient.baseline --seeds 42              # baseline vs ensemble, client subsets
```

Photo pipeline (optional):

```bash
python scripts/fetch_ecg_images.py --folds 1 2 3 4 5 6 7 8 9 10   # ECG-Image-Database, PTB-XL part
python -m cardiomoe.images.real --folds 1 2 3 4 5 6 7 8 9          # digitize the photos once
python -m cardiomoe.images.finetune --seeds 42                     # image clients, gate and triage
python -m cardiomoe.images.evaluate --set real-test                # one-time test on fold-10 photos
```

Every script accepts `--debug`, which runs end to end on CPU in a few minutes. Outputs go to
`outputs/fourclient/` (and `outputs/images/` for the photo pipeline):
- `seed*/`: checkpoints, histories and predictions
- `tables/`: comparison, per-class metrics, significance, baselines, client subsets, triage
- `figures/`: training curves, model comparison, confusion matrices and weights

Paper experiments (traditional ML, FedAvg / FedProx / SCAFFOLD / FedNova, explanations, ablations)
are in `paper/scripts/`.

## Paper and report

```bash
python paper/scripts/make_results.py           # regenerate the paper's tables from outputs/fourclient/
python paper/scripts/make_onecolumn.py         # paper/CardioSentinel_onecolumn.pdf
python paper/scripts/make_docx.py              # paper/CardioSentinel_onecolumn.docx (pandoc)
python report/make_report.py --template <VIT BCSE497J template .docx>   # report/CardioSentinel_Project_Report.{docx,pdf}
```

The report reads its result tables and figure captions from the paper's LaTeX sources, so the two
always agree. It needs python-docx, tectonic and LibreOffice. Add `--debug` to any of these for a
quick build.

## Tests

```bash
pytest       # 37 tests: patient-level splits and leakage, labels, signal processing, models, gate, metrics, images, web
```
