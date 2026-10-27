# Paper notes: facts, numbers, and gaps (Step 0)

Compiled 2026-10-21 from the repository at commit `607a2fd`. Every number below comes from a
file in the repo; the source is given for each one. Nothing has been drafted yet.

---

## 1. What the project actually is

The system is **CardioMoE**, a four-client ECG classifier with weighted gated aggregation
(`CLAUDE.md`, `README.md`, `src/cardiomoe/fourclient/`).

- **Task:** multi-label classification of 12-lead ECGs into 5 PTB-XL diagnostic superclasses:
  NORM, MI, STTC, CD, HYP (`src/cardiomoe/data/labels.py`). There is **no regression task**.
- **Modality:** one modality only, the raw 12-lead signal at 100 Hz, shape (12, 1000)
  (`src/cardiomoe/fourclient/models.py`). An earlier three-modality design (ECG + text + image)
  was removed on 2026-10-17. Its code is gone, and it appears only as history in `RESULTS.md`
  (Phases 0–3).
- **"Federated" setup:** four clients, each holding a disjoint, patient-separated share of the
  training folds. Each client trains a **different architecture**. Clients never share data or
  parameters, and the server combines only their **calibrated output probabilities**:
  `P(y|x) = Σ_i w_i P_i(y|x)`, with w on the simplex (`src/cardiomoe/fourclient/gate.py`).
  This is a **one-shot, prediction-level aggregation** (closer to a federated ensemble or
  mixture of experts). There is **no FedAvg, no communication rounds, no local epochs per
  round, and no parameter averaging**. Averaging is impossible anyway, because the
  architectures differ (CLAUDE.md §2.5).
- **Explainability:** **none implemented**. There is no SHAP, Grad-CAM, saliency or attention
  code anywhere in `src/`, `web/` or `scripts/`.
- **Attention models:** **none**. There is no CNN+Attention model.

## 2. Data (`src/cardiomoe/data/ptbxl.py`, `signal.py`, `datasets.py`, `RESULTS.md` Phase 0)

| Fact | Value | Source |
|---|---|---|
| Dataset | PTB-XL v1.0.3, PhysioNet, CC BY 4.0, `records100/` (100 Hz) | CLAUDE.md §3, `scripts/download.sh` |
| Records / patients in the raw data | 21,799 / 18,869 | RESULTS.md Phase 0 |
| Records kept (≥1 diagnostic superclass) | 21,388 (411 dropped) | CLAUDE.md §3 |
| Train (folds 1–8) / val (fold 9) / test (fold 10) | 17,084 / 2,146 / 2,158 | `seed42/ALL/meta.json`, RESULTS.md |
| Test patients | 1,877 | RESULTS.md (3-seed section) |
| Per-client train sizes | A 4,277, B 4,280, C 4,264, D 4,263 | `seed*/X/meta.json` |
| Test positives per class (support) | NORM 963, MI 550, STTC 521, CD 496, HYP 262 | `tables/per_class.csv` (tp+fn) |
| Label source | `scp_codes` → `diagnostic_class` of `scp_statements.csv` rows with `diagnostic == 1` | `labels.py` |
| Leads | I, II, III, aVR, aVL, aVF, V1–V6, reordered case-insensitively | `signal.py` |
| Cleaning | NaN/inf → 0 (count logged); crop/zero-pad to 1,000 samples | `signal.py: clean, fix_length` |
| Filter | Butterworth band-pass 0.5–40 Hz, order 3, zero-phase (`sosfiltfilt`) | `signal.py: bandpass` |
| Storage | float16 [N, 12, 1000] in mV, un-normalised | `ptbxl.py: load_signals` |
| Normalisation (at load time) | per-lead z-score per recording, clipped to ±10; flat leads → 0 | `signal.py: normalize` |
| Augmentation (train only) | amplitude ×U(0.9, 1.1); white noise at SNR U(20, 30) dB; baseline wander 0.1–0.5 Hz with amplitude U(0.05, 0.3) SD; lead dropout p = 0.05 | `signal.py: augment` |
| Class imbalance | BCE `pos_weight` = neg/pos, from training prevalence | `train/common.py: pos_weight_from` |
| Split | official PTB-XL stratified folds, patient-level; **80/10/10**, not 80/20 | `configs/fourclient.yaml` |

**Missing:** the class distribution of the full dataset and of the training folds. The
prevalence table (`ptbxl_prevalence.csv`) is not in the repo, and `data/` is absent. Computing
it needs `ptbxl_database.csv` and `scp_statements.csv` (a few MB, downloadable).

## 3. Models (`src/cardiomoe/fourclient/models.py`, `meta.json`)

| Client | Architecture (exact) | Params |
|---|---|---|
| A | 1D CNN: 4 × [Conv1d k7 (no bias) → BN → ReLU → MaxPool 2], channels 32/64/128/256; avg+max pool → 512 | 437,317 |
| B | 1D ResNet: stem Conv k7/s2, 64 ch, BN, ReLU, MaxPool 3/s2; residual blocks k5, stages [64,128,256,256] × [3,4,6,3]; zero-init last BN; avg-pool shortcut; avg+max → Linear 512→256 | 6,729,349 |
| C | BiLSTM: 2 layers, hidden 128, bidirectional, inter-layer dropout 0.2, over 1,000 steps; avg+max pool → 512 | 673,285 |
| D | CNN-LSTM: 3 conv blocks (32/64/128, k7, MaxPool 2; 1000 → 125 steps) → 1-layer BiLSTM, hidden 128; avg+max → 512 | 471,621 |
| Head (all) | Linear(out_dim → 256) → ReLU → Dropout 0.3 → Linear(256 → 5) | — |
| Calibration | one temperature per client, fitted on fold 9 by LBFGS on BCE, clamped to [0.05, 20] | `fusion/calibration.py` |

Temperatures (seeds 42/43/44): A 1.990/1.464/1.774; B 1.420/1.411/1.728; C 1.126/1.352/1.270;
D 1.302/1.311/1.433. Epochs run (with early stopping): A 21/18/23; B 19/20/26; C 34/37/34;
D 31/28/33; ALL 33/35/25.

## 4. Training (`configs/fourclient.yaml`, `train/common.py`)

| Hyperparameter | Value |
|---|---|
| Loss | BCEWithLogits with `pos_weight` (masked for −1 labels) |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-2 |
| Schedule | OneCycleLR (max lr 1e-3, PyTorch defaults otherwise) |
| Batch size | 128 (eval batch 128) |
| Max epochs | 50; early stopping on fold-9 macro AUROC, patience 8 |
| Gradient clipping | max-norm 1.0 |
| Mixed precision | CUDA only (not used: trained on Apple MPS) |
| Seeds | 42, 43, 44 |
| Hardware | Apple M1 MacBook Air, 8 GB, PyTorch MPS backend; triage gate re-fit on a Windows CPU (RESULTS.md) |
| Software | Python 3.11.17, PyTorch "2.14.1" according to RESULTS.md Phase 0. **Needs checking**: this machine currently has Python 3.10 / torch 2.11 and no `.venv`. |

**Aggregation** (`gate.py`, `aggregate.py`; config `fourclient.gate`):
- Equal: w_i = 1/4.
- Static: 4 logits → softmax, full-batch Adam lr 0.05, 500 steps, BCE on mixed probabilities.
- Dynamic: an MLP over [20 probabilities + 4 mean binary entropies] → 32 → ReLU → Dropout 0.2 →
  4 → softmax. The last layer is zero-initialised, so it starts at equal weights. AdamW lr 1e-3,
  wd 1e-2, up to 300 epochs, patience 30, on a 20% patient-grouped inner hold-out.
- Selection: patient-grouped 5-fold CV inside fold 9. Candidates go simplest-first; a more
  complex combiner must beat the current one by ≥ 0.002 CV macro AUROC.
- Thresholds: per class, chosen to maximise F1 on fold 9.
- Triage: per-class rule-in/rule-out bands (§6).

## 5. Results (all fold-10 numbers come from a single evaluation; nothing was tuned on fold 10)

### 5.1 Main comparison (mean ± SD over seeds 42–44; `tables/comparison.md`, `comparison.csv`)

| Model | Test macro AUROC [95% CI] | Macro F1 | Prec | Rec | Spec | Exact-match | Label-wise acc |
|---|---|---|---|---|---|---|---|
| A 1D CNN | 0.888 ± 0.002 [0.879, 0.897] | 0.680 | 0.646 | 0.726 | 0.867 | 0.509 | 0.848 |
| B 1D ResNet | 0.882 ± 0.003 [0.873, 0.891] | 0.672 | 0.630 | 0.734 | 0.857 | 0.506 | 0.840 |
| C BiLSTM | 0.894 ± 0.000 [0.885, 0.902] | 0.692 | 0.662 | 0.733 | 0.877 | 0.542 | 0.855 |
| D CNN-LSTM | 0.896 ± 0.001 [0.887, 0.904] | 0.690 | 0.670 | 0.722 | 0.882 | 0.547 | 0.857 |
| **Equal weights (selected)** | **0.905 ± 0.000 [0.897, 0.913]** | **0.707** | **0.698** | 0.725 | **0.894** | **0.581** | **0.869** |
| Static weights | 0.903 ± 0.001 | 0.703 | 0.685 | 0.728 | 0.887 | 0.570 | 0.866 |
| Dynamic gate | 0.901 ± 0.002 | 0.702 | 0.681 | 0.729 | 0.888 | 0.568 | 0.864 |
| All-data CNN-LSTM (pooled folds 1–8) | 0.909 ± 0.001 | 0.707 | 0.685 | 0.740 | 0.887 | 0.566 | 0.864 |

The macro AUPRC is computed by `evaluate.py` and stored in `seed*/test_results.json`; it still
needs extracting.

### 5.2 Fold-9 selection (`seed*/gate/aggregate.json`, RESULTS.md)

| Seed | A | B | C | D | CV equal | CV static | CV dynamic | Static w A/B/C/D |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.894 | 0.883 | 0.901 | 0.898 | 0.908 | 0.907 | 0.904 | 0.00/0.01/0.70/0.28 |
| 43 | 0.893 | 0.889 | 0.901 | 0.897 | 0.909 | 0.907 | 0.903 | 0.00/0.05/0.50/0.45 |
| 44 | 0.892 | 0.886 | 0.899 | 0.900 | 0.909 | 0.908 | 0.905 | 0.00/0.07/0.44/0.49 |

Mean static weights: A 0.001, B 0.042, C 0.549, D 0.407. Mean dynamic weights on test:
0.002/0.092/0.532/0.374.

### 5.3 Significance: ensemble − best client, paired bootstrap, 1,000 resamples (`significance.json`)

| Seed | Best client | Δ AUROC [CI] | Δ macro F1 [CI] |
|---|---|---|---|
| 42 | C | +0.012 [+0.009, +0.016] | +0.018 [+0.009, +0.028] |
| 43 | C | +0.011 [+0.007, +0.014] | +0.010 [−0.001, +0.021] |
| 44 | D | +0.009 [+0.005, +0.013] | +0.024 [+0.014, +0.034] |

Pooled-model baseline − ensemble: ΔAUROC +0.003/+0.005/+0.003; ΔF1 −0.012/+0.012/−0.001.

### 5.4 Per-class test AUROC / F1, mean of 3 seeds (RESULTS.md)

| Class | C | D | Ensemble |
|---|---|---|---|
| NORM | 0.931 / 0.842 | 0.933 / 0.853 | 0.938 / 0.850 |
| MI | 0.901 / 0.702 | 0.906 / 0.715 | 0.912 / 0.725 |
| STTC | 0.926 / 0.741 | 0.923 / 0.727 | 0.933 / 0.768 |
| CD | 0.902 / 0.720 | 0.909 / 0.716 | 0.914 / 0.735 |
| HYP | 0.809 / 0.454 | 0.808 / 0.438 | 0.828 / 0.461 |

Seed-42 equal-weight confusion matrices (TP/FP/FN/TN), thresholds, and precision/recall/specificity
per class are in `tables/per_class.csv`, e.g. NORM 887/236/76/959 and HYP 129/168/133/1728.

### 5.5 Leave-one-client-out, equal-weight subsets (`tables/client_subsets.csv`)

Test AUROC: ABCD 0.905, ACD 0.905, BCD 0.904, ABD 0.902, ABC 0.901, CD 0.903, D 0.896, C 0.894,
A 0.888, B 0.882 (all 15 subsets with fold-9 values in the CSV).

### 5.6 Triage (`tables/triage.md`, `triage.json`)

On auto-decided findings: macro precision 0.937, macro recall 0.908 (automated classes),
accuracy 0.967. Coverage is 53.7% of findings and 17.8% of ECGs fully automatic. Per class:
NORM P/R 0.927/0.937; MI 0.940/0.915; CD 0.935/0.880; STTC automated in 1 of 3 seeds; HYP rule-out only.

### 5.7 Training curves (`seed*/X/history.jsonl`, `figures/training_curves.png`)

Per epoch, the logs hold **training loss, fold-9 macro AUROC, and seconds** only. There is no
validation loss, no accuracy curve and no test-over-time curve.

### 5.8 Existing figures (`outputs/fourclient/figures/`)

`training_curves.png`, `model_comparison.png`, `confusion_matrices.png`,
`aggregation_weights.png`. All are seed 42 where per-model.

---

## 6. Mismatches between the requested paper template and the project

| Template item | Status in the project |
|---|---|
| paper_refs/ (layout.jpeg + 5 PDFs) | **Not in the repo.** Only `~/Downloads/vi.explainability.pdf` (HAMF, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3563177) was found. |
| "≈20,000 samples" | 21,388 records (matches). |
| 80/20 train/test split | Actual split is 80/10/10 (folds 1–8 / 9 / 10). CLAUDE.md forbids changing the test split. |
| CNN → CNN+Attention → LSTM hybrid (≈93%) → Federated (≈96%) | **Does not exist.** There is no attention model. No accuracy of 93% or 96% was produced. Real progression: 1D CNN 0.888 → ResNet 0.882 → BiLSTM 0.894 → CNN-LSTM 0.896 → ensemble 0.905 macro AUROC; label-wise accuracy 0.848 → 0.869. |
| Multimodal; feature-level fusion; unimodal vs multimodal (Table 2); modality ablation | Single modality. Fusion is **decision-level** (probability averaging). |
| Regression task (Section III-D) | **None.** Classification only. |
| FedAvg, rounds, local epochs; FedProx/SCAFFOLD/FedNova comparison; server accuracy over rounds | **None.** One-shot probability aggregation across heterogeneous architectures. |
| Explainable AI (SHAP / Grad-CAM) | **None implemented.** |
| Traditional ML baselines (SVM, RF, KNN, LR, XGBoost) | **Not run.** |
| Activation / optimizer ablations | **Not run.** |
| Hand-crafted features, correlation heat-map | **None.** The models learn from the raw signal. |
| Accuracy as headline metric | The project reports macro AUROC (primary), F1, exact-match and label-wise accuracy (multi-label). |
| Per-client accuracy and loss curves over rounds | Per-epoch training loss and fold-9 AUROC per client exist. |
| LaTeX toolchain | `pdflatex` / `latexmk` are **not installed** on this Mac. |
| PTB-XL data for new experiments | `data/` is absent. Re-download needed (~0.55 GB). Trained models (`model.pt`) and saved probabilities (`val_preds.pt`, `test_preds.pt`) **are** present. |

## 7. Project rules that constrain new experiments (CLAUDE.md)

- Any new evaluation on fold 10 needs the owner's approval (`evaluate.py` needs `--rerun`).
- Nothing may be tuned on fold 10. No datasets other than PTB-XL. The test split may not change.
- Design decisions may not change without asking. New runs must support `--debug`, use seeds
  42/43/44, and be logged in `RESULTS.md`.

---

## 8. Drafting decisions (2026-10-22, after the owner said "make the paper")

The questions in Section 6 were not answered, so the draft uses these defaults. Each one is easy to
reverse.

- **Framing:** a one-shot, prediction-level federated ensemble of heterogeneous architectures. It is
  not described as FedAvg.
- **No invented numbers.** The ≈93% and ≈96% figures from the handwritten layout are not used. The
  headline metric is macro AUROC; label-wise accuracy, F1, precision, recall and exact-match
  accuracy are also reported.
- **Split:** reported as the real 80/10/10 (folds 1–8 / 9 / 10).
- **Sections without data are kept, honestly retitled or marked `\todo{}`:**
  - Traditional ML rows (Table IV): `\todo`. The text cites the Wavelet+NN result of Strodthoff
    2021 (0.874) as a literature reference only.
  - "Unimodal vs multimodal" becomes single clients vs the federated gate, per class.
  - "Modality ablation" becomes the client-subset ablation (15 subsets) plus Shapley values per
    client.
  - Fusion ablation uses equal / static / dynamic (real data).
  - Activation / optimizer tables: only the trained configuration (ReLU + AdamW, Client D, fold-9
    AUROC 0.898) is real; the other rows are `\todo`.
  - "Rounds" do not exist. Client curves are plotted per local epoch (fold-9 AUROC and training
    loss), and the server figure plots AUROC against the number of aggregated clients.
  - FedAvg / FedProx / SCAFFOLD / FedNova: `\todo` rows. The local, centralised and proposed rows are
    real.
  - Regression (III-D): states that there is none, with a `\todo` asking the owner to confirm.
  - XAI: Integrated Gradients on the seed-42 gate, computed **on fold 9 only** (no test-fold use),
    plus a deletion test and exact Shapley values from client_subsets.csv.
- **No new fold-10 evaluation was run.** ROC curves and confusion matrices are drawn from the saved
  `test_preds.pt` and `per_class.csv` of the evaluation that already happened.

## 9. Still missing (needs the owner)

1. Title confirmation, author names, affiliations, corresponding e-mail, funding, target journal.
2. The macOS version at training time.
3. Approval and compute time for: traditional ML baselines, activation/optimizer ablations, and the
   FedAvg/FedProx/SCAFFOLD/FedNova runs (each needs a new fold-10 evaluation, except the ablations,
   which can use fold 9).
4. Whether a regression target should be added, or III-D may stay as written.
5. Related work has 13 distinct verified 2022–2026 papers (target 15–20). More can be added once
   verified.

## 10. Update (2026-10-22, "complete the paper")

The traditional-ML, activation/optimizer and FedAvg/FedProx/SCAFFOLD/FedNova experiments were run (see
RESULTS.md) and are in the paper. The only `\todo`s left are the author names, affiliations,
membership and journal name in `main.tex`. The paper is 18 pages.
