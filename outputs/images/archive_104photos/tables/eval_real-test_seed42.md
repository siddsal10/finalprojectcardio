### Image pipeline, `real-test` (seed 42)

100 images of 62 records; digitized 98.0%; median per-lead correlation with the recorded signal 0.299. Gate: static.

| Model | Macro AUROC (95% CI, records) | Macro AUPRC | Macro F1 | Precision | Recall | Label-wise acc. | Exact match |
|---|---|---|---|---|---|---|---|
| Image gate (fine-tuned) | 0.841 (0.746-0.916) | 0.710 | 0.558 | 0.572 | 0.700 | 0.798 | 0.388 |
| Image client A | 0.841 (0.764-0.905) | 0.682 | 0.543 | 0.476 | 0.773 | 0.718 | 0.194 |
| Image client B | 0.841 (0.777-0.903) | 0.685 | 0.562 | 0.530 | 0.781 | 0.708 | 0.204 |
| Image client C | 0.778 (0.702-0.845) | 0.520 | 0.446 | 0.352 | 0.765 | 0.690 | 0.163 |
| Image client D | 0.841 (0.746-0.916) | 0.710 | 0.558 | 0.572 | 0.700 | 0.798 | 0.388 |
| Signal gate on digitized signal (no fine-tuning) | 0.859 (0.792-0.919) | 0.707 | 0.343 | nan | 0.532 | 0.612 | 0.082 |
| Signal gate on recorded signal (reference) | 0.946 (0.906-0.975) | 0.808 | 0.696 | 0.684 | 0.745 | 0.898 | 0.684 |

| Capture | Images | Digitized | Median r | AUROC image gate | AUROC signal gate on digitized | AUROC reference |
|---|---|---|---|---|---|---|
| colour scan | 20 | 100.0% | 0.441 | 0.608 | 0.706 | 0.818 |
| digital (green grid) | 10 | 100.0% | 0.177 | – | – | – |
| digital (red grid) | 10 | 100.0% | 0.778 | – | – | – |
| grayscale scan | 20 | 100.0% | 0.191 | 0.871 | 0.871 | 0.964 |
| phone photo | 10 | 80.0% | 0.340 | – | – | – |
| phone photo, stained | 10 | 100.0% | -0.004 | – | – | – |
| photo of a monitor | 10 | 100.0% | 0.336 | – | – | – |
| scan, stained | 10 | 100.0% | 0.119 | – | – | – |

Triage on the image gate: 50.0% of findings auto-decided, accuracy 0.943; 10.2% of images fully automatic.

Digitizer failures: found # rows of traces (1); could not follow the trace in row # (1)
