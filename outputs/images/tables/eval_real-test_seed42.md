### Image pipeline, `real-test` (seed 42)

100 images of 62 records; digitized 98.0%; median per-lead correlation with the recorded signal 0.299. Gate: equal.

| Model | Macro AUROC (95% CI, records) | Macro AUPRC | Macro F1 | Precision | Recall | Label-wise acc. | Exact match |
|---|---|---|---|---|---|---|---|
| Image gate (fine-tuned) | 0.890 (0.830-0.941) | 0.769 | 0.584 | 0.480 | 0.834 | 0.810 | 0.398 |
| Image client A | 0.870 (0.798-0.925) | 0.706 | 0.589 | 0.482 | 0.818 | 0.796 | 0.449 |
| Image client B | 0.869 (0.810-0.923) | 0.737 | 0.609 | 0.564 | 0.710 | 0.818 | 0.449 |
| Image client C | 0.848 (0.794-0.903) | 0.649 | 0.538 | 0.429 | 0.773 | 0.757 | 0.265 |
| Image client D | 0.860 (0.774-0.927) | 0.710 | 0.537 | 0.468 | 0.759 | 0.773 | 0.429 |
| Signal gate on digitized signal (no fine-tuning) | 0.859 (0.792-0.919) | 0.707 | 0.343 | nan | 0.532 | 0.612 | 0.082 |
| Signal gate on recorded signal (reference) | 0.946 (0.906-0.975) | 0.808 | 0.696 | 0.684 | 0.745 | 0.898 | 0.684 |

| Capture | Images | Digitized | Median r | AUROC image gate | AUROC signal gate on digitized | AUROC reference |
|---|---|---|---|---|---|---|
| colour scan | 20 | 100.0% | 0.441 | 0.740 | 0.706 | 0.818 |
| digital (green grid) | 10 | 100.0% | 0.177 | – | – | – |
| digital (red grid) | 10 | 100.0% | 0.778 | – | – | – |
| grayscale scan | 20 | 100.0% | 0.191 | 0.924 | 0.871 | 0.964 |
| phone photo | 10 | 80.0% | 0.340 | – | – | – |
| phone photo, stained | 10 | 100.0% | -0.004 | – | – | – |
| photo of a monitor | 10 | 100.0% | 0.336 | – | – | – |
| scan, stained | 10 | 100.0% | 0.119 | – | – | – |

Triage on the image gate: 58.6% of findings auto-decided, accuracy 0.930; 25.5% of images fully automatic.

Digitizer failures: found # rows of traces (1); could not follow the trace in row # (1)
