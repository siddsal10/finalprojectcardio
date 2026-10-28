"""Text of the BCSE497J Project-I report, as blocks that make_report.py lays out in the VIT template.

Every number comes from the paper (paper/tables, paper/scripts/*.json), RESULTS.md or the repository
itself. Markup: **bold**, *italic*, _{sub}, ^{sup}; {fig:key} / {tab:key} / {eq:key} / {sec:key} are
replaced by the report's own numbers. Tables and figures marked ("tex", label) are taken from the
paper's LaTeX sources so that their values cannot drift from the paper.
"""

TITLE = "CardioSentinel: A Multi-Model Federated Framework for ECG-Based Heart Disease Detection"
STUDENTS = [("23BDS0210", "Sanvi Saket"), ("23BDS0243", "Siddhi Salunkhe")]   # sorted on register number

ABSTRACT = [
    "Cardiovascular diseases are the leading cause of death worldwide, and the 12-lead electrocardiogram (ECG) "
    "is the first, cheapest and most widely available test for myocardial infarction, ischaemic ST/T changes, "
    "conduction disturbances and ventricular hypertrophy. Deep networks can read ECGs at a high level, but they "
    "are trained on data pooled at one site, whereas hospitals cannot share patient records. Federated learning "
    "removes the need to pool data, yet the common methods average model parameters over many communication "
    "rounds and therefore force every hospital to train the same network.",
    "This project builds CardioSentinel, a one-shot federated system for multi-label ECG classification. Four "
    "clients, standing for four hospitals, each train a structurally different network on their own, "
    "patient-separated share of the PTB-XL dataset: a 1D CNN, a 1D ResNet, a bidirectional LSTM and a CNN-LSTM. "
    "Each client calibrates its output with a single temperature and shares only five class probabilities per "
    "ECG. A server combines them with weights on the probability simplex. Equal weights, learned static weights "
    "and a per-patient dynamic gate are compared with patient-grouped cross-validation on the validation fold, "
    "and a more complex combiner is adopted only if it improves the validation macro AUROC by at least 0.002.",
    "On 21,388 PTB-XL records (80% training, 10% validation, 10% held-out test, three random seeds), the "
    "selected equal-weight gate reaches a test macro AUROC of 0.905, a macro F1 of 70.7% and a label-wise "
    "accuracy of 86.9%. It improves on the best single client in every seed (+0.009 to +0.012 AUROC, paired "
    "bootstrap) and comes within 0.004 AUROC of a model trained on all data pooled together and within 0.002 of "
    "FedAvg, which needs 20 communication rounds and a shared architecture. Integrated Gradients show which "
    "leads and waveform segments drove each finding, and Shapley values show how much each hospital contributes. "
    "A per-finding triage rule reports 53.7% of findings automatically at a precision of 0.937 and an accuracy "
    "of 0.967, and refers the rest to a cardiologist.",
    "The model is deployed in a clinician and patient web portal. Patients can upload a digital ECG or a photo, "
    "scan or PDF of a paper ECG, read a plain-language result, keep a symptom journal and decide what their "
    "doctor sees; clinicians review a queue of AI pre-reads, view the 12-lead tracing and sign off results. "
    "Photos are digitized with a U-Net trace segmenter; on 100 real fold-10 photos the photo pipeline reaches a "
    "macro AUROC of 0.841, which is below the triage target, so every photo finding is referred to a doctor. "
    "The project shows that most of the benefit of pooling ECG data can be obtained without exchanging data, "
    "parameters or architectures.",
]

ABBREVIATIONS = [
    ("1D", "One-Dimensional"), ("AdamW", "Adam with Decoupled Weight Decay"), ("AI", "Artificial Intelligence"),
    ("API", "Application Programming Interface"), ("AUPRC", "Area Under the Precision–Recall Curve"),
    ("AUROC", "Area Under the Receiver Operating Characteristic Curve"), ("BCE", "Binary Cross-Entropy"),
    ("BiLSTM", "Bidirectional Long Short-Term Memory"), ("CD", "Conduction Disturbance"),
    ("CI", "Confidence Interval"), ("CNN", "Convolutional Neural Network"), ("CPU", "Central Processing Unit"),
    ("CSV", "Comma-Separated Values"), ("CV", "Cross-Validation"), ("DFD", "Data Flow Diagram"),
    ("ECG", "Electrocardiogram"), ("F1", "Harmonic Mean of Precision and Recall"),
    ("FedAvg", "Federated Averaging"), ("FedNova", "Federated Normalized Averaging"),
    ("FedProx", "Federated Proximal Optimization"), ("FL", "Federated Learning"), ("HYP", "Hypertrophy"),
    ("IG", "Integrated Gradients"), ("KNN", "K-Nearest Neighbours"), ("LLM", "Large Language Model"),
    ("LSTM", "Long Short-Term Memory"), ("MI", "Myocardial Infarction"), ("MLP", "Multi-Layer Perceptron"),
    ("MPS", "Metal Performance Shaders"), ("NORM", "Normal ECG"), ("NPV", "Negative Predictive Value"),
    ("NPY", "NumPy Array File"), ("PDF", "Portable Document Format"),
    ("PTB-XL", "Physikalisch-Technische Bundesanstalt Extra-Large ECG Dataset"), ("ReLU", "Rectified Linear Unit"),
    ("ResNet", "Residual Network"), ("ROC", "Receiver Operating Characteristic"),
    ("SCAFFOLD", "Stochastic Controlled Averaging for Federated Learning"),
    ("SCP-ECG", "Standard Communications Protocol for Computer-Assisted Electrocardiography"),
    ("SD", "Standard Deviation"), ("SNR", "Signal-to-Noise Ratio"), ("SPA", "Single-Page Application"),
    ("SQL", "Structured Query Language"), ("STTC", "ST/T Change"), ("SVM", "Support Vector Machine"),
    ("WFDB", "Waveform Database (PhysioNet format)"), ("XAI", "Explainable Artificial Intelligence"),
    ("XGBoost", "Extreme Gradient Boosting"),
]

SYMBOLS = [
    ("x", "Input ECG, 12 leads × 1,000 samples (10 s at 100 Hz)"),
    ("y_{c}", "Label of superclass c ∈ {NORM, MI, STTC, CD, HYP} (multi-hot)"),
    ("K", "Number of clients (K = 4: A, B, C, D)"),
    ("P_{i}(y | x)", "Calibrated class probabilities of client i"),
    ("P(y | x)", "Global (server) probabilities after aggregation"),
    ("w_{i}", "Aggregation weight of client i, w_{i} ≥ 0, Σ w_{i} = 1"),
    ("T_{i}", "Temperature of client i (fitted on fold 9)"),
    ("z_{ic}", "Logit of client i for class c"),
    ("σ(·)", "Logistic sigmoid"),
    ("β_{c}", "Positive-class weight N_{c}^{−} / N_{c}^{+} in the loss"),
    ("τ_{c}", "Decision threshold of class c (maximises F1 on fold 9)"),
    ("l_{c}, h_{c}", "Rule-out and rule-in thresholds of the triage band of class c"),
    ("H_{i}", "Mean binary entropy of client i (input of the dynamic gate)"),
    ("IG_{ℓt}", "Integrated-Gradients attribution of lead ℓ at time t"),
    ("φ_{i}", "Shapley value of client i for the macro AUROC"),
    ("v(S)", "Macro AUROC of the equal-weight sub-federation S"),
]


def chapters():
    B = []
    add = B.append

    # ------------------------------------------------------------------ 1
    add(("chapter", "Introduction"))
    add(("h2", "Background"))
    add(("p", "Cardiovascular diseases remain the leading cause of death worldwide, and the Global Burden of Disease "
              "study estimated that their prevalence almost doubled between 1990 and 2019 [1]. The 12-lead "
              "electrocardiogram (ECG) is the first-line, low-cost and non-invasive test for most of these conditions. "
              "It is recorded in primary care, emergency departments and wards, and a single ten-second recording can "
              "reveal myocardial infarction (MI), ischaemic ST/T changes (STTC), conduction disturbances (CD) and "
              "ventricular hypertrophy (HYP). Because a large fraction of ECGs is acquired where no cardiologist is "
              "present, reliable automatic interpretation directly affects how quickly a patient with an acute finding "
              "is identified and referred."))
    add(("p", "In current practice the ECG is read visually by a physician, often supported by rule-based software "
              "built into the electrocardiograph. Visual reading depends on training and workload, and the built-in "
              "algorithms make frequent errors, especially for non-sinus rhythms and subtle morphological changes, so "
              "their statements have to be over-read by an expert [2]. A single ECG can also carry several co-existing "
              "diagnoses, which makes the task multi-label rather than a choice between mutually exclusive classes."))
    add(("p", "Deep learning has substantially changed automatic ECG analysis. Convolutional and recurrent networks have "
              "reached cardiologist-level performance on single-lead ambulatory recordings [3] and on large 12-lead "
              "cohorts [4], and the public PTB-XL dataset [5] has enabled reproducible benchmarks in which ResNet, "
              "Inception and LSTM models reach a macro area under the ROC curve (AUROC) of 0.921–0.930 on the "
              "five-superclass task [6]. Later work added self-supervised pre-training [7], [8], state space models "
              "and patient metadata [9], and ECG–text multimodal learning [10], [11]. Almost all of these models are, "
              "however, trained on data pooled at one site."))
    add(("h2", "Motivations"))
    add(("p", "ECG archives are held by separate hospitals that cannot share patient records. Federated learning (FL) "
              "[12], [13] lets institutions train together without moving data, but the usual methods – FedAvg [14], "
              "FedProx [15] and SCAFFOLD [16] – average model parameters. Every hospital must therefore train the same "
              "network and take part in many synchronised communication rounds. In practice, hospitals have different "
              "computing resources and in-house models, and repeated synchronisation is difficult to organise."))
    add(("p", "Clinical use also requires more than a high AUROC. A clinician needs to see why the model reached a "
              "finding, and the system must state explicitly when it should not decide on its own. Most ECG studies "
              "provide neither [17]. These three observations – data that cannot be pooled, institutions that want "
              "their own architectures, and the need for explanations and safe referral – motivate this project."))
    add(("p", "The project asks a deliberately conservative question: if each hospital trains its own network once and "
              "shares only calibrated class probabilities, is the combined model better than the best single hospital, "
              "and how close does it come to a model trained on all data pooled together?"))
    add(("h2", "Scope of the Project"))
    add(("p", "**In scope.** The project covers:"))
    add(("bullets", [
        "multi-label classification of 10-s, 12-lead ECGs (100 Hz) into the five PTB-XL diagnostic superclasses "
        "NORM, MI, STTC, CD and HYP;",
        "four simulated institutions (clients) that hold disjoint, patient-separated shares of the PTB-XL training "
        "folds and train four different architectures;",
        "one-shot, prediction-level aggregation with equal, static and dynamic weights, selected on validation data;",
        "calibration, per-class thresholds and a per-finding triage rule that refers uncertain findings;",
        "explanations with Integrated Gradients (leads and time) and Shapley values (clients);",
        "comparison with traditional machine learning, a centrally trained model and FedAvg, FedProx, SCAFFOLD and "
        "FedNova;",
        "the CardioSentinel web portal for patients and clinicians, including reading photos, scans and PDFs of "
        "paper ECGs.",
    ]))
    add(("p", "**Out of scope.** Deployment across real hospitals, formal privacy guarantees such as differential "
              "privacy, rhythm and form statements beyond the five superclasses, regression targets such as severity "
              "scores, and clinical validation with cardiologists. The portal is a research prototype and not a "
              "medical device."))

    # ------------------------------------------------------------------ 2
    add(("chapter", "Project Description and Goals"))
    add(("h2", "Literature Review"))
    add(("p", "This section reviews peer-reviewed studies published between 2022 and 2026 that are closest to the "
              "project. They are grouped into unimodal deep-learning approaches for the ECG, multimodal approaches, "
              "federated-learning approaches and explainable approaches. The reported figures are those stated by the "
              "authors on their own datasets and splits, so they are not directly comparable with each other. "
              "{tab:related} summarises the review."))
    add(("h3", "Unimodal Deep-Learning Approaches"))
    add(("p", "Mehari and Strodthoff (2022) [7] adapted contrastive predictive coding, SimCLR, BYOL and SwAV to 12-lead "
              "ECGs from PTB-XL and reached a macro AUROC of 94.18% on the 71-statement task after self-supervised "
              "pre-training and fine-tuning, but the gain over supervised training was about one percentage point and "
              "pre-training relied on pooling several large corpora at one site. Na et al. (2024) [8] proposed ST-MEM, "
              "a spatio-temporal masked auto-encoder, and reached 82.5% accuracy and a macro AUROC of 93.3% on the five "
              "superclasses, but the method needs large-scale centralised pre-training and does not explain its "
              "decisions. Mehari and Strodthoff (2023) [9] applied structured state space (S4) models to the raw "
              "signal and reached a macro AUROC of 94.17% on the 71-statement task, again on a single pooled dataset. "
              "These studies set strong single-site baselines, but each assumes that all records can be gathered in "
              "one place."))
    add(("h3", "Multimodal Approaches"))
    add(("p", "Mehari and Strodthoff (2023) [9] also fused demographic metadata with the S4 representation and reached "
              "94.63% macro AUROC, only 0.18 percentage points more. Li et al. (2023) [10] proposed METS, which aligns "
              "ECGs with generated reports through a frozen language model and reached 84.2% zero-shot accuracy on the "
              "superclass task, but its macro F1 of 65.7% stayed below the 77.6% of a supervised ResNet-18. Liu et al. "
              "(2024) [11] proposed MERL, ECG–report pre-training with test-time clinical-knowledge prompts, and reached "
              "an average zero-shot AUROC of 75.2% on six datasets, but it needs a free-text report for every training "
              "ECG. Palakayala et al. (2025) [18] fused MRI, gait and speech for Parkinson's disease with hierarchical "
              "attention (94.2% accuracy), but with modalities from different cohorts and centralised training. "
              "Additional modalities can help, but every modality must be available for each patient at one site."))
    add(("h3", "Federated-Learning Approaches"))
    add(("p", "Raza et al. (2022) [19] combined federated transfer learning with an explanation module on MIT-BIH beats "
              "(98.9% accuracy on clean signals), but evaluated beats of a 48-record database rather than patient-level "
              "12-lead diagnosis. Goto et al. (2022) [20] trained ECG and echocardiogram models across hospitals in the "
              "United States and Japan (AUROC 90%–96% internally, 93% externally), with the same architecture at every "
              "site and a single binary task. Quan et al. (2025) [21] proposed FedOcw, which optimises client-specific "
              "aggregation weights every round and outperformed FedAvg, FedProx, SCAFFOLD and FedNova (74.81%), but "
              "still averages parameters of one network over 100 rounds. Sathi et al. (2025) [22] proposed explainable "
              "FL on ECG fiducial features with SHAP and LIME (92.3%), with one model family on all clients. In all of "
              "these studies the clients train the same model and communicate repeatedly."))
    add(("h3", "Explainable AI Approaches"))
    add(("p", "Jahmunah et al. (2022) [23] used DenseNet and CNN models with Grad-CAM on the PTB database to detect MI "
              "(accuracy above 95%), on a small cohort with qualitative heat maps. Wagner et al. (2024) [17] proposed "
              "building blocks for auditing ECG models on PTB-XL and found that saliency was the only attribution "
              "method that passed all their sanity checks, without proposing a new classifier. Pantelidis et al. (2025) "
              "[24] proposed ECG-XPLAIM, an inception-style 1D CNN with Grad-CAM trained on MIMIC-IV and validated on "
              "PTB-XL (98% macro AUROC for conduction disturbances), but its long-QT sensitivity was only 69.1%. "
              "Explainability is increasingly expected, but it is rarely combined with privacy-preserving training or "
              "with a rule that decides which predictions should be referred."))
    add(("table", "tab:related", ("tex", "tab:related")))
    add(("h2", "Research Gap"))
    add(("p", "The review shows three gaps:"))
    add(("numbered", [
        "**Pooling.** Single-site ECG models reach a high AUROC, but they assume that all records can be pooled.",
        "**Homogeneity and rounds.** Federated methods remove that assumption, but they average parameters, so every "
        "institution must adopt the same architecture and take part in many synchronised rounds.",
        "**Explanation and referral.** Explainable models are seldom combined with privacy-preserving training or "
        "with an explicit rule for deferring uncertain findings to a clinician.",
    ]))
    add(("p", "CardioSentinel addresses these gaps by letting each institution train its own architecture once, "
              "combining only calibrated probabilities through a validated simplex gate, explaining the combined "
              "prediction on the raw signal, and reporting automatically only those findings for which the validated "
              "precision and recall reach the clinical target."))
    add(("h2", "Objectives"))
    add(("numbered", [
        "Build four heterogeneous ECG classifiers (1D CNN, 1D ResNet, BiLSTM, CNN-LSTM), each trained only on its own "
        "patient-separated share of PTB-XL.",
        "Combine their calibrated probabilities with equal, static and per-patient dynamic weights on the probability "
        "simplex, and select the combiner on validation data with a parsimony rule fixed in advance.",
        "Test, once and on held-out data over three seeds, whether the federated model beats the best single client, "
        "and measure how close it comes to centralised training and to FedAvg, FedProx, SCAFFOLD and FedNova.",
        "Quantify the contribution of each client with subset ablations and Shapley values, and explain predictions "
        "with Integrated Gradients.",
        "Design a per-finding triage rule that reports findings automatically only at precision, recall and accuracy "
        "≥ 0.90 and refers the rest.",
        "Deploy the model in a patient and clinician portal that also reads photos, scans and PDFs of paper ECGs.",
    ]))
    add(("h2", "Problem Statement"))
    add(("p", "Given a 10-s, 12-lead ECG x ∈ ℝ^{12×1000}, predict the multi-hot label vector y ∈ {0, 1}^{5} over "
              "NORM, MI, STTC, CD and HYP. The training data are split among K = 4 institutions that may not exchange "
              "records or model parameters and that may each use a different network. The server may only receive "
              "each client's predicted class probabilities P_{i}(y | x). The task is to find weights w_{i} ≥ 0 with "
              "Σ_{i} w_{i} = 1 such that P(y | x) = Σ_{i} w_{i} P_{i}(y | x) is better than the best individual client "
              "on unseen patients, with every modelling choice made without looking at the test data, and to decide "
              "for each finding whether it can be reported automatically or must be referred."))
    add(("h2", "Project Plan"))
    add(("p", "The work was carried out from July to 13 October 2026. {fig:plan} shows the timeline and {tab:plan} the "
              "milestones. After the literature review and data preparation, an earlier three-modality prototype (ECG, "
              "clinical text and images, with other datasets) was built and then replaced by the four-client design "
              "described in this report."))
    add(("fig", "fig:plan", "diagram:plan", "Project timeline (July to October 2026)", 6.2))
    add(("table", "tab:plan", ("rows", ["Date (2026)", "Milestone"], [
        ["1–15 Jul", "Literature review and problem definition"],
        ["16–31 Jul", "PTB-XL downloaded and verified (21,799 records, 18,869 patients, folds 1–10); pre-processing "
                      "pipeline"],
        ["1–20 Aug", "Earlier three-modality prototype"],
        ["21–31 Aug", "Redesign: four clients with different architectures and a weighted gate"],
        ["1–20 Sep", "Clients A–D trained over seeds 42–44; aggregation and the ≥ 0.002 selection rule"],
        ["21–30 Sep", "All-data baseline, leave-one-client-out, traditional ML and FL baselines; per-finding triage"],
        ["1–7 Oct", "CardioSentinel web portal, photo / scan / PDF reading, voice assistant; Integrated Gradients"],
        ["8–13 Oct", "Paper and this report"],
]), "Project milestones"))

    # ------------------------------------------------------------------ 3
    add(("chapter", "Technical Specification"))
    add(("h2", "Requirements"))
    add(("h3", "Functional"))
    add(("table", "tab:fr", ("rows", ["ID", "Requirement"], [
        ["FR1", "Prepare PTB-XL v1.0.3: map SCP codes to the five superclasses, drop unlabelled records, band-pass "
                "filter and store the signals (python -m cardiomoe.data.ptbxl --prepare)."],
        ["FR2", "Train four clients on their own folds (A 1–2, B 3–4, C 5–6, D 7–8) with early stopping on fold 9."],
        ["FR3", "Fit one temperature per client and per-class decision thresholds on fold 9."],
        ["FR4", "Fit equal, static and dynamic combiners by patient-grouped 5-fold CV on fold 9 and select one with "
                "the ≥ 0.002 macro-AUROC rule."],
        ["FR5", "Evaluate every client and combiner once on fold 10 (AUROC with 95% bootstrap CI, AUPRC, precision, "
                "recall, specificity, F1, accuracy, confusion matrices, paired bootstrap)."],
        ["FR6", "Fit per-class triage bands on fold 9 and report coverage, precision, recall and accuracy on fold 10."],
        ["FR7", "Explain predictions with Integrated Gradients and compute Shapley values over client subsets."],
        ["FR8", "Portal: sign-up and sign-in for patients and clinicians with separated roles."],
        ["FR9", "Portal: analyse an uploaded ECG (CSV, NPY, WFDB) or up to four photos / scans / PDFs of one paper ECG "
                "and show the verdict, the five findings and the rhythm strip."],
        ["FR10", "Portal: symptom journal, sharing with the doctor (nothing is shared until the patient presses "
                 "Send), messages, appointments and a voice assistant that answers from the patient's own records."],
        ["FR11", "Portal: clinician queue of ECGs to review, 12-lead viewer, sign-off and release, replies to shares."],
    ]), "Functional requirements"))
    add(("h3", "Non-Functional"))
    add(("table", "tab:nfr", ("rows", ["ID", "Requirement"], [
        ["NFR1 Privacy", "Clients never share data or parameters; the server sees only five probabilities per client "
                         "and ECG. Patient self-checks stay private until shared."],
        ["NFR2 Validity", "Patient-level splits; fold 10 is used once (evaluate.py refuses a second run without "
                          "--rerun); nothing is tuned on it."],
        ["NFR3 Reproducibility", "All randomness seeded (seed 42); final results over seeds 42, 43, 44; every script "
                                 "supports --debug (about 200 samples, CPU, a few minutes)."],
        ["NFR4 Safety", "Findings outside the validated triage bands, and all photo findings, are referred to a "
                        "doctor; every screen states that the result is an AI pre-read and not a diagnosis."],
        ["NFR5 Efficiency", "Training and inference on an 8-GB laptop without a discrete GPU."],
        ["NFR6 Usability", "Plain-language results for patients; standard 25 mm/s, 10 mm/mV ECG grid for clinicians."],
        ["NFR7 Maintainability", "Modular Python package (cardiomoe), configuration in YAML, 37 automated tests."],
    ]), "Non-functional requirements"))
    add(("h2", "Feasibility Study"))
    add(("h3", "Technical Feasibility"))
    add(("p", "The complete pipeline – data preparation, training of four clients over three seeds, aggregation, "
              "evaluation, triage, explanations and the portal – was run on an Apple M1 MacBook Air with 8 GB of unified "
              "memory using PyTorch on the Metal (MPS) backend. A training epoch took 26.8 s for Client A, 38.5 s for "
              "Client B, 65.4 s for Client C and 28.2 s for Client D. Integrated Gradients needed the CPU, because MPS "
              "ran out of memory on 8 GB, and took about 47 s per batch of 16 records. The models are small "
              "(0.44M–6.73M parameters), so inference in the portal runs on the same machine. The project is therefore "
              "technically feasible with commodity hardware."))
    add(("h3", "Economic Feasibility"))
    add(("p", "All data are open access: PTB-XL v1.0.3 (CC BY 4.0) from PhysioNet and the ECG-Image-Database "
              "(CC BY-ND 4.0) from Kaggle. All software is open source (Python, PyTorch, scikit-learn, FastAPI, SQLite, "
              "Tectonic). No paid cloud computing or GPU was used. The only optional running cost is the large language "
              "model behind the voice assistant (OpenRouter, pay per use); without a key the assistant falls back to "
              "offline answers. The licence cost of the project is therefore zero (Section {sec:cost})."))
    add(("h3", "Social Feasibility"))
    add(("p", "Automatic ECG pre-reading can shorten the time to referral where no cardiologist is present, and the "
              "federated design lets hospitals collaborate without moving patient records. The portal is patient-first: "
              "patients decide what their doctor sees, uncertain findings are referred rather than guessed, and every "
              "result is labelled as an AI pre-read. The main social risks – over-reliance on the model and unequal "
              "performance across populations – are addressed by the triage rule, by the explicit referral messages and "
              "by stating the limits of a single-country dataset."))
    add(("h2", "System Specification"))
    add(("h3", "Hardware Specification"))
    add(("table", "tab:hw", ("rows", ["Component", "Specification used"], [
        ["Machine", "Apple M1 MacBook Air"],
        ["Memory", "8 GB unified memory"],
        ["Accelerator", "Apple GPU via Metal Performance Shaders (MPS); CPU for Integrated Gradients"],
        ["Storage", "PTB-XL records100 0.55 GB; ECG-Image-Database fold 10 about 2.5 GB (all 977 D0 records "
                    "about 22 GB)"],
        ["Client device", "Any modern browser (desktop or phone) for the portal"],
    ]), "Hardware specification"))
    add(("h3", "Software Specification"))
    add(("table", "tab:sw", ("rows", ["Category", "Software"], [
        ["Operating system", "macOS"],
        ["Language", "Python 3.11.17"],
        ["Deep learning", "PyTorch 2.14.1 (torch ≥ 2.2)"],
        ["Data and ML", "NumPy, pandas, SciPy, scikit-learn, XGBoost, PyArrow, wfdb ≥ 4.1"],
        ["Images", "PyMuPDF ≥ 2.0, OpenCV (headless) ≥ 5.0, Pillow"],
        ["Plots", "Matplotlib ≥ 3.8"],
        ["Web", "FastAPI ≥ 0.110, Uvicorn ≥ 0.29, python-multipart, httpx; SQLite; HTML, CSS and JavaScript SPA"],
        ["Testing", "pytest ≥ 8.0"],
        ["Assistant (optional)", "OpenRouter API (OPENROUTER_API_KEY), offline fallback"],
        ["Documents", "Tectonic (LaTeX), pandoc"],
    ]), "Software specification"))

    # ------------------------------------------------------------------ 4
    add(("chapter", "Design Approach and Details"))
    add(("h2", "System Architecture"))
    add(("p", "{fig:arch} shows the architecture. The data block at the bottom holds the PTB-XL records, split by the "
              "official stratified folds into four client partitions, a validation fold held by the server and a test "
              "fold. The pre-processing block applies the same signal pipeline at every client. Each client learns its "
              "own features with its own architecture and ends in a classifier head (C) that outputs five logits, which "
              "are converted into calibrated probabilities with a per-client temperature. The federated server combines "
              "the four probability vectors with simplex weights into the global output, the classification block turns "
              "it into five binary decisions and triage decisions, and the explainability block attributes the output "
              "to leads and time."))
    add(("fig", "fig:arch", "paperpng:fig1", "Architecture of the proposed CardioSentinel model", 6.2))
    add(("p", "{fig:flow} shows the federated workflow. Each client trains once on its private data and fits its "
              "temperature (step 1). Afterwards the server sends an ECG to the clients and receives calibrated "
              "probabilities (step 2), and mixes them (step 3). No parameters or raw training data are exchanged; at "
              "query time 20 numbers per ECG reach the server."))
    add(("fig", "fig:flow", "paperpng:fig3", "Federated workflow: clients train once on private data; the server "
                                             "queries them and receives only calibrated probabilities", 5.0))
    add(("p", "The deployed system adds the CardioSentinel portal on top of this model. A FastAPI server exposes a REST "
              "API (for example POST /api/me/analyze for patients and POST /api/analyze for clinicians), stores users, "
              "ECGs, results, journal entries, messages and audit records in SQLite, and serves a single-page web "
              "application. The inference Engine loads the four seed-42 clients, the selected gate and the triage bands; "
              "photos, scans and PDFs are first converted to a 12 × 1,000 signal by the digitizer."))
    add(("h2", "Design"))
    add(("h3", "Data Flow Diagram"))
    add(("p", "{fig:dfd0} is the context diagram (level 0). The system receives signals and labels from PhysioNet and "
              "ECG photos from Kaggle, receives ECG files, photos and symptoms from patients, and exchanges AI pre-reads "
              "and sign-offs with clinicians. {fig:dfd1} refines it into five processes and three data stores."))
    add(("fig", "fig:dfd0", "diagram:dfd0", "Data flow diagram, level 0 (context)", 5.6))
    add(("fig", "fig:dfd1", "diagram:dfd1", "Data flow diagram, level 1", 6.2))
    add(("h3", "Use Case Diagram"))
    add(("p", "{fig:usecase} shows the two actors of the portal. Patients sign up, check their heart, read results, keep "
              "a symptom journal, share with their doctor, ask the voice assistant and book appointments. Clinicians "
              "review the queue of ECGs, view the 12-lead tracing, sign off and release results, reply to shares and add "
              "ECGs for patients. Both analysis use cases include the AI pre-read."))
    add(("fig", "fig:usecase", "diagram:usecase", "Use case diagram of the CardioSentinel portal", 5.6))
    add(("h3", "Class Diagram"))
    add(("p", "{fig:classes} shows the main classes. SignalClient (src/cardiomoe/fourclient/models.py) extends "
              "BaseClient, which holds the temperature buffer, and wraps one of four backbones: CNN1D, ResNet1D (made of "
              "ResBlock1d), BiLSTM or CNNLSTM. The three combiners in gate.py share the interface fit(), weights() and "
              "predict(); DynamicGate contains a GateNet MLP. The web Engine uses four SignalClients, the selected "
              "combiner and the Triage rule."))
    add(("fig", "fig:classes", "diagram:classes", "Class diagram (main classes of the model and the portal)", 6.2))
    add(("h3", "Sequence Diagram"))
    add(("p", "{fig:sequence} shows a patient's heart check. The browser posts the file to /api/me/analyze. A photo, "
              "scan or PDF is first digitized; a CSV, NPY or WFDB file is parsed directly. The Engine runs the four "
              "clients, mixes their calibrated probabilities with the gate weights and applies the triage rule. The "
              "result is stored privately in SQLite and returned to the browser."))
    add(("fig", "fig:sequence", "diagram:sequence", "Sequence diagram of a patient heart check", 6.2))

    # ------------------------------------------------------------------ 5
    add(("chapter", "Methodology and Testing"))
    add(("p", "{fig:alg1} summarises the complete method, from pre-processing to the explained, triaged prediction, "
              "and the following sections describe each module."))
    add(("fig", "fig:alg1", "paperpng:algorithm1", "Algorithm of the complete CardioSentinel method", 6.0))
    add(("h2", "Module Description"))
    add(("h3", "Data Preparation"))
    add(("p", "bash scripts/download.sh downloads PTB-XL records100 (100 Hz) from PhysioNet's open-data mirror and "
              "writes a verified manifest (21,799 records, 18,869 patients). python -m cardiomoe.data.ptbxl --prepare "
              "maps each record's SCP-ECG statements to the diagnostic_class column of scp_statements.csv, keeping only "
              "the 44 statements flagged as diagnostic, and encodes them as a multi-hot vector over NORM, MI, STTC, CD "
              "and HYP. Records without a diagnostic superclass cannot be labelled and are removed (21,799 → 21,388); no "
              "other record is excluded. The official stratified folds are kept unchanged, so all recordings of a "
              "patient lie in one fold."))
    add(("h3", "Pre-processing and Augmentation"))
    add(("p", "{tab:prep} lists all pre-processing steps. They are implemented once and run identically by every client, "
              "so no client-specific statistics are needed and no information crosses client boundaries. The waveforms "
              "are read with WFDB, reordered to I, II, III, aVR, aVL, aVF, V1–V6, cleaned of non-finite values, cropped "
              "or padded to 1,000 samples and band-pass filtered (0.5–40 Hz, third-order Butterworth, forwards and "
              "backwards, so without phase shift). At load time each lead of each recording is z-scored with its own "
              "statistics and clipped:"))
    add(("eq", "eq:norm", r"x^{\mathrm{norm}}_{\ell t}=\operatorname{clip}\!\left(\frac{x_{\ell t}-\mu_\ell}{\sigma_\ell},\,-10,\,10\right)"))
    add(("p", "Because the statistics are per recording, normalisation needs no shared population statistics, which is "
              "essential in the federated setting. During training only, the signal is scaled by a random factor in "
              "[0.9, 1.1], Gaussian noise at an SNR of 20–30 dB and a 0.1–0.5 Hz baseline wander are added, and each "
              "lead is zeroed with probability 0.05."))
    add(("table", "tab:prep", ("tex", "tab:prep")))
    add(("h3", "Client Models"))
    add(("p", "The four clients span the main families of ECG networks, from purely convolutional to purely recurrent, "
              "so that their errors are as diverse as possible ({tab:features}). Client A is a plain 1D CNN with four "
              "blocks of a 7-tap convolution, batch normalisation, ReLU and max-pooling (32–256 channels). Client B is a "
              "1D ResNet [25] with a 64-channel stem and stages of 64, 128, 256 and 256 channels with 3, 4, 6 and 3 "
              "residual blocks. Client C is a two-layer bidirectional LSTM [26], [27] with 128 hidden units per "
              "direction that reads all 1,000 time steps. Client D is a CNN-LSTM in which three convolutional blocks "
              "shorten the sequence to 125 steps before a one-layer BiLSTM. All clients summarise the time axis with "
              "concatenated average and max pooling and share the head Linear(d → 256) → ReLU → Dropout(0.3) → "
              "Linear(256 → 5)."))
    add(("table", "tab:features", ("tex", "tab:features")))
    add(("h3", "Local Training and Calibration"))
    add(("p", "Each client minimises the class-weighted binary cross-entropy"))
    add(("eq", "eq:loss", r"\mathcal{L}_i=-\frac{1}{5N}\sum_{n=1}^{N}\sum_{c=1}^{5}\Big[\beta_c\,y_{nc}\log\sigma(z_{nc})+(1-y_{nc})\log\big(1-\sigma(z_{nc})\big)\Big]"))
    add(("p", "with β_{c} = N_{c}^{−} / N_{c}^{+} computed from the client's own training labels, using AdamW (learning "
              "rate 10^{−3}, weight decay 10^{−2}), a OneCycle schedule, batch size 128, gradient clipping at 1.0 and up "
              "to 50 epochs with early stopping on the fold-9 macro AUROC (patience 8). After training, a single "
              "temperature T_{i} is fitted on fold 9 with LBFGS [28], which rescales the confidence without changing "
              "the ranking:"))
    add(("eq", "eq:temp", r"P_i(y_c\mid x)=\sigma\!\left(z_{ic}(x)/T_i\right)"))
    add(("p", "{tab:hyper} lists all hyperparameters; they are identical for every client so that only the architecture "
              "and the data differ."))
    add(("table", "tab:hyper", ("tex", "tab:hyper")))
    add(("h3", "Federated Aggregation"))
    add(("p", "The server aggregates the calibrated predictions,"))
    add(("eq", "eq:gate", r"P(y\mid x)=\sum_{i=1}^{K}w_i(x)\,P_i(y\mid x),\qquad w_i(x)\ge 0,\quad \sum_{i=1}^{K}w_i(x)=1"))
    add(("p", "with the weights parameterised by a softmax so that the simplex constraint holds by construction. "
              "**Equal weights** use w_{i} = 1/K. **Static weights** learn one logit per client by minimising the BCE of "
              "P(y | x) on validation data (Adam, learning rate 0.05, 500 steps). The **dynamic gate** is a per-patient "
              "mixture of experts [32]: an MLP takes the 20 client probabilities and the four mean binary entropies"))
    add(("eq", "eq:entropy", r"H_i=-\frac{1}{5}\sum_{c}\big[p_{ic}\log p_{ic}+(1-p_{ic})\log(1-p_{ic})\big]"))
    add(("p", "passes them through a 32-unit ReLU layer with dropout 0.2 and outputs w(x) by a softmax; its last layer "
              "starts at zero, i.e. from equal weights. All three are evaluated by patient-grouped five-fold "
              "cross-validation inside fold 9, and a more complex combiner is adopted only if its cross-validated macro "
              "AUROC exceeds that of the current choice by at least 0.002 ({fig:alg2}). This rule was approved on 18 "
              "October 2026, before any three-seed test evaluation."))
    add(("fig", "fig:alg2", "paperpng:algorithm2", "One-shot federated protocol and gate selection", 6.0))
    add(("h3", "Per-Finding Triage"))
    add(("p", "When the clinical requirement is a precision and recall of at least 0.90, no single threshold can satisfy "
              "it for every class. For each class c a rule-out threshold l_{c} and a rule-in threshold h_{c} are "
              "therefore fitted on the selected combiner's cross-validated fold-9 predictions:"))
    add(("eq", "eq:triage", r"\hat y_c=\begin{cases}1 \;(\text{rule in}) & P(y_c\mid x)\ge h_c\\ 0 \;(\text{rule out}) & P(y_c\mid x)\le l_c\\ \text{refer to a cardiologist} & \text{otherwise}\end{cases}"))
    add(("p", "Each band is the widest one with precision, recall and accuracy ≥ 0.92 (target + 0.02) and with at least "
              "10% of the class's positives decided automatically (mode 'automated'). Otherwise the class is 'rule-out' "
              "only (negatives with NPV ≥ 0.95, ruling out ≤ 5% of positives) or 'review'. The frozen bands were applied "
              "once to the test fold."))
    add(("h3", "Explainability"))
    add(("p", "Because the equal-weight output F_{c}(x) = (1/K) Σ_{i} σ(z_{ic}(x)/T_{i}) is differentiable, it is "
              "explained with Integrated Gradients [33] with an all-zero baseline x′ and a 32-step Riemann sum:"))
    add(("eq", "eq:ig", r"\mathrm{IG}_{\ell t}(x)=(x_{\ell t}-x'_{\ell t})\int_0^1\frac{\partial F_c\big(x'+\alpha(x-x')\big)}{\partial x_{\ell t}}\,d\alpha"))
    add(("p", "Global lead importance is the share of Σ_{t} |IG_{ℓt}| falling on each lead; a deletion test checks "
              "faithfulness by zeroing the most-attributed leads. At the federation level, the contribution of each "
              "client to the macro AUROC v(S) of an equal-weight sub-federation S is its Shapley value [36], computed "
              "exactly over the 15 non-empty subsets with v(∅) = 0.5:"))
    add(("eq", "eq:shapley", r"\phi_i=\sum_{S\subseteq\mathcal{K}\setminus\{i\}}\frac{|S|!\,(K-|S|-1)!}{K!}\big[v(S\cup\{i\})-v(S)\big]"))
    add(("h3", "Evaluation Metrics"))
    add(("p", "Every metric is computed per class from the confusion counts TP_{c}, FP_{c}, FN_{c} and TN_{c} on the N "
              "test records and then macro-averaged over the five classes:"))
    add(("eq", "eq:prf", r"\mathrm{Precision}_c=\frac{TP_c}{TP_c+FP_c},\qquad \mathrm{Recall}_c=\frac{TP_c}{TP_c+FN_c},\qquad \mathrm{F1}_c=\frac{2\,\mathrm{Precision}_c\,\mathrm{Recall}_c}{\mathrm{Precision}_c+\mathrm{Recall}_c}"))
    add(("eq", "eq:acc", r"\mathrm{Acc}_{\mathrm{label}}=\frac{1}{5}\sum_{c=1}^{5}\frac{TP_c+TN_c}{N},\qquad \mathrm{Acc}_{\mathrm{exact}}=\frac{1}{N}\sum_{n=1}^{N}\big[\mathbf{y}^{\mathrm{pred}}_n=\mathbf{y}_n\big]"))
    add(("p", "Macro AUROC is the primary metric, because it does not depend on thresholds or class prevalence; its 95% "
              "confidence interval uses 1,000 bootstrap resamples of the test records [41], and differences between two "
              "models use a paired bootstrap on the same resamples."))
    add(("h3", "Printed-ECG Reading (Photos, Scans and PDFs)"))
    add(("p", "To accept pictures of paper ECGs, the project added ECG-Image-Database (Kaggle physionet/ecg-image-database, "
              "CC BY-ND 4.0), PTB-XL part only: real printouts of PTB-XL records that were scanned, photographed with "
              "phones, photographed off monitors, stained and creased. Folds and the test split are unchanged. "
              "synth.py draws synthetic printouts with a pixel-exact trace mask; segnet.py is a half-resolution U-Net "
              "trace segmenter trained only on those (4,000 steps on the CPU, validation Dice 0.853); digitize.py "
              "straightens the page, tracks the row baselines, follows the one-pixel centre line of each trace "
              "(Zhang–Suen thinning) and outputs the 12 × 1,000 signal with a quality score. finetune.py fine-tunes each "
              "client on the real photos of its own folds (13 per training fold, 104 in total), with early stopping, "
              "temperature, gate and triage on 60 fold-9 photos."))
    add(("h3", "Web Portal and Voice Assistant"))
    add(("p", "The CardioSentinel portal (web/) is a FastAPI back end with a static single-page front end. The patient "
              "app offers a home page with the latest heart check, 'Check my heart' (upload or example recordings), a "
              "plain-language result with the five findings and the rhythm strip, a symptom journal, 'Share with my "
              "doctor', appointments, a heart-health guide and a voice assistant. The doctor workspace offers an inbox of "
              "patient shares, a queue of ECGs to review, a 12-lead viewer with sign-off, patients, messages and "
              "appointments. The assistant (web/assistant.py) retrieves passages from the patient's own records with "
              "TF-IDF and asks a language model through OpenRouter to answer only from them, with an offline extractive "
              "fallback. Self-checks are private: a doctor sees a check, note or file only after the patient shares it."))
    add(("h2", "Testing"))
    add(("h3", "Automated Tests"))
    add(("p", "The repository contains 37 automated pytest tests in seven files ({tab:tests}). They were run for this "
              "report with python -m pytest on Python 3.11.17: **37 passed** in 9.9 s."))
    add(("table", "tab:tests", ("rows", ["Test file", "Tests", "What is checked", "Result"], [
        ["test_splits.py", "3", "No patient in two splits; official folds 1–10; every record has a label", "Pass"],
        ["test_labels.py", "1", "SCP code → superclass mapping", "Pass"],
        ["test_signal.py", "4", "500 → 100 Hz shape; case-insensitive lead order; flat-lead normalisation and "
                               "clipping; augmentation shapes", "Pass"],
        ["test_features.py", "3", "Prepared metadata carries only allow-listed features", "Pass"],
        ["test_fourclient.py", "11", "Client partitions disjoint and patient-separated; output shape of all four "
                                    "models; weights on the simplex for all three combiners; static weights find the "
                                    "informative client; metrics; triage bands meet targets and modes", "Pass"],
        ["test_images.py", "9", "Layout masks; masked normalisation; page mask; one-pixel thinning; digitizer "
                               "recovers a known signal (3×4 and 12×1); blank page rejected; transparency; U-Net "
                               "output size", "Pass"],
        ["test_web.py", "5", "Roles separated; patient sees only released results; clinician views; sign-up, "
                            "self-check privacy and sharing; assistant answers from own records", "Pass"],
        ["Total", "37", "", "37 passed"],
    ]), "Automated test suite (python -m pytest, 37 passed)"))
    add(("h3", "Protocol and Leakage Checks"))
    add(("bullets", [
        "**Patient separation.** test_splits.py and test_fourclient.py verify that no patient appears in more than one "
        "client's training data, the validation fold or the test fold.",
        "**Single use of the test fold.** evaluate.py and triage.py refuse a second full fold-10 run without --rerun, "
        "and all choices (combiner, thresholds, triage bands, ablations, explanations) were made on fold 9.",
        "**Validation versus test.** The selected gate scores 0.909 cross-validated on fold 9 and 0.905 on the test "
        "fold; the small gap shows that the validation-based selection did not over-fit.",
    ]))
    add(("h3", "Upload Smoke Test"))
    add(("p", "Six fold-10 recordings (one single-label case per class, plus one MI + STTC case) were each submitted to "
              "the portal as a CSV, a rendered 3×4 PDF, a PNG and a photo-like JPG. The CSV results matched the Engine's "
              "direct predictions on the signal, every image path digitized without error, and, as designed, every "
              "photo finding was referred because photo triage is not yet validated. This test also found that PyMuPDF "
              "and OpenCV were missing from requirements.txt; both were added."))

    # ------------------------------------------------------------------ 6
    add(("chapter", "Project Demonstration"))
    add(("p", "The system is started with the commands below; the portal then runs at http://127.0.0.1:8000. Demo "
              "accounts use the password demo1234: clinician@cardiosentinel.demo (doctor workspace) and "
              "patient@cardiosentinel.demo (patient app). With trained models present the portal runs in live mode: "
              "demo patients are seeded with real PTB-XL fold-10 recordings and real predictions."))
    add(("code", "pip install -r requirements.txt && pip install -e .\n"
                 "bash scripts/download.sh\n"
                 "python -m cardiomoe.data.ptbxl --prepare\n"
                 "python -m cardiomoe.fourclient.train --seeds 42 43 44\n"
                 "python -m cardiomoe.fourclient.aggregate --seeds 42 43 44\n"
                 "python -m cardiomoe.fourclient.evaluate --seeds 42 43 44\n"
                 "python -m cardiomoe.fourclient.triage --seeds 42 43 44\n"
                 "python web/server.py            # http://127.0.0.1:8000"))
    add(("p", "The following screenshots were taken from the running portal."))
    add(("fig", "fig:s1", "shot:01_landing", "Welcome page of the CardioSentinel portal", 6.0))
    add(("fig", "fig:s3", "shot:03_patient_home", "Patient home: latest heart check, doctor, next appointment and "
                                                  "mood picker", 6.0))
    add(("fig", "fig:s4", "shot:04_check", "'Check my heart': upload an ECG file or up to four photos, scans or PDFs, "
                                           "or try an example recording", 6.0))
    add(("fig", "fig:s5", "shot:05_result", "Result page: overall verdict, the five findings with their AI likelihood, "
                                            "and sharing with the doctor", 6.0))
    add(("fig", "fig:s6", "shot:06_assistant", "Voice assistant that answers from the patient's own records", 6.0))
    add(("fig", "fig:s7", "shot:07_queue", "Doctor workspace: queue of ECGs to review with the unclear findings", 6.0))
    add(("fig", "fig:s8", "shot:08_ecg_viewer", "12-lead ECG viewer (3×4 + rhythm, 25 mm/s, 10 mm/mV) with sign-off", 6.0))
    add(("fig", "fig:s9", "shot:09_inbox", "Patient shares inbox in the doctor workspace", 6.0))

    # ------------------------------------------------------------------ 7
    add(("chapter", "Result and Discussion"))
    add(("h2", "Dataset"))
    add(("p", "All experiments use PTB-XL v1.0.3 [5], an open-access (CC BY 4.0) collection distributed through "
              "PhysioNet [37]: 21,799 clinical 12-lead ECGs of 10 s from 18,869 patients, annotated with SCP-ECG "
              "statements by up to two cardiologists. After removing 411 records without a diagnostic superclass, "
              "21,388 records from 18,617 patients remain. The mean age is 59.4 ± 16.8 years, 48.1% of records belong "
              "to female patients, the mean number of labels per record is 1.30 and 24.1% of records are multi-label. "
              "Folds 1–8 (17,084 records, 80%) are the training data, fold 9 (2,146, 10%) the validation set and fold 10 "
              "(2,158 records from 1,877 patients, 10%) the test set ({tab:dataset}, {fig:classdist}). NORM is the most "
              "frequent superclass (44.5%) and HYP the rarest (12.4%)."))
    add(("table", "tab:dataset", ("tex", "tab:dataset")))
    add(("fig", "fig:classdist", "paperfig:class_distribution", "Number of records carrying each diagnostic superclass "
                                                               "in the training, validation and test folds", 4.6))
    add(("h2", "Comparison with Traditional Machine Learning and Single Clients"))
    add(("p", "{tab:main} compares the proposed model with traditional classifiers on 172 hand-crafted features, the "
              "individual deep clients, the learned aggregation variants and a centrally trained reference. All rows "
              "were evaluated once on the same test fold with thresholds chosen on fold 9."))
    add(("table", "tab:main", ("tex", "tab:main")))
    add(("p", "The four deep clients, each trained on a quarter of the data, reach test macro AUROCs between 0.882 "
              "(Client B, 1D ResNet) and 0.896 (Client D, CNN-LSTM). The ResNet, despite the most parameters (6.73M), is "
              "the weakest and stops early after 13.7 epochs on average, a sign of over-fitting on 4,280 records. The "
              "proposed equal-weight gate raises the macro AUROC to 0.905 ± 0.000, the label-wise accuracy to 86.9% "
              "(from at most 85.7%), the macro F1 to 70.7% (from at most 69.2%) and the macro precision to 69.8%, while "
              "keeping the recall (72.5%) at the level of the clients. A CNN-LSTM trained centrally on all eight folds "
              "attains 0.909 ± 0.001 and the same macro F1, so the federated gate recovers about 70% of the AUROC lost by "
              "splitting the data and ties the pooled model in F1."))
    add(("p", "The traditional classifiers are strong baselines. On pooled data XGBoost reaches 0.909 ± 0.002 and a "
              "macro F1 of 72.6%, because its amplitude features keep the absolute voltages that per-recording z-scoring "
              "removes. Trained on a single client's folds, however, XGBoost reaches 0.898 and every locally trainable "
              "model stays below the federated gate ({tab:tradlocal}). Per class, the gate has the highest AUROC in every "
              "superclass, with its largest gain on the hardest class, HYP (0.828 versus at most 0.809; {tab:clientgate}, "
              "{fig:modelcomp})."))
    add(("table", "tab:tradlocal", ("tex", "tab:tradlocal")))
    add(("table", "tab:clientgate", ("tex", "tab:clientgate")))
    add(("fig", "fig:modelcomp", "paperfig:model_comparison", "Test macro AUROC of the four clients and the three "
                                                              "aggregation variants (mean ± SD over 3 seeds)", 4.6))
    add(("p", "The gain over the best single client is statistically significant. In a paired bootstrap on the test "
              "fold the gate exceeds the best client chosen on fold 9 by +0.012 [+0.009, +0.016] AUROC for seed 42, "
              "+0.011 [+0.007, +0.014] for seed 43 and +0.009 [+0.005, +0.013] for seed 44; the macro F1 improves by "
              "+0.018, +0.010 and +0.024, with the interval excluding zero in two of three seeds."))
    add(("h2", "Ablation Experiments"))
    add(("p", "**Client subsets.** Every one of the 15 non-empty subsets of {A, B, C, D} was combined with equal weights "
              "({tab:subsets}). The full federation is best on both splits (fold 9: 0.909; test: 0.905), every pair "
              "beats every single client, and returns diminish (single at most 0.896, pairs 0.903, triples 0.905). The "
              "Shapley values ({tab:shapley}) rank C and D highest but are of the same order (0.097–0.105 on the test "
              "fold). This explains why equal weighting works: the learned static weights give Client A almost nothing "
              "(0.001), although its marginal contribution is comparable."))
    add(("table", "tab:subsets", ("tex", "tab:subsets")))
    add(("table", "tab:shapley", ("tex", "tab:shapley")))
    add(("p", "**Fusion strategy.** Equal weights reach the highest cross-validated macro AUROC (0.909) and were "
              "selected in all three seeds; static weights (0.907) and the dynamic gate (0.904) did not reach the "
              "margin ({tab:fusion}). The learned combiners have a lower BCE (better calibration), but they concentrate "
              "the weight on Clients C and D ({fig:weights}) and lose diversity. The same ordering holds on the test "
              "fold (0.905, 0.903, 0.901), confirming the selection. A later re-examination on fold 9 (logit mean "
              "0.9089, per-class weights 0.9047, stacked logistic regression 0.9075 and 0.9084 per class, versus equal "
              "0.9087) found nothing that clears the 0.002 rule."))
    add(("table", "tab:fusion", ("tex", "tab:fusion")))
    add(("fig", "fig:weights", "paperfig:weights", "Static aggregation weights learned on fold 9 for each seed; the "
                                                   "dashed line marks the equal weight 1/4", 4.6))
    add(("p", "**Activation function and optimizer.** On Client D (seed 42, fold 9 only) ReLU and LeakyReLU perform "
              "identically (0.898) and the activation changes the AUROC by at most 0.004 ({tab:act}). The optimizer "
              "matters much more: AdamW and Adam reach 0.898, whereas SGD with momentum reaches 0.786 and RMSprop 0.855 "
              "at the same maximum learning rate ({tab:opt})."))
    add(("table", "tab:act", ("tex", "tab:act")))
    add(("table", "tab:opt", ("tex", "tab:opt")))
    add(("h2", "Federated Learning Analysis"))
    add(("p", "Because the federation has a single round, its training dynamics are those of the four local trainings "
              "({fig:clientcurves}, {tab:clienttrain}). The convolutional clients converge fastest (best epochs 12.7 and "
              "13.7) and then over-fit; the recurrent clients learn more slowly but keep improving (27.0 and 22.7) and end "
              "with the highest validation AUROCs (0.900 and 0.898). All fitted temperatures exceed one (1.25–1.74), so "
              "every client is over-confident and calibration is needed before averaging. {fig:server} shows that the "
              "global model improves monotonically with the number of aggregated clients and flattens after three."))
    add(("fig", "fig:clientcurves", "paperfig:client_curves", "Local training of the four clients (seed 42): fold-9 "
                                                              "macro AUROC and class-weighted training BCE per epoch", 6.2))
    add(("table", "tab:clienttrain", ("tex", "tab:clienttrain")))
    add(("fig", "fig:server", "paperfig:server_convergence", "Macro AUROC of the global (server) model as a function of "
                                                             "the number of aggregated clients", 4.6))
    add(("p", "**Comparison with other FL frameworks.** Parameter averaging cannot combine heterogeneous clients, so "
              "FedAvg, FedProx (μ = 0.01), SCAFFOLD and FedNova were run with one shared architecture (the CNN-LSTM) on "
              "the same four partitions for 20 rounds of one local epoch ({tab:fedcomp}, {fig:flconv}). FedAvg and FedNova "
              "reach 0.907 and FedProx 0.906, slightly above the one-shot gate (0.905) and close to centralised training "
              "(0.909). SCAFFOLD failed with AdamW local steps: its best model was the one after round 1 (0.842). The "
              "proposed model therefore trades about 0.002 AUROC for a single communication round, five probabilities "
              "per client and query, and a free choice of architecture."))
    add(("table", "tab:fedcomp", ("tex", "tab:fedcomp")))
    add(("fig", "fig:flconv", "paperfig:fl_convergence", "Convergence of the parameter-averaging frameworks (shared "
                                                         "CNN-LSTM, seed 42)", 6.2))
    add(("h2", "Class-wise Analysis and Triage"))
    add(("p", "{fig:confusion} shows the per-class confusion matrices for seed 42 at the fold-9 thresholds (0.42 for NORM, "
              "0.62 for MI and STTC, 0.64 for CD and HYP) and {fig:roc} the ROC curves. NORM is detected most reliably "
              "(887 of 963 positives), CD is predicted conservatively (58 false positives, 177 of 496 positives missed) "
              "and HYP is the most difficult class (129 true positives, 133 false negatives), because hypertrophy "
              "criteria rely on voltage amplitudes that per-recording z-scoring partly removes. {tab:classwise} gives the "
              "class-wise metrics over three seeds: every class except HYP has an AUROC above 0.91."))
    add(("fig", "fig:confusion", "paperfig:confusion_combined.jpg", "Per-class confusion matrices of the proposed "
                                                                    "model on the test fold (seed 42)", 6.2))
    add(("p", "{fig:confusionmean} gives the confusion matrices averaged over the three seeds. Summed over the five "
              "classes, the model makes 2,177 true-positive and 7,198 true-negative decisions out of 10,790 (2,158 ECGs "
              "× 5 classes), a label-wise accuracy of 0.869 with a micro-averaged precision of 0.731 and recall of "
              "0.780; the 615 false negatives are dominated by CD (158), MI (148) and HYP (134)."))
    add(("fig", "fig:confusionmean", "paperfig:confusion_mean.jpg", "Per-class confusion matrices of the proposed model "
                                                                    "on the test fold, mean of seeds 42, 43 and 44", 6.2))
    add(("p", "Compared with the four clients, each at its own fold-9 thresholds (seed 42), for MI the gate both "
              "finds the most positives (437 against 408–431) and raises the fewest false alarms (200 against "
              "226–258). For CD it trades recall for precision: the clients find 314–355 positives with 110–163 false "
              "positives, the gate 319 with only 58. The gain of the combination is therefore mainly in precision, in "
              "line with {tab:main}, where the macro precision rises from at most 67.0% to 69.8% while the recall "
              "stays at the level of the clients."))
    add(("fig", "fig:roc", "paperfig:roc", "ROC curves of the proposed model on the test fold (seed 42)", 4.2))
    add(("table", "tab:classwise", ("tex", "tab:classwise")))
    add(("p", "**Triage.** Applying the rule-in / rule-out bands fitted on fold 9 to the test fold, 53.7% of all findings "
              "are decided automatically with a macro precision of 0.937, a macro recall of 0.908 over the automated "
              "classes and an accuracy of 0.967; 17.8% of the ECGs are decided completely without referral. NORM, MI and "
              "CD were automated in all seeds, STTC in one of three seeds and HYP only in rule-out mode. Not every target "
              "was met: the test recall of CD was 0.880. Stricter targets trade coverage for precision and recall on "
              "fold 9: a 0.95 target automates 39% of findings at 0.976 / 0.973 and a 0.97 target 33% at 0.995 / 0.994."))
    add(("h2", "Explainability Results"))
    add(("p", "Explanations were computed for the seed-42 equal-weight gate on 300 random fold-9 records; the test fold "
              "was not used. The attributions are spread over all twelve leads (none above 11.0%; uniform 8.3%), but the "
              "emphasis follows the expected anatomy ({fig:xaileads}): for MI aVF, V2, II and V3; for HYP V6, V5, aVR "
              "and V1, the leads of the classical voltage criteria; for CD II, aVR, V6 and V1, where bundle branch "
              "blocks are recognised. Local explanations ({fig:xailocal}) peak at the QRS complexes and ST segments "
              "rather than the baseline. The deletion test ({tab:deletion}) confirms faithfulness for NORM, MI and CD "
              "(removing the top-3 leads lowers the probability by 0.135, 0.057 and 0.070 against 0.074, 0.006 and "
              "0.009 for random leads); for STTC and HYP the evidence is redundant across leads."))
    add(("fig", "fig:xaileads", "paperfig:xai_leads", "Global explanation: share of the absolute Integrated-Gradients "
                                                      "attribution per lead (%) for each superclass", 4.8))
    add(("fig", "fig:xailocal", "paperfig:xai_local", "Local explanations for the most confident true-positive MI and "
                                                      "STTC validation records", 6.2))
    add(("table", "tab:deletion", ("tex", "tab:deletion")))
    add(("h2", "Reading Photos of Paper ECGs"))
    add(("p", "The digitizer's median per-lead correlation with the recorded signal on fold 10 is 0.78 for digital red "
              "grids, 0.44 for colour scans, 0.34 for phone photos and monitor photos, 0.19 for grayscale scans, 0.18 "
              "for green grids, 0.12 for stained scans and 0.00 for stained phone photos; 98% of the photos were "
              "digitized. {tab:photo} gives the one-time fold-10 evaluation on 100 real photos of 62 records."))
    add(("table", "tab:photo", ("rows", ["Model", "Macro AUROC (95% CI by record)", "Macro F1", "Exact match"], [
        ["Image gate (static weights, almost all on D)", "0.841 (0.746–0.916)", "0.558", "0.388"],
        ["Signal gate on the same digitized photos (no fine-tuning)", "0.859 (0.792–0.919)", "0.343", "0.082"],
        ["Signal gate on the recorded signal of the same records", "0.946 (0.906–0.975)", "0.696", "0.684"],
    ]), "Real fold-10 photos (100 photos, 62 records, seed 42, evaluated once)"))
    add(("p", "Fine-tuning on 104 photos did not raise the AUROC over the untuned models (0.841 versus 0.859), but it "
              "fixed the thresholds (F1 and exact match). Photo triage fitted on 59 fold-9 photos decides 50% of "
              "findings at an accuracy of 0.943, but with a macro precision of 0.873 and recall of 0.812 (MI precision "
              "0.60), below the 0.90 target. The portal therefore refers every photo finding to a doctor until the bands "
              "are fitted on more validation photos."))
    add(("h2", "Comparison with the State of the Art"))
    add(("p", "{tab:sota} compares the proposed model with published results on the PTB-XL superclass task. The models "
              "of [6] were trained centrally on the same folds and are the most comparable. The proposed model does not "
              "exceed the best centrally trained models: its 0.905 is 0.016–0.029 below them. Part of the gap comes from "
              "the data restriction (each client sees 25% of the records) and part from the training recipe, since our "
              "own pooled CNN-LSTM reaches 0.909; the recipe was kept fixed for all models to isolate the effect of "
              "federation."))
    add(("table", "tab:sota", ("tex", "tab:sota")))
    add(("p", "**Strengths.** The approach delivers most of the benefit of pooling without moving any record or "
              "parameter, outperforms every model that can be trained at a single site, comes within 0.002 AUROC of "
              "FedAvg with one communication step and free architectures, is significantly better than the best "
              "institution in every seed and is highly reproducible (SD of the macro AUROC below 0.001). **Limitations.** "
              "The four clients are simulated partitions of one dataset with near-identical label distributions; fold 9 "
              "is used for early stopping, calibration, thresholds and the gate, so validation figures are optimistic; "
              "the learned gates did not beat equal weighting; HYP remains weak; FedAvg-type baselines and pooled "
              "XGBoost are stronger; the FL baselines and ablations used a single seed; and photo reading is not yet "
              "accurate enough for automatic reports."))
    add(("h2", "Cost Analysis"))
    add(("p", "The project has no licence or data cost and used no paid cloud resources ({tab:cost}). The compute time "
              "of the main experiment follows from the measured epoch times and the mean number of epochs run "
              "({tab:clienttrain}): Client A 20.7 × 26.8 s ≈ 9.2 min, Client B 21.7 × 38.5 s ≈ 13.9 min, Client C "
              "35.0 × 65.4 s ≈ 38.2 min and Client D 30.7 × 28.2 s ≈ 14.4 min, i.e. about 76 minutes per seed and about "
              "3.8 hours for the three seeds on a single laptop. Because the clients train independently, real "
              "institutions would train in parallel, so the wall-clock time of the federation is that of the slowest "
              "client (about 38 minutes). Communication is one round; at query time each client returns five numbers."))
    add(("table", "tab:cost", ("rows", ["Item", "Cost"], [
        ["PTB-XL v1.0.3 (CC BY 4.0), ECG-Image-Database (CC BY-ND 4.0)", "Free (open access)"],
        ["Python, PyTorch, scikit-learn, XGBoost, FastAPI, SQLite, Tectonic", "Free (open source)"],
        ["Hardware", "Existing laptop (8 GB memory); no GPU server"],
        ["Training compute, 4 clients × 3 seeds", "≈ 3.8 h on the laptop (≈ 76 min per seed)"],
        ["Voice assistant language model (optional)", "Pay-per-use via OpenRouter; offline fallback is free"],
        ["Communication per ECG at inference", "20 numbers (5 probabilities × 4 clients)"],
    ]), "Cost analysis"))

    # ------------------------------------------------------------------ 8
    add(("chapter", "Conclusion"))
    add(("p", "This project built CardioSentinel, a privacy-preserving, one-shot federated model for the multi-label "
              "classification of 12-lead ECGs into the five PTB-XL diagnostic superclasses, and deployed it in a patient "
              "and clinician portal. Four clients trained a 1D CNN, a 1D ResNet, a BiLSTM and a CNN-LSTM on disjoint, "
              "patient-separated quarters of the training data, calibrated them, and shared only their class "
              "probabilities with a server that combines them on the probability simplex. A parsimony-first selection on "
              "patient-grouped validation predictions chose equal weighting in all three seeds."))
    add(("p", "On the untouched test fold the federated model reached a macro AUROC of 0.905 ± 0.000, a macro F1 of "
              "70.7% and a label-wise accuracy of 86.9%, improved on the best single client in every seed (+0.009 to "
              "+0.012 AUROC, paired bootstrap), and came within 0.004 AUROC of a model trained on all data pooled "
              "together while matching its macro F1. It outperformed every model trainable at a single site, whereas "
              "FedAvg with a shared CNN-LSTM was 0.002 AUROC better after 20 rounds and pooled-data XGBoost 0.004 better. "
              "Client-subset ablations and Shapley values showed that every institution contributes. Integrated Gradients "
              "attributed the predictions to plausible leads and waveform segments, and per-finding triage automated "
              "53.7% of findings at a precision of 0.937 and an accuracy of 0.967 while referring the rest."))
    add(("p", "**Future work.** (1) Evaluate the method on genuinely heterogeneous institutions with different devices, "
              "populations and label distributions, where a learned gate is expected to matter more. (2) Add a client "
              "trained on hand-crafted amplitude features, and repeat the parameter-averaging baselines over several "
              "seeds. (3) Give the gate inputs that differ between clients, such as other modalities or site metadata, "
              "and study formal privacy guarantees such as differential privacy on the shared probabilities. (4) Improve "
              "hypertrophy recognition by restoring absolute voltage information, fit the photo triage on more "
              "validation photos, and validate the explanations with cardiologists."))
    return B


APPENDIX_FILES = [
    ("A.1 Weighted gated aggregation (src/cardiomoe/fourclient/gate.py)", "src/cardiomoe/fourclient/gate.py", 23, 77),
    ("A.2 Client D and the client wrapper (src/cardiomoe/fourclient/models.py)", "src/cardiomoe/fourclient/models.py", 111, 142),
    ("A.3 Temperature scaling (src/cardiomoe/fusion/calibration.py)", "src/cardiomoe/fusion/calibration.py", 9, 22),
    ("A.4 Per-finding triage rule (src/cardiomoe/fourclient/triage.py)", "src/cardiomoe/fourclient/triage.py", 90, 107),
]
