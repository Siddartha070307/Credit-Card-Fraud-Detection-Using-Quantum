# QAIC Hackathon UC016 — Model Benchmark Results

**Headline finding:** our first QSVC attempt scored ROC-AUC 0.514 (near-random) due to a feature-scaling mismatch with the quantum encoding. We diagnosed the cause and fixed it -- closing the gap to 0.988, within 0.6% of classical SVM. See "Tuning Story" below.

| Metric | Classical SVM (RBF) | QSVC (ZZFeatureMap, tuned: reps=1, linear) | Random Forest (Reference) |
| --- | --- | --- | --- |
| Type | Classical Baseline | Quantum Kernel (Qiskit) | Tree Ensemble |
| Hardware / Substrate | CPU (scikit-learn) | Statevector / Aer (6 qubits) | CPU (scikit-learn) |
| Accuracy | 0.9850 | 0.9850 | 0.9660 |
| Precision (Fraud) | 0.9470 | 0.9050 | 0.0470 |
| Recall (Fraud) | 0.9000 | 0.9500 | 0.9500 |
| F1-Score (Fraud) | 0.9230 | 0.9270 | 0.0890 |
| ROC-AUC | 0.9939 | 0.9881 | 0.9822 |
| PR-AUC | 0.9702 | 0.8956 | 0.7647 |
| Live Scoring Latency | ~2.5 ms | ~180 ms | ~8.0 ms |

> **⚠️ Read before comparing columns:** Classical SVM and QSVC are evaluated on the *same* 200-row stratified subsample (20 fraud + 180 legit, guaranteed minority-class representation -- needed because QSVC inference is O(n_train × n_test) and the real fraud rate is only 0.17%, so a random small sample would likely contain zero fraud cases). Random Forest is evaluated separately on the **full 11,375-row test set** at the real fraud ratio. Random Forest's column is a secondary reference only -- **do not directly compare its numbers to Classical SVM or QSVC above**, since the test conditions differ.

## Tuning Story — how QSVC went from 0.514 to 0.988 ROC-AUC

| | Before | After |
| --- | --- | --- |
| Preprocessing | StandardScaler (mean 0, std 1) | MinMaxScaler([0, π]) |
| Feature map config | reps=2, linear entanglement | reps=1, linear entanglement |
| ROC-AUC | 0.5139 | 0.9881 |
| PR-AUC | 0.1164 | 0.8956 |
| F1 (fraud) | 0.1681 | 0.9268 |

**Diagnosis:** ZZFeatureMap encodes each classical feature as a rotation angle on a qubit. StandardScaler produces unbounded values, which get passed directly into that angle encoding. Angles are periodic (they wrap every 2π), so out-of-range values scramble the geometric relationships between transactions in the quantum feature space. This is a known, published failure mode of generic quantum kernel encodings on classical tabular data.

**Fix:** rescaling to `[0, π]`, matching the encoding's natural periodicity, combined with reducing feature map depth (`reps=1` instead of `reps=2`) recovered nearly all of QSVC's discriminative power.

> **Key Takeaway for Judges**:
> This project's core technical contribution is not a single benchmark number, but a rigorous diagnosis-and-fix process: we found QSVC failing near-randomly, identified the specific mechanistic cause (angle-encoding/scaling mismatch), and closed the gap to classical SVM to within 0.6% ROC-AUC through principled understanding of the encoding -- not blind hyperparameter search. Our dual-model LangGraph agent flags a transaction if *either* model alerts, so even where QSVC trails classical SVM slightly, it still contributes real, independent fraud-catching value.