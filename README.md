# SmallFSL

> Few-shot learning for small tabular data. Adaptive prototypes, calibrated prediction sets.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-planning-orange.svg)]()

**SmallFSL** is a library for **few-shot classification on tabular data** ($K$-way, $N$-shot with $N \le 20$). Unlike image-based few-shot methods, SmallFSL does not require meta-training — it adapts prototypes and hyperparameters directly from the support set. Prediction sets come with finite-sample coverage guarantees via weighted conformal prediction.

- **No meta-training.** Works directly from the support set.
- **Adaptive hyperparameters** derived from $K$, $N$, $d$.
- **Calibrated prediction sets** via weighted conformal.
- **Tabular-first.** Designed for the regime where images are not available.

---

## Why SmallFSL?

Few-shot learning is dominated by image benchmarks (Omniglot, Mini-ImageNet, CIFAR-FS). On **tabular** data — medical cohorts, chemistry, sensors — few-shot is under-explored, yet the need is real: collecting even 5 labeled examples per class is often expensive in scientific settings.

Existing approaches:

| Method          | Meta-training | Images | Tabular | Calibrated sets |
| --------------- | ------------- | ------ | ------- | --------------- |
| ProtoNet        | required      | ✓      | ✗       | ✗               |
| MAML            | required      | ✓      | ✗       | ✗               |
| In-context (LLM)| required      | —      | partial | ✗               |
| **SmallFSL**    | **none**      | ✗      | ✓       | ✓               |

---

## Status

**Planning stage.** Placeholder for future development.

Planned milestones:
- [ ] Prototype-based baseline (ProtoNet-style) for tabular
- [ ] Adaptive width/dropout derived from $K$, $N$, $d$
- [ ] Weighted conformal prediction sets
- [ ] Benchmark on 10–15 small tabular datasets
- [ ] Comparison with fine-tuned MLP, kNN, logistic regression
- [ ] PyPI release

---

## Planned API

```python
import numpy as np
from smallfsl import SmallFSLClassifier

# Support set: K classes, N examples each
X_support = np.random.randn(5 * 5, 10)   # 5-way, 5-shot
y_support = np.repeat(np.arange(5), 5)

# Query set
X_query = np.random.randn(50, 10)
y_query = np.random.randint(0, 5, 50)

clf = SmallFSLClassifier()
clf.fit(X_support, y_support)
y_pred = clf.predict(X_query)
sets = clf.predict_set(X_query, alpha=0.1)   # prediction sets
```

---

## Planned Method

### 1. Adaptive prototype computation

Class prototypes computed as **weighted** means of support examples, with weights from a learned metric:

$$
\mu_k = \frac{\sum_{i: y_i = k} w_i \, \phi(x_i)}{\sum_{i: y_i = k} w_i}
$$

where $\phi$ is an adaptive embedding (MLP with width from SmallMLP formula).

### 2. Adaptive hyperparameters

Width, dropout, and bias init derived from $K$, $N$, $d$ — same approach as SmallMLP, adapted to the few-shot regime.

### 3. Calibrated prediction sets

Weighted conformal prediction on top of prototype-based probabilities. Coverage guaranteed under exchangeability of query examples.

---

## When to use SmallFSL (planned)

**Good fit:**
- $K \le 10$ classes, $N \le 20$ examples per class.
- Tabular data with **nonlinear** class boundaries.
- Scientific applications: rare disease cohorts, novel molecules, sensor calibration.
- When **calibrated uncertainty** matters.

**Not a good fit:**
- Image or text data (use standard FSL methods).
- $N > 50$ (few-shot becomes regular classification).
- Linear problems (logistic regression may suffice).

---

## Roadmap

| Milestone | Status |
|-----------|--------|
| Tabular ProtoNet baseline | planned |
| Adaptive hyperparameters | planned |
| Conformal prediction sets | planned |
| Benchmark (10–15 datasets) | planned |
| Comparison with baselines | planned |
| PyPI release | planned |
| arXiv preprint | planned |

---

## Related work

- **SmallGBM** — gradient boosting for small tabular data. [GitHub](https://github.com/nsdmlk/SmallGBM)
- **SmallMLP** — adaptive MLP for small nonlinear data. [GitHub](https://github.com/nsdmlk/SmallMLP)
- **SmallGP** — Gaussian Processes for small data. [GitHub](https://github.com/nsdmlk/SmallGP)

Part of the **Small ML** series.

---

## References (planned reading)

- Vinyals et al. (2016), *Matching Networks for One Shot Learning*
- Snell et al. (2017), *Prototypical Networks for Few-Shot Learning*
- Finn et al. (2017), *Model-Agnostic Meta-Learning*
- Triantafillou et al. (2020), *Meta-Dataset*
- (to be extended) tabular FSL: search "few-shot learning tabular data"

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgments

Built independently during undergraduate studies at Beijing Institute of Technology.
