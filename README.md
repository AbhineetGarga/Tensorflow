# TensorFlow Labs: Graphs, Tensors, CNNs & Deep Networks

A clean set of hands-on notebooks exploring TensorFlow concepts — from raw graphs and tensors to convolutional networks and deeper architectures. Curated for learning and quick experimentation.

_Last updated: **2025-08-28**_

---

## 📁 Repository Structure

```
.
├── cnn.ipynb       # Convolutional Neural Networks experiments
├── deep.ipynb      # Deeper MLP/ANN experiments
├── graph.ipynb     # TensorFlow graphs & low-level ops
├── Tensor.ipynb    # Tensors, shapes, dtypes, broadcasting basics
└── README.md
```

> Tip: Open the notebooks in order: **Tensor → graph → deep → cnn**.

---

## 🔍 Notebook Index (auto‑generated)

| Notebook | Title / First Heading | Headings (first few) | ~Code lines | ~Markdown lines | Top imports |
|---|---|---|---:|---:|---|
| `graph.ipynb` | (no title found) | — | 36 | 0 | tensorflow |
| `Tensor.ipynb` | (no title found) | — | 119 | 3 | tensorflow |
| `deep.ipynb` | (no title found) | — | 107 | 7 | tensorflow, numpy, matplotlib |
| `cnn.ipynb` | What is CNN? | What is CNN?; 🎨 Matplotlib Colormap (`cmap`) Guide | 142 | 73 | sklearn, numpy, pandas, matplotlib, seaborn, keras, tensorflow |

---

## 🚀 Quickstart

> Recommended: **Python 3.11** with a fresh virtual environment. TensorFlow currently has limited/laggy support on 3.12+ across platforms.

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# 2) Upgrade pip
python -m pip install --upgrade pip

# 3) Install dependencies
pip install -r requirements.txt

# 4) (Optional) Create a Jupyter kernel tied to this venv
python -m ipykernel install --user --name=tensorflow-labs --display-name "Python (tensorflow-labs)"

# 5) Launch Jupyter
jupyter notebook
# or
jupyter lab
```

---

## 🧪 Datasets

Detected in notebooks: **Fashion-MNIST, MNIST**.

If you see code like `tf.keras.datasets.mnist.load_data()` or `cifar10.load_data()`, datasets will download automatically at first run. If behind a firewall/proxy, set `HTTPS_PROXY` and `HTTP_PROXY` env vars.

---

## ⚙️ Requirements

Pinned, conservative versions to minimize setup pain:

```
ipykernel
jupyter
keras>=2.15,<3.0
matplotlib>=3.8
numpy>=1.26,<2.0
pandas>=2.0
scikit-learn
seaborn
tensorflow>=2.15,<2.17
```

Save the above as **requirements.txt** and install with `pip install -r requirements.txt`.

> Windows note: If you see build errors for packages like `opencv-python`, try `pip install --only-binary :all: opencv-python` or install Visual C++ Build Tools.

---

## 🧠 TensorFlow 1.x vs 2.x — known gotchas

- **TF1-style sessions/graphs detected.** This repo uses some TensorFlow 1.x patterns (e.g., `Session()`, `get_default_graph()`). In TensorFlow 2.x, wrap them with `tf.compat.v1` and optionally disable eager execution.
  ```python
  import tensorflow as tf
  tf.compat.v1.disable_eager_execution()
  with tf.compat.v1.Session() as sess:
      ...
  ```
  Replace `tf.get_default_graph()` with `tf.Graph()` or `tf.compat.v1.get_default_graph()`. 


**Common fixes:**

- Use `tf.keras` high-level APIs for new code (layers, models, optimizers).
- Replace `tf.ConfigProto`, `tf.Session`, `tf.placeholder` with Keras/TF2 equivalents or `tf.compat.v1.*` fallbacks.
- Prefer `model.fit` over manual training loops unless you need full control.

---

## 🗺️ How to run each notebook

1. **Tensor.ipynb** — tensor creation, dtypes, reshaping, broadcasting, basic math.
2. **graph.ipynb** — low-level ops, graphs vs. eager mode, device placement.
3. **deep.ipynb** — dense networks for classification; training loop, metrics, overfitting/regularization.
4. **cnn.ipynb** — conv/pool stacks, data augmentation, training curves, evaluation.

> Pro tip: Enable GPU if available. On Colab, switch **Runtime → Change runtime type → GPU**.

---

## 🧾 Reproducibility checklist

```python
import os, random, numpy as np, tensorflow as tf
os.environ["PYTHONHASHSEED"] = "0"
random.seed(0); np.random.seed(0); tf.random.set_seed(0)
```
- Record package versions with `pip freeze > pip-freeze.txt`.
- Log training runs (TensorBoard): `tensorboard --logdir runs/` and add callbacks in Keras.

---

## 🧰 Troubleshooting

- **`AttributeError: module 'tensorflow' has no attribute 'get_default_graph'`**  
  Use `tf.compat.v1.get_default_graph()` or create an explicit `tf.Graph()` context.

- **`RuntimeError: tf.function` with graph tensors**  
  Stick to eager tensors or wrap legacy code under `@tf.function` carefully.

- **Dataset download errors**  
  Use a mirror or download manually, then point code to local paths.

---

## 📜 License

MIT — feel free to use and adapt. If you build on these notebooks, a star ⭐️ on the repo would be awesome.

---

## 🙌 Acknowledgements

- TensorFlow & Keras teams for outstanding docs and examples.
- Open datasets community (e.g., MNIST, CIFAR-10).

---

_Hand‑crafted README generated from the notebooks present in this repository._
