# ML Tutorial — Fundamental Algorithms from Scratch

A beginner-friendly series of Jupyter notebooks that walks through key machine learning algorithms with step-by-step explanations, visualizations, and runnable code. Each notebook is self-contained and designed to be read top-to-bottom.

---

## 📚 Notebooks

| # | Notebook | Topic | Algorithms |
|---|---|---|---|
| 0 | [0_svm-iforest.ipynb](0_svm-iforest.ipynb) | Anomaly Detection — Deep Dive | OC-SVM · Isolation Forest · LOF · PCA |
| 1 | [1_svm-apply.ipynb](1_svm-apply.ipynb) | Anomaly Detection — Applied | OC-SVM · Isolation Forest · Feature Engineering |
| 2 | [2_transformer-tutorial.ipynb](2_transformer-tutorial.ipynb) | Transformer Architecture | Attention · BPE · Positional Encoding · PyTorch |
| 3 | [3_convnet-interpretability.ipynb](3_convnet-interpretability.ipynb) | Interpreting ConvNets | Grad-CAM · Filter Visualization · t-SNE · VGG16 |
| 4 | [4_language-models.ipynb](4_language-models.ipynb) | Language Models & Transformers | GRU · Seq2Seq · RoBERTa · Attention Head Ablation |
| 5 | [5_neural-network-autoencoder.ipynb](5_neural-network-autoencoder.ipynb) | Neural Network — Autoencoder | Autoencoder · LOF · ROC/PR · Permutation Importance · Model Export |
| 6 | [6_feedforward-nn.ipynb](6_feedforward-nn.ipynb) | Toy Neural Network from Scratch | MLP · Xavier Init · Backprop · Activation Functions · Ablation Studies |

---

## 🗺️ Recommended Learning Path

```
Beginner          →   0_svm-iforest  →  1_svm-apply  →  5_neural-network-autoencoder  →  6_feedforward-nn
PyTorch / DL      →   2_transformer  →  3_convnet    →  4_language-models
```

Notebooks **0** and **1** use only `scikit-learn` and `numpy` — no deep learning required. Notebooks **2–4** require PyTorch.

---

## ⚙️ Setup

### Requirements
- Python 3.9+
- Jupyter Notebook or JupyterLab (or VS Code with the Jupyter extension)

### Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

For the deep learning notebooks (2–4), also install PyTorch:

```bash
# CPU-only
pip install torch torchvision

# Or visit https://pytorch.org/get-started/locally/ to get the right command for your hardware (CUDA / MPS)
```

Notebook **4** optionally uses HuggingFace Transformers for fine-tuning RoBERTa:

```bash
pip install transformers datasets
```

### Run

```bash
git clone https://github.com/rajatsainju2025/ml-tutorial.git
cd ml-tutorial
jupyter notebook
```

---

## 📖 Topics Covered

- **Anomaly detection**: One-Class SVM, Isolation Forest, Local Outlier Factor, KDE, PCA decision boundaries
- **Feature engineering**: Time-domain (Kurtosis, Crest Factor, Skewness) and frequency-domain (Spectral Centroid) features from vibration signals
- **Transformer architecture**: Attention mechanism, multi-head attention, masked self-attention, cross-attention, positional encoding, BPE tokenization
- **ConvNet interpretability**: Intermediate activations, filter visualization via gradient ascent, Grad-CAM, t-SNE embeddings
- **Language modeling**: Character-level GRU, Seq2Seq, logit lens, probing classifiers, fine-tuning pre-trained transformers
- **Neural networks / autoencoders**: PyTorch autoencoder for anomaly detection, reconstruction error threshold, ROC/PR curves, confusion matrices, hyperparameter sensitivity, permutation feature importance, model persistence
- **Feedforward NN from scratch**: MLP built with NumPy only, Xavier initialisation, forward propagation, binary cross-entropy, backpropagation, decision boundary visualisation, activation/depth/lr ablation studies

---

## 📄 License

MIT
