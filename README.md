<p align="center"><sub><b>ANNALS OF TECHNOLOGY</b></sub></p>

<h1 align="center">Taking the Machine Apart</h1>

<p align="center"><i>A set of notebooks that builds machine learning by hand—one bearing, one neuron,<br>one attention head at a time—and asks the reader to watch it turn.</i></p>

<p align="center">By Rajat Sainju</p>

---

**SOMEWHERE IN A PUMP ROOM**, a bearing has begun to fail. Nobody can hear it yet. The motor hums at the pitch it has held for years, and the gauges read what they have always read. But a small flaw has opened on the inner race, and each time a ball rolls across it the shaft gives a shudder a few thousandths of a second long, as regular as a metronome. An accelerometer bolted to the casing records it faithfully, along with everything else. The question that opens this tutorial, and in a sense the whole field, is how a machine might notice what the people in the room have not.

The first notebook answers by building the problem from nothing. It simulates the bearing: a clean rotation for the healthy shaft, sharp impulses for the fault, a wobble for imbalance, noise for the world. Then it does what an engineer with a stethoscope would do, and reduces each signal to a few numbers—kurtosis, crest factor, skewness, spectral centroid—that say how spiky, how lopsided, and how high-pitched it is. Only then do the algorithms arrive. A One-Class SVM draws a fence around normal. An Isolation Forest counts how many random cuts it takes to corner a point; the strange ones are easy to trap. A Local Outlier Factor compares each point with its neighbors and flags the ones that live alone. By the end, the reader has met three definitions of "odd," and has learned that choosing among them is less a matter of accuracy than of temperament.

Most introductions to machine learning begin with a library call. This one begins with a suspicion: that a model you cannot take apart is a model you do not understand. The notebooks are meant to be read from top to bottom, like chapters, and each one builds the thing it describes before it uses it.

<p align="center">◆</p>

The purest example is the sixth notebook, which builds a neural network with nothing but NumPy. There is no PyTorch here, and no Keras. The data are two interlocking crescents, the kind no straight line can separate. The notebook writes sigmoid, tanh, and ReLU by hand, along with their derivatives. It sets the starting weights with Xavier's rule, so the signal neither fades nor explodes. Then it pushes data forward, measures the error with binary cross-entropy, and sends the blame backward through the layers—backpropagation, which turns out to be the chain rule applied with the patience of a bookkeeper. When training ends, the decision boundary has bent itself around the crescents. The ablations that follow ask the questions a curious student would ask: what if the network were deeper, the activation different, the learning rate too bold?

The fifth notebook returns to the pump room with that new tool. It trains an autoencoder only on healthy vibrations, so the network becomes fluent in normal and nothing else. Shown a faulty signal, it stumbles, and the size of the stumble, the reconstruction error, becomes the alarm. The notebook then holds the autoencoder to the same standard as the classical methods, with ROC and precision-recall curves, confusion matrices, and a permutation test that reveals which feature the network was listening to.

<p align="center">◆</p>

In 2017, eight researchers, most of them at Google, published a paper with a title that sounded like a dare: "Attention Is All You Need." The second notebook takes it at its word. It starts with attention at its smallest, a few vectors deciding how much to look at one another. It shows why the Transformer could read a sentence all at once while the recurrent networks before it had to read word by word. Then it assembles the rest: byte-pair encoding to cut text into tokens, embeddings to turn tokens into vectors, and positional encodings to give the model a sense of order, without which "dog bites man" and "man bites dog" would be the same sentence.

The third and fourth notebooks are PyTorch translations of companion notebooks from François Chollet's "Deep Learning with Python," with more pictures and more prying. The third opens a VGG16 network that has seen more than a million photographs and asks what it learned. Its first layers turn out to detect edges and colors. Deeper filters, coaxed into view by gradient ascent, answer to textures and patterns of growing intricacy. Grad-CAM paints a heat map over a photograph of an elephant, showing where the network was looking when it made up its mind. The fourth teaches a small recurrent network to write like Shakespeare, one character at a time, and then to translate English into Spanish. Then it opens a Transformer and starts removing parts: it silences attention heads one by one, reads the model's half-formed guesses with the logit lens, and traces how information flows through the residual stream.

The seventh notebook is a short parable about meaning. It asks two models how to catch a computer virus, and offers them five answers: two about computers, one about doctors and hand-washing, one about the flu, and one about a stock that "caught a cold." The fast model reads the question and each answer separately; the slower one reads them side by side. Both find the obvious answer, the one that repeats the question's words. But both rank the doctors above the sentence about antivirus software, which says the right thing in different words. Meaning, it turns out, is still partly a matter of vocabulary.

<p align="center">◆</p>

The last folder, `neo-nanogpt`, is the capstone. Following Andrej Karpathy's lecture "Let's reproduce GPT-2," it rebuilds the hundred-and-twenty-four-million-parameter GPT-2 from blank files. It first checks its work by loading OpenAI's released weights, and then begins training its own copy on a million characters of Shakespeare. Along the way it uses the tricks that make modern training fast: fused attention kernels, `torch.compile`, bfloat16 arithmetic, gradient accumulation to reach half-million-token batches, a warmup-then-cosine learning rate, gradient clipping, and training across several GPUs at once. A companion notebook shrinks attention to eight dimensions and two heads, so small that every matrix can be printed and read like a ledger.

None of this replaces the libraries. Nobody should hand-write backpropagation for work that matters. But the bearing in the pump room is a good teacher. It suggests an order of operations that applies well beyond pumps: listen first, build the instrument yourself, and trust it only after you have reëxamined every part. ♦

<p><sub><i>Published in the <code>main</code> branch of <code>rajatsainju2025/ml-tutorial</code>, with the headline "Taking the Machine Apart."</i></sub></p>

---

## The particulars

### The notebooks

| # | Notebook | What it builds | Needs |
|---|---|---|---|
| 0 | [`0_svm-iforest`](various_models/0_svm-iforest.ipynb) | Anomaly detection, in depth: One-Class SVM, Isolation Forest, LOF, on simulated bearing vibrations | scikit-learn |
| 1 | [`1_svm-apply`](various_models/1_svm-apply.ipynb) | The same pipeline, shorter and applied: features, two detectors, comparison | scikit-learn |
| 2 | [`2_transformer-tutorial`](various_models/2_transformer-tutorial.ipynb) | The Transformer from attention up: BPE, embeddings, positional encoding | PyTorch |
| 3 | [`3_convnet-interpretability`](various_models/3_convnet-interpretability.ipynb) | Inside VGG16: activations, filter visualization, Grad-CAM, guided backprop, t-SNE | PyTorch, torchvision |
| 4 | [`4_language-models`](various_models/4_language-models.ipynb) | Character-level GRU, Seq2Seq translation, Transformer, attention-head ablation, logit lens | PyTorch, transformers |
| 5 | [`5_neural-network-autoencoder`](various_models/5_neural-network-autoencoder.ipynb) | An autoencoder for anomalies, with ROC/PR curves, permutation importance, model export | PyTorch |
| 6 | [`6_feedforward-nn`](various_models/6_feedforward-nn.ipynb) | A neural network in NumPy only: forward pass, backprop, ablations | NumPy |
| 7 | [`7_word2vec`](various_models/7_word2vec.ipynb) | Bi-encoder vs. cross-encoder on the "virus" test | sentence-transformers |
| — | [`neo-nanogpt/`](neo-nanogpt) | GPT-2 (124M) rebuilt and trained: [`train_gpt2.py`](neo-nanogpt/train_gpt2.py), [`nb_gpt2`](neo-nanogpt/nb_gpt2.ipynb), [`nb_snippets`](neo-nanogpt/nb_snippets.ipynb) | PyTorch, tiktoken, a GPU |

### A reading order

```
Classical ML   0 → 1 → 6 → 5
Deep learning  2 → 3 → 4 → 7 → neo-nanogpt
```

Notebooks 0 and 1 need no deep learning at all.

### Setup

```bash
git clone https://github.com/rajatsainju2025/ml-tutorial.git
cd ml-tutorial
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
pip install torch torchvision          # notebooks 2–5 (see pytorch.org for CUDA or MPS builds)
pip install transformers datasets      # notebook 4
pip install sentence-transformers      # notebook 7
jupyter lab
```

To train GPT-2 (a CUDA GPU is strongly recommended):

```bash
pip install tiktoken transformers
cd neo-nanogpt
python train_gpt2.py                                    # one GPU
torchrun --standalone --nproc_per_node=8 train_gpt2.py  # eight GPUs
```

### Credits

Notebooks 3 and 4 adapt the companion notebooks of François Chollet's
[*Deep Learning with Python*, third edition](https://www.manning.com/books/deep-learning-with-python-third-edition).
`neo-nanogpt` follows Andrej Karpathy's
[build-nanogpt](https://github.com/karpathy/build-nanogpt). The Transformer is from
[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).

### License

MIT
