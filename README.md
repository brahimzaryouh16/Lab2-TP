# Lab 2 — Deep Learning on MNIST: CNN, Faster R-CNN, VGG16/AlexNet and a ViT from scratch

**Author:** Brahim Zaryouh
**Course:** Deep Learning — Pr. ELAACHAk LOTFI
**Programme:** LSI, Département Génie Informatique, Faculté des Sciences et Techniques de Tanger (Université Abdelmalek Essaadi)
**Hardware:** NVIDIA Tesla T4 (Google Colab, free tier) driven from WSL with the Colab CLI
**Stack:** PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, Python 3.12

This repository contains the code, the raw JSON reports and the figures for Lab 2. The
experiments answer four questions: how well a small CNN classifies MNIST, how a Faster
R-CNN behaves when MNIST is recast as a detection problem, whether ImageNet-pretrained
VGG16/AlexNet transfer to digits, and how a from-scratch Vision Transformer compares with
the CNN.

---

## Results at a glance

| Model | Task | Test accuracy | Macro F1 | Train time | Params | Inference |
|---|---|---:|---:|---:|---:|---:|
| CNN (scratch) | classification | **99.52 %** | **99.52 %** | 114 s | 0.47 M | 0.19 ms/img |
| VGG16 (pretrained, fine-tuned) | classification | 99.25 % | 99.24 % | 793 s | 134.3 M | 4.93 ms/img |
| AlexNet (pretrained, fine-tuned) | classification | 99.00 % | 98.99 % | 107 s | 57.0 M | 1.71 ms/img |
| ViT (scratch) | classification | 98.07 % | 98.05 % | 260 s | 1.20 M | 0.21 ms/img |
| Faster R-CNN (COCO-pretrained) | detection | 93.65 %* | 93.61 %* | 281 s | 18.9 M | 6.88 ms/img |

\* Detection-derived: top-scoring box, correct digit **and** IoU ≥ 0.5; mAP@0.5 = **0.976**.
Faster R-CNN is trained on 8 k images and the pretrained classifiers on 12 k (2 k test),
the others on the full 60 k/10 k split — see the protocol table for the exact budgets.
The CNN is the best accuracy-per-cost choice; the ViT is the most interesting negative
result; the detector solves a different (harder) problem.

![Model comparison](results/figures/fig_comparison.png)

---

## 1. Dataset

MNIST: 60 000 training and 10 000 test 28×28 grayscale images of handwritten digits
(10 balanced classes). The lab links the [Kaggle release](https://www.kaggle.com/datasets/hojjatk/mnist-dataset);
the code uses `torchvision.datasets.MNIST`, which downloads the same 60 k/10 k split and
keeps the pipeline reproducible. Classification images are normalised with the MNIST
mean/std `(0.1307, 0.3081)`.

For the detector the dataset is **reformulated**: each digit is pasted at a deterministic
random position on a 128×128 black canvas, and the 28×28 digit rectangle becomes the
ground-truth box (labels 1…10, 0 = background). This gives Faster R-CNN a localisation
target while keeping the digit content of MNIST. The canvas positions are seeded, so the
train and test sets are fixed across runs.

---

## 2. Protocol and reproducibility

Every run is one `python src/train.py --model …` invocation that writes a self-describing
JSON (config, environment, per-epoch history, final metrics, timing). One model runs per
`colab exec`, and each report is downloaded immediately, so a dead free-tier session loses
only the stage in flight. Re-running skips stages whose report already exists.

| Model | Train / test | Input | Epochs | Batch | Optimiser | Schedule | Extras |
|---|---|---|---:|---:|---|---|---|
| CNN | 60 000 / 10 000 | 28×28×1 | 8 | 128 | Adam, lr 1e-3, wd 1e-4 | cosine | BN, dropout 0.25 |
| ViT | 60 000 / 10 000 | 28×28×1 | 12 | 128 | AdamW, lr 3e-4, wd 0.05 | cosine | label smoothing 0.1, grad clip 1.0 |
| VGG16 | 12 000 / 2 000 | 224×224×3 | 4 | 32 | AdamW, lr 1e-4, wd 1e-4 | cosine | full fine-tune |
| AlexNet | 12 000 / 2 000 | 224×224×3 | 4 | 64 | AdamW, lr 1e-4, wd 1e-4 | cosine | full fine-tune |
| Faster R-CNN | 8 000 / 2 000 | 128×128×3 | 3 | 8 | SGD, lr 5e-3, mom 0.9, wd 5e-4 | cosine | COCO weights, 11 classes |

Why the budgets differ: VGG16 at 224 px costs ≈0.5 s per 32-image step on a T4, so the full
60 k split would need hours; the 12 k subset and 2 k test set keep each stage inside a free
session. The CNN and ViT see all 60 k images. The detector sees 8 k canvases, which is
plenty for a single centred digit. Results are therefore compared **at comparable
Colab-feasible budgets**, not in a strictly controlled benchmark — this is stated again in
the limitations.

```bash
# local CPU smoke test (no GPU needed)
python src/train.py --model cnn --epochs 1 --max-batches 2 \
  --train-subset 512 --test-subset 256 --num-workers 0

# full runs on a Colab T4, from WSL
bash scripts/colab_run.sh                 # setup + cnn, vit, frcnn, vgg16, alexnet
STAGES="cnn vit" REUSE=1 bash scripts/colab_run.sh run
python scripts/make_figures.py            # rebuild every figure from results/*.json
```

The Colab driver uses a session named `lab2`, uploads the code as a small tarball, runs one
stage per `colab exec`, gates on the `[done] <stage>` marker (the CLI can exit 0 even when
the cell raised), and downloads `results/<stage>.json` immediately.

---

## 3. Part 1.1 — CNN classifier from scratch

### Architecture

| Layer | Configuration | Output |
|---|---|---|
| Conv2d | 1→32, kernel 3, stride 1, padding 1 | 32×28×28 |
| BatchNorm + ReLU | — | 32×28×28 |
| Conv2d | 32→32, kernel 3, stride 1, padding 1 | 32×28×28 |
| BatchNorm + ReLU | — | 32×28×28 |
| MaxPool2d | kernel 2, stride 2 | 32×14×14 |
| Conv2d | 32→64, kernel 3, stride 1, padding 1 | 64×14×14 |
| BatchNorm + ReLU | — | 64×14×14 |
| Conv2d | 64→64, kernel 3, stride 1, padding 1 | 64×14×14 |
| BatchNorm + ReLU | — | 64×14×14 |
| MaxPool2d | kernel 2, stride 2 | 64×7×7 |
| Flatten | — | 3 136 |
| Linear + ReLU + Dropout | 3 136→128, dropout 0.25 | 128 |
| Linear | 128→10 | 10 |

Regularisation is BatchNorm after every convolution plus dropout before the classifier;
weight decay 1e-4 is the second regulariser. The model has **468 202** parameters.

### Results

| Metric | Value |
|---|---:|
| Test accuracy | **99.52 %** |
| Macro F1 | **99.52 %** |
| Final training loss | 0.0061 |
| Training time (8 epochs) | 114.5 s (14.3 s/epoch) |
| Inference | 0.186 ms/image |
| Best epoch | 8 |

The CNN reaches 98.41 % after the **first** epoch and 99.52 % after eight. Per-class F1 is
between 0.992 (digit 9) and 0.998 (digit 0); the remaining errors concentrate on digits 9,
7 and 5 (per-class F1 0.992–0.995), not on a class-imbalance effect (MNIST is balanced).

![Training curves](results/figures/fig_training_curves.png)

---

## 4. Part 1.2 — Faster R-CNN on canvas-MNIST

### Reformulation

Faster R-CNN is a detector, so a classification dataset cannot be fed to it unchanged.
Each MNIST digit is pasted on a 128×128 black canvas at a seeded random position and the
digit's rectangle is the target box. There is exactly one object per image, with 10 digit
classes plus background (11 outputs). This keeps the MNIST content while adding the
localisation supervision the architecture expects.

### Architecture and training

* `fasterrcnn_mobilenet_v3_large_fpn`, COCO-pretrained, with a new
  `FastRCNNPredictor` (11 classes).
* Input transform fixed to 128×128 (`min_size = max_size = 128`).
* RPN + RoIAlign + box head; loss = classification + box regression + objectness +
  RPN box regression.
* SGD, lr 5e-3, momentum 0.9, weight decay 5e-4, cosine schedule, 3 epochs, batch 8.
* 18.9 M trainable parameters.

### Results

| Metric | Value |
|---|---:|
| mAP@0.5 | **0.976** |
| Detection accuracy (digit + IoU ≥ 0.5) | **93.65 %** |
| Label accuracy (digit only) | 93.65 % |
| Macro F1 over digits | 93.61 % |
| Mean IoU of the top box | 0.895 |
| Final loss (sum of 4 terms) | 0.778 |
| Training time (3 epochs, 8 k) | 281 s (93.7 s/epoch) |
| Inference | 6.88 ms/image |

Final loss decomposition: classifier 0.289, box regression 0.478, objectness 0.010,
RPN box regression 0.0007. Localisation is essentially solved on this easy canvas task
(mAP 0.976, mean IoU 0.895); the residual 6.35 % are digit-classification errors, which is
why top-1 detection accuracy coincides with label accuracy. mAP is slightly higher than
top-1 accuracy because it can recover a correct-class box even when that box is not the
highest-scoring one.

![Faster R-CNN curves](results/figures/fig_frcnn.png)

Per-class AP@0.5 ranges from 0.954 (digit 2) to 0.990 (digit 1); digits 2, 8, 9 and 7 are
the hardest — the same digits that confuse the classifiers.

---

## 5. Part 1.3 — CNN vs Faster R-CNN

| Metric | CNN | Faster R-CNN | Comment |
|---|---:|---:|---|
| Task | classification | detection | not the same problem |
| Accuracy | 99.52 % | 93.65 %* | 5.9 pp gap |
| Macro F1 | 99.52 % | 93.61 %* | same ordering |
| Loss | 0.0061 (CE) | 0.778 (4-term sum) | different scales |
| Training time | 114 s / 8 epochs / 60 k | 281 s / 3 epochs / 8 k | ≈18× more per sample |
| Parameters | 0.47 M | 18.9 M | 40× larger |
| Inference | 0.19 ms/img | 6.88 ms/img | 37× slower |

\* Top-1 detection metrics; mAP@0.5 is 0.976.

**Reading.** The detector is doing more work — it must also propose and regress a box — and
it is trained on a harder objective with far fewer images. Its 93.65 % is therefore not a
failure of Faster R-CNN; it is the cost of solving localisation and classification jointly.
On this dataset the localisation part is trivial (the digit is always a 28×28 blob on a
black canvas), so a detector buys nothing: the CNN classifier is more accurate, 40× smaller
and 37× faster. A detector becomes the right tool when objects can appear at different
scales, overlap, or when the box itself is the product (counting, cropping, tracking).
This is the main lesson of Part 1.3.

---

## 6. Part 1.4 — Fine-tuning VGG16 and AlexNet

Both models keep their ImageNet backbone. MNIST is resized to 224×224, replicated to three
channels and normalised with the ImageNet mean/std; the final classifier layer is replaced
by a 10-way linear layer and **all** parameters are fine-tuned (no frozen layers) with a
small learning rate.

| Metric | VGG16 | AlexNet |
|---|---:|---:|
| Pretrained | ImageNet-1k | ImageNet-1k |
| Parameters | 134.3 M | 57.0 M |
| Train / test | 12 000 / 2 000 | 12 000 / 2 000 |
| Epochs | 4 | 4 |
| Test accuracy | **99.25 %** | **99.00 %** |
| Macro F1 | 99.24 % | 98.99 % |
| Final loss | 0.0036 | 0.0077 |
| Training time | 793 s (198 s/epoch) | 107 s (27 s/epoch) |
| Inference | 4.93 ms/img | 1.71 ms/img |

**Comparison with the CNN and Faster R-CNN.** VGG16 and AlexNet land 0.27 and 0.52 pp
behind the scratch CNN, at 287× and 122× the parameters and much higher latency (26× and
9× per image). They do **not** beat the small CNN. That is the expected outcome once the
domain gap is taken seriously:

* ImageNet features encode RGB textures and object parts; MNIST is grayscale, low-resolution
  and centred, with almost no texture. The pretrained filters are useful but not ideal.
* Only 12 k images and four epochs were used, and every one of the 134 M / 57 M parameters
  was updated. On a small dataset that is a lot of freedom; AlexNet's test accuracy even
  dipped at epoch 2 before recovering.
* The task is easy enough that a 0.47 M-parameter CNN with the right inductive bias already
  saturates the test set.

Fine-tuning clearly *works* (VGG16 is at 98.75 % after one epoch), so transfer learning is
not useless here — it is simply not the most efficient route. For a production MNIST-like
problem, freezing the backbone and training a linear probe, or using a smaller pretrained
network, would be the sensible next experiment.

**Conclusion of Part 1:** architecture should match the problem. A compact CNN is the best
accuracy/cost trade-off for centred 28×28 digits; a detector is overkill unless localisation
is required; large ImageNet backbones transfer but are dominated by a small task-specific
CNN on this dataset.

---

## 7. Part 2 — Vision Transformer from scratch

### Architecture

Following the [Medium tutorial](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c),
the ViT is implemented from primitive PyTorch modules (no `timm`, no pretrained weights):

| Component | Configuration |
|---|---|
| Patch embedding | Conv2d 1→128, kernel 4, stride 4 → 7×7 = 49 tokens of dim 128 |
| [CLS] token | learned, prepended to the 49 patch tokens |
| Positional encoding | learned, 50×128, added to the tokens |
| Encoder | 6 pre-norm blocks |
| Attention | 8 heads, dim 128, dropout 0.1, residual |
| MLP | 128→512→128, GELU, dropout 0.1, residual |
| Head | LayerNorm → [CLS] → Linear 128→10 |

Total: **1 199 882** parameters — about 2.6× the CNN, deliberately comparable in capacity.
Training: AdamW, lr 3e-4, weight decay 0.05, cosine schedule, batch 128, 12 epochs,
label smoothing 0.1, gradient clipping 1.0, no augmentation.

### Results

| Metric | Value |
|---|---:|
| Test accuracy | **98.07 %** |
| Macro F1 | 98.05 % |
| Final training loss | 0.528 (label-smoothed) |
| Training time (12 epochs) | 260 s (21.7 s/epoch) |
| Inference | 0.213 ms/image |
| Best epoch | 11 (98.10 %) |

The ViT starts at 80.5 % after one epoch (the CNN is already at 98.4 %) and improves
monotonically to 98.07 %, but the curve flattens after epoch 9. The training loss sits at
≈0.53, which is almost exactly the irreducible cross-entropy of a 10-class problem with
label smoothing 0.1 (≈0.54): the model has fit the training distribution as far as this
objective allows, so more epochs of the same recipe would not close the gap. The remaining
1.45 pp is a **generalisation/inductive-bias** gap, not underfitting.

![Convergence](results/figures/fig_convergence.png)

### ViT vs the models of Part 1

| Metric | CNN | ViT | ViT vs CNN |
|---|---:|---:|---|
| Test accuracy | 99.52 % | 98.07 % | −1.45 pp |
| Macro F1 | 99.52 % | 98.05 % | −1.47 pp |
| Parameters | 0.47 M | 1.20 M | 2.6× |
| Training time | 114 s | 260 s | 2.3× |
| Inference | 0.186 ms | 0.213 ms | 1.15× |
| Accuracy after 1 epoch | 98.41 % | 80.46 % | −17.9 pp |

**Interpretation.** On MNIST the CNN wins on every axis that matters: accuracy, F1,
parameter count and wall-clock time. This is the textbook consequence of inductive bias.
Convolution hard-codes locality, translation equivariance and weight sharing — exactly the
priors that handwritten digits satisfy — so the CNN learns from few examples and converges
in one epoch. The ViT must *learn* those priors from data through global self-attention;
with 49 tokens, no augmentation and 60 k images it gets close (98 %) but not equal, and it
needs 2.3× the compute. The gap is much smaller than the "ViT needs ImageNet-scale data"
story would suggest, because MNIST is large (60 k) and low-dimensional; but the direction is
unambiguous.

The comparison also shows why ViTs took over computer vision only after large-scale
pretraining and strong augmentation (DeiT-style recipes): with enough data, global attention
scales better than convolution; at MNIST scale, the CNN's prior is worth more than the ViT's
flexibility. A fair follow-up would add random crops/rotations, train the ViT longer, or
distil the CNN into the ViT — all of which would likely close most of the 1.45 pp.

![Per-class F1](results/figures/fig_per_class_f1.png)

The per-class view is consistent across models: digit 9 is the hardest for both (ViT F1
0.971, CNN 0.992), and digit 0 is the easiest for the CNN (0.998). The ViT's weakest classes
are 9, 3, 8 and 5 — the same visually ambiguous digits, not a systematic failure of one
class.

---

## 8. Synthesis — what I learned

1. **A small CNN is still a very strong baseline.** 0.47 M parameters, 114 s on a T4 and
   99.52 % on MNIST. Architecture priors beat raw parameter count on small, structured
   images.
2. **Framing matters as much as the model.** Faster R-CNN is not "a worse classifier"; it
   solves a different problem. Recasting MNIST as detection was the only way to train it
   meaningfully, and the 93.65 % top-1 / 0.976 mAP result is a statement about the cost of
   joint localisation + classification, not about the backbone.
3. **Transfer learning is not automatically better.** VGG16 and AlexNet reached 99.25 % and
   99.00 % but with 287×/122× more parameters and much higher latency. Domain gap, dataset
   size and fine-tuning budget decide whether pretraining pays off.
4. **The ViT is a data-efficiency story.** Implemented from scratch (patch embedding, CLS
   token, learned positions, pre-norm blocks), it reached 98.07 % — close, but behind the
   CNN, and its loss plateau at the label-smoothing floor showed the gap is generalisation,
   not optimisation.
5. **Measurement discipline.** One JSON per run with config, environment, per-epoch history,
   timing and per-class metrics made the report reproducible and exposed details that a
   single accuracy number would hide (e.g. detection accuracy equals label accuracy because
   localisation is saturated on the canvas task). Running one stage per `colab exec` and
   downloading each report immediately made the free-tier Colab workflow robust.

---

## 9. Repository layout

```
lab2/
├── README.md                  # this report
├── requirements.txt
├── src/
│   ├── data.py                # MNIST classifiers + canvas detection dataset
│   ├── models.py              # SmallCNN, ViT, Faster R-CNN, VGG16, AlexNet
│   ├── metrics.py             # confusion matrix, macro F1, IoU, AP@0.5 (NumPy only)
│   ├── train.py               # unified CLI: --model cnn|vit|vgg16|alexnet|frcnn
│   └── utils.py               # seeding, device, timing, JSON, environment info
├── scripts/
│   ├── colab_run.sh           # WSL/Colab CLI driver (setup + one stage per exec)
│   ├── colab_runner.py        # remote stage definitions and execution
│   ├── colab_smoke.py         # one-batch GPU check for the pretrained models
│   ├── make_figures.py        # all figures from results/*.json
│   └── summarize.py           # compact console summary of the reports
└── results/
    ├── cnn.json, vit.json, frcnn.json, vgg16.json, alexnet.json
    └── figures/*.png
```

`data/` (MNIST download) and `logs/` (Colab session logs) are git-ignored. Every number in
this README is generated from the committed JSON reports.

### Environment actually used

```
GPU        Tesla T4 (16 GB)
torch      2.11.0+cu130
torchvision 0.26.0+cu130
Python     3.12
Colab CLI  0.7.5 (WSL)
```

---

## 10. Limitations

* **Budgets differ** between models (table in §2). The CNN/ViT see 60 k images, the
  pretrained classifiers 12 k, the detector 8 k; the pretrained models are evaluated on a
  2 k test subset. This is a Colab-feasible comparison, not a controlled ablation.
* **One seed per model.** Differences below ~0.5 pp (VGG16 vs CNN, AlexNet vs VGG16) are
  within run-to-run noise, especially on the 2 k test subset; the large effects (ViT vs CNN,
  detector vs classifier, parameter/latency ratios) are robust.
* **Faster R-CNN metrics are detection-derived** (top-scoring box, 11-class head) and are
  not directly comparable to pure classification accuracy; mAP@0.5 is reported separately.
* **The canvas task is easy.** A single centred 28×28 digit on a black background makes
  localisation almost trivial; a harder multi-digit/cluttered dataset would change the
  detector's story.
* **No augmentation** was used for the scratch models, and the ViT used label smoothing 0.1
  while the CNN/pretrained models used plain cross-entropy. Both choices are documented;
  they favour neither model in a way that would reverse the conclusions, but a fully
  controlled study would equalise them.
* `cudnn.benchmark = True` favours speed over bit-exact reproducibility; timings vary by a
  few percent between sessions.

---

## 11. References

1. Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. *Gradient-based learning applied to document
   recognition.* Proceedings of the IEEE, 1998.
2. S. Ren, K. He, R. Girshick, J. Sun. *Faster R-CNN: Towards Real-Time Object Detection with
   Region Proposal Networks.* NeurIPS, 2015.
3. K. Simonyan, A. Zisserman. *Very Deep Convolutional Networks for Large-Scale Image
   Recognition.* ICLR, 2015.
4. A. Krizhevsky, I. Sutskever, G. E. Hinton. *ImageNet Classification with Deep Convolutional
   Neural Networks.* NeurIPS, 2012.
5. A. Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition
   at Scale.* ICLR, 2021.
6. *Vision Transformers from Scratch (PyTorch): A Step-by-Step Guide*, Medium / MLearning.ai
   (the tutorial followed for Part 2).
7. torchvision documentation: `fasterrcnn_mobilenet_v3_large_fpn`, `vgg16`, `alexnet`,
   `MNIST`.
