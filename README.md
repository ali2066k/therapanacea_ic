# TheraPanacea Binary Classifier

This project implements a binary image classifier using deep learning. It supports training with multiple architectures, configurable loss functions (`bce` vs `bcewithlogits`), automatic model selection, early stopping, and submission-ready prediction.

---

## 📦 Project Structure

```
herapanacea_ic/
├── data/               # Dataset, transforms, utility functions
│   ├── dataset.py
│   └── utils.py
├── engine/             # Training loop and metric logic
│   └── trainer.py
├── models/             # Model architecture (e.g. ResNet, MobileNet)
│   └── classifier.py
├── main.py             # CLI: training script
├── predict.py          # CLI: generate label_val.txt for submission
├── requirements.txt    # Dependencies
└── README.md           # Project documentation (this file)
```

---

## 🛠 Installation

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

Or use conda to create an isolated environment:

```bash
conda create -n therapanacea python=3.10
conda activate therapanacea
pip install -r requirements.txt
```

---

## 🚀 Training

```bash
python main.py \
  --image_dir ./ml_exercise_therapanacea/train_img \
  --label_file ./ml_exercise_therapanacea/label_train.txt \
  --arch resnet50 \
  --loss bcewithlogits \
  --epochs 50 \
  --gpu 1 \
  --batch_size 1024 \
  --lr 1e-4 \
  --early_stop 7 \
  --save_model ./models/last_model_gpu1.pth \
  --save_best_model ./models/best_model_gpu1.pth \
  --save_plot ./plots/plot_gpu1.png \
  --save_hter_plot ./plots/hter_gpu1.png
```

> `--loss` can be `bce` (with sigmoid in model) or `bcewithlogits` (recommended, more stable).

---

## 🔍 Prediction for Submission

Once the best model is trained, generate the required `label_val.txt`:

```bash
python predict.py \
  --model_path ./models/best_model_gpu1.pth \
  --val_img_dir ./ml_exercise_therapanacea/val_img \
  --arch resnet50 \
  --batch_size 1024 \
  --output_file label_val.txt \
  --loss bcewithlogits
```

This will generate a file named `label_val.txt` with 20,000 lines — one prediction per test image, in order.

---

## 🧾 Outputs

- `models/last_model_gpu1.pth`: Final model after all epochs
- `models/best_model_gpu1.pth`: Model with lowest validation HTER
- `plots/plot_gpu1.png`: Training metrics over epochs (loss, acc, F1)
- `plots/hter_gpu1.png`: Validation HTER vs epoch
- `label_val.txt`: Your submission predictions

---
