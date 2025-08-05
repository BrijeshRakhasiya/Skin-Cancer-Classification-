# 🧪 Skin Cancer Image Classification 

This project focuses on classifying different types of skin lesions using a deep learning model trained on the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

## 📂 Dataset

- **Source:** Kaggle - [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **File Used:** `hmnist_28_28_L.csv`
- **Classes:**
  - `nv` (Melanocytic nevi)
  - `mel` (Melanoma)
  - `bkl` (Benign keratosis-like lesions)
  - `bcc` (Basal cell carcinoma)
  - `akiec` (Actinic keratoses and intraepithelial carcinoma)
  - `vasc` (Vascular lesions)
  - `df` (Dermatofibroma)

Each image is grayscale, size `28x28`, and flattened into 784 pixels.

---

## 🧠 Model Overview

- A deep learning classifier (CNN/MLP as per notebook) was trained on the dataset.
- Data preprocessing includes normalization and label encoding.
- The dataset is imbalanced, which affects performance on minority classes.

---

## 📊 Model Evaluation

Here is the classification performance of the model:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| nv    | 0.19      | 0.06   | 0.09     | 69      |
| mel   | 0.48      | 0.12   | 0.19     | 93      |
| bkl   | 0.35      | 0.22   | 0.27     | 228     |
| bcc   | 0.00      | 0.00   | 0.00     | 28      |
| akiec | 0.74      | 0.96   | 0.83     | 1338    |
| vasc  | 0.00      | 0.00   | 0.00     | 21      |
| df    | 0.46      | 0.15   | 0.22     | 226     |

- **Overall Accuracy:** 69%
- **Macro Avg F1-Score:** 0.23
- **Weighted Avg F1-Score:** 0.62

🔍 _Note: The model performs very well on the dominant class `akiec`, but poorly on minority classes due to class imbalance._

---

## 🛠 Technologies Used

- Python
- NumPy, Pandas
- TensorFlow / Keras
- Scikit-learn
- Matplotlib, Seaborn

---

## ⚠️ Challenges

- **Class Imbalance:** Heavily skewed class distribution (e.g., `akiec` has ~67% of all samples).
- **Low Image Resolution:** 28x28 grayscale images may limit model accuracy.

---

## 🚀 Future Work

- Apply data augmentation or SMOTE for balancing.
- Use original RGB images instead of downsampled 28x28 versions.
- Try advanced architectures like ResNet or EfficientNet.
- Incorporate transfer learning on higher-resolution images.

---

## 📁 How to Run

```bash
git clone https://github.com/BrijeshRakhasiya/Skin-Cancer-Classification-.git
cd skin-cancer-classification
jupyter notebook skin-cancer-image-classification.ipynb

```


# 🙋‍♂️ Author
Brijesh Rakhasiya
AI/ML Enthusiast | Data Scientist | Problem Solver


## 📄 License

This project is licensed under the MIT License.

---
**Made ❤️ by Brijesh Rakhasiya**
