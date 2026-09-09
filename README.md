# EsoPredict: Multimodal Deep Learning for Esophageal Cancer Status & Survival Prediction

> Predicting esophageal cancer status and survival outcomes by fusing clinical, endoscopic imaging, and genomic data — with explainable AI (SHAP & LIME) at every step.

## 📌 Project Description

**EsoPredict** is an end-to-end deep learning pipeline that tackles esophageal cancer prediction from four complementary angles:

| Model | Input Data | Task | Architecture |
|---|---|---|---|
| **Clinical** | TCGA clinical biomarkers & demographics | Tumor status (Tumor Free vs. With Tumor) | Dense Neural Network (Keras) |
| **Imaging** | Endoscopy images | Esophagus abnormality detection | EfficientNetB0 (Transfer Learning) |
| **Genomic** | TCGA PanCancer Atlas 2018 (ESCA) clinical/genomic markers | Overall survival status (Living vs. Deceased) | Dense Neural Network with feature selection |
| **Multimodal** | Clinical + Imaging (fused) | Tumor status (joint prediction) | Dual-branch fusion network (Dense + EfficientNetB0) |

Each model notebook follows a consistent workflow — data loading, cleaning, class-balance visualization, training with early stopping/LR scheduling, evaluation (accuracy, AUC, confusion matrix), and **explainability** via **SHAP** (global feature importance) and **LIME** (per-patient local explanations) — so predictions aren't just accurate, they're interpretable for clinical review.

A dedicated `Dataset_Preparation.ipynb` notebook builds the multimodal dataset by splitting clinical and imaging data independently (to avoid leakage) and pairing them by class label into train/val/test sets.

---

## 🗂️ Repository Structure
EsoPredict/
│
├── README.md
├── LICENSE
│
├── notebooks/
│ ├── 01_Clinical_Model.ipynb # Clinical biomarker → tumor status classifier
│ ├── 02_Imaging_Model.ipynb # Endoscopy image → tumor status classifier (EfficientNetB0)
│ ├── 03_Genomic_Model.ipynb # TCGA genomic/clinical → survival status classifier
│ ├── 04_Multimodal_Model.ipynb # Fused clinical + imaging classifier
│ └── Dataset_Preparation.ipynb # Builds multimodal train/val/test splits

> Raw datasets, trained models, and generated reports are not included in this repository due to size and licensing. See the Datasets section below for sources.

---

## 📊 Datasets

- **Clinical Data**: Esophageal cancer patient biomarkers and demographics, target = `person_neoplasm_cancer_status`.
- **Genomic Data**: [ESCA TCGA PanCancer Atlas 2018](https://www.cbioportal.org/study/summary?id=esca_tcga_pan_can_atlas_2018) clinical/genomic dataset, target = `Overall Survival Status`.
- **Imaging Data**: Endoscopy images labeled `esophagus` (abnormal) vs. `no-esophagus` (normal).
- **Multimodal Data**: Clinical records paired with endoscopy images by class, generated via `Dataset_Preparation.ipynb`.

---

## ⚙️ Setup

Key dependencies: `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `shap`, `lime`, `seaborn`, `matplotlib`, `tqdm`

```bash
pip install tensorflow scikit-learn pandas numpy shap lime seaborn matplotlib tqdm
```

---

## 🚀 Usage

1. Run `notebooks/Dataset_Preparation.ipynb` first if you plan to train the multimodal model — it generates the paired train/val/test splits.
2. Run any of the four model notebooks independently:
   - `01_Clinical_Model.ipynb`
   - `02_Imaging_Model.ipynb`
   - `03_Genomic_Model.ipynb`
   - `04_Multimodal_Model.ipynb`
3. Each notebook trains, evaluates, and generates SHAP/LIME explainability outputs inline.

---

## 🔍 Explainability

Every model includes:
- **SHAP** global summary plots — which features/pixels matter most across the dataset.
- **LIME** local explanations — why the model made a specific prediction for one patient (tabular features and/or image superpixels).

---

## 📈 Evaluation

Each notebook reports:
- Accuracy & ROC-AUC
- Precision / Recall / F1 (classification report)
- Confusion matrix heatmap
- Training/validation accuracy & loss curves

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Clinical & genomic data: [TCGA PanCancer Atlas](https://www.cancer.gov/tcga) via cBioPortal
- Imaging backbone: EfficientNetB0 (ImageNet pretrained weights)
