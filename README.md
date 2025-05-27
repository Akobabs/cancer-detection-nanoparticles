---
# 🧬 Cancer Detection Using Nanoparticles
This repository presents a machine learning-powered system for breast cancer detection, simulating the effect of nanoparticle-enhanced biosensor data using the Wisconsin Breast Cancer (Diagnostic) Dataset. The project includes:

* 🔍 A Jupyter Notebook for exploratory data analysis, model training, and evaluation.
* 🌐 A Streamlit web app for interactive predictions and visualization of model performance.

---

## 🚀 Project Overview

This project aims to classify breast cancer cells as **malignant** or **benign** using 30 features extracted from fine needle aspirate (FNA) images, simulating nanoparticle-based enhancements.

### 🧾 Key Components

* **Dataset**: Wisconsin Breast Cancer (Diagnostic) via [`ucimlrepo`](https://pypi.org/project/ucimlrepo/).
* **Models**:

  * Support Vector Machine (SVM)
  * Random Forest
  * Neural Network (MLP)
    Trained in Google Colab with optional GPU support.
* **Frontend**: Streamlit app to:

  * Upload data or enter manually
  * Make predictions
  * View performance metrics and charts
* **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, Mean Squared Error (MSE), and ROC-AUC.
* **Deployment Options**: Local, Streamlit Cloud, and Docker.

---

## 📁 Repository Structure

```text
cancer-detection-nanoparticles/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # Raw dataset (wisconsin_breast_cancer.csv)
│   └── processed/            # Placeholder for processed data
├── models/
│   ├── trained_models/       # Saved models (e.g., svm_model.joblib)
│   └── model_scripts/        # Placeholder for training scripts
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── app.py                # Streamlit frontend
├── notebooks/
│   ├── cancer_detection_eda.ipynb
│   ├── figures/
│   └── results/
├── tests/                    # Placeholder for unit tests
└── config/
    └── docker_config/        # Dockerfile and related config
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Akobabs/cancer-detection-nanoparticles.git
cd cancer-detection-nanoparticles
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
# For Linux/macOS
source venv/bin/activate
# For Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

📦 Includes: `ucimlrepo`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `streamlit`, `plotly`

---

## 📦 Prepare Required Artifacts

### ✅ Download and place the following files:

| Artifact                           | Location                                 |
| ---------------------------------- | ---------------------------------------- |
| `scaler.joblib`, model files       | `models/trained_models/`                 |
| `model_performance.csv`            | `notebooks/results/`                     |
| `performance_chart.json`           | `notebooks/results/`                     |
| `model_performance_comparison.png` | `notebooks/figures/`                     |
| `wisconsin_breast_cancer.csv`      | `data/raw/` or auto-fetched via notebook |

---

## 🧪 Run the Notebook (Optional)

Open the notebook in [Google Colab](https://colab.research.google.com/):

```bash
notebooks/cancer_detection_eda.ipynb
```

* Change runtime type to GPU for faster training
* Run all cells to fetch data, explore features, train models, and export artifacts

---

## 💻 Run the Streamlit App Locally

```bash
streamlit run app.py
```

* Open your browser at `http://localhost:8501`

### 🌟 App Features:

* Upload CSV with 30 features or enter values manually
* Choose prediction model: SVM, Random Forest, or Neural Network
* View predictions (Malignant/Benign)
* Display metrics and performance charts

#### 📄 Sample CSV Format:

```csv
radius1,texture1,perimeter1,area1,...,fractal_dimension3
17.99,10.38,122.8,1001.0,...,0.07871
```

---

## ☁️ Deployment Options

### ✅ Streamlit Cloud

1. Push your repository to GitHub

```bash
git add .
git commit -m "Deploy Streamlit app"
git push origin main
```

2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Create a new app linked to your GitHub repo
4. Set the main file as `src/app.py`
5. Ensure `requirements.txt` is in the root directory
6. Deploy and get a public link

> 🔗 Note: If models or results are not in the repo, host them using Google Drive or other cloud storage and adjust `app.py` paths.

---

### 🐳 Docker (Optional)

```bash
# Build the Docker image
docker build -t cancer-detection-nanoparticles .

# Run the container
docker run -p 8501:8501 cancer-detection-nanoparticles
```

Access at: `http://localhost:8501`

---

## 📊 Artifacts Summary

| Type         | Location                                   |
| ------------ | ------------------------------------------ |
| Models       | `models/trained_models/`                   |
| Dataset      | `data/raw/wisconsin_breast_cancer.csv`     |
| Results      | `notebooks/results/model_performance.csv`  |
| Visuals      | `notebooks/figures/`                       |
| Chart Config | `notebooks/results/performance_chart.json` |

---

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch:

```bash
git checkout -b feature-name
```

3. Commit your changes:

```bash
git commit -m "Add feature-name"
```

4. Push to GitHub:

```bash
git push origin feature-name
```

5. Open a pull request

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

* **Dataset**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)
* **Libraries**: `scikit-learn`, `streamlit`, `plotly`, `joblib`, `matplotlib`, `seaborn`, `ucimlrepo`
* **Tools**: Google Colab for training, Streamlit for deployment

---

## 📬 Contact

For questions, suggestions, or bug reports, open an [issue](https://github.com/Akobabs/cancer-detection-nanoparticles/issues) or contact the repository maintainer via GitHub.

---
