# CreditLens AI — Credit Card Default Prediction Platform 💳⚡

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://creditlens-ai-dbku.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![UCI ML Repo](https://img.shields.io/badge/Dataset-UCI%20%23350-brightgreen)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

👉 **Live Application:** **[https://creditlens-ai-dbku.onrender.com](https://creditlens-ai-dbku.onrender.com)**

An end-to-end Machine Learning web application predicting credit card default probability in real-time. Built with a 3-layer **Artificial Neural Network (ANN)** in TensorFlow/Keras, served via high-performance **FastAPI**, and packaged with a modern, consumer-grade fintech UI inspired by Dribbble design standards.

---

## 🌟 Key Features

- **Deep Learning Model (ANN)**: Evaluates 23 multi-dimensional indicators (repayment timeliness across 6 months, credit ceiling, billing statements, and demographic metrics) trained on 30,000 real client records.
- **Real-Time Live Inference**: FastAPI backend delivering sub-15ms predictions with Pydantic schema validation.
- **Interactive UI/UX**:
  - Dribbble-inspired warm peach aesthetic with frosted glassmorphism.
  - Quick-start persona presets (Low-Risk Client, Revolving Credit, Delinquent Risk).
  - SVG dynamic radial risk gauge with animated needle and probability score.
  - Delinquency & credit utilization diagnostic breakdown.
  - 250ms debounced auto-predict on input changes.

---

## 🧠 Neural Network Architecture

```
Input Layer (23 Features)
       ↓
Dense Layer (32 Neurons, ReLU Activation)
       ↓
Dense Layer (16 Neurons, ReLU Activation)
       ↓
Output Layer (1 Neuron, Sigmoid Activation) → Default Probability [0.0 - 1.0]
```

- **Preprocessing**: `StandardScaler` (Z-score normalization)
- **Loss Function**: `binary_crossentropy`
- **Optimizer**: `adam`
- **Evaluation**: Stratified 80/20 train-test split (`random_state=44`)
- **Test Accuracy**: **~81%**

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/sahal9229/Creditlens-ai.git
cd Creditlens-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run or Retrain the Model (Optional)
```bash
python train_model.py
```

### 4. Launch the Web Application
```bash
python -m uvicorn main:app --reload --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser!

---

## 📁 Project Structure

```
├── Credit Card.ipynb        # Jupyter Notebook: Data exploration, preprocessing & ANN training
├── credit_ann_model.keras   # Trained Keras neural network weights
├── scaler.pkl               # Fitted scikit-learn StandardScaler
├── train_model.py           # UCI dataset ingestion & ANN training script
├── main.py                  # FastAPI REST API & static file server
├── requirements.txt         # Project dependencies
├── static/                  # Modern Fintech UI
│   ├── index.html           # Structure & interactive components
│   ├── styles.css           # Dribbble-inspired warm peach theme
│   ├── app.js               # Live inference controller & gauge logic
│   └── images/              # 3D fintech illustrations & assets
└── README.md
```

---

## 👨‍💻 Author

**Muhammed Sahal KK**
- 🎓 Data Scientist · AI & Machine Learning Student
- 🌐 GitHub: [@sahal9229](https://github.com/sahal9229)
- 💼 LinkedIn: [Muhammed Sahal K K](https://www.linkedin.com/in/muhammed-sahal-k-k/?isSelfProfile=true)

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
