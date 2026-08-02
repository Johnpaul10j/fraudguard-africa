# 🛡️ FraudGuard Africa

**Real-time Mobile Money Fraud Detection System**

FraudGuard Africa is an end-to-end machine learning system designed to detect fraudulent mobile money transactions in real-time. The project focuses on the African fintech ecosystem and is built to be practical, explainable, and production-ready.

## 🚀 Live Demo

Try the live application here:  
**[FraudGuard Africa Live Demo](https://fraudguard-africa-gh7cmpkqmwbmynndfnqyr2.streamlit.app/)**
---

## 📌 Project Overview

Mobile money fraud is a growing problem across Africa. This project uses machine learning to detect suspicious transactions with high accuracy while keeping false positives under control.

**Key Features:**
- Real-time fraud prediction
- Feature engineering tailored for mobile money transactions
- FastAPI backend for production use
- Modern Streamlit dashboard for easy testing
- Trained on the PaySim mobile money dataset

---

## 🛠️ Tech Stack

- **Language:** Python
- **Machine Learning:** XGBoost
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Model Persistence:** Joblib

---

## 📁 Project Structure

```bash
fraudguard-africa/
│
├── data/                       # Dataset
├── models/                     # Trained model files
├── notebooks/                  # Exploration & training notebooks
├── src/
│   ├── main.py                 # FastAPI application
│   └── dashboard.py            # Streamlit dashboard
├── logs/                       # Prediction logs
├── requirements.txt
└── README.md
```
## How to Run the Project
### Dataset
The dataset is not included in this repository because of its size.

Download the PaySim dataset from Kaggle:
https://www.kaggle.com/datasets/ealaxi/paysim1

Place the CSV file inside the `data/` folder.

1. Clone the repository
```bash
git clone https://github.com/Johnpaul10j/fraudguard-africa.git
cd fraudguard-africa
```
2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Run the FastAPI
```Bash
uvicorn src.main:app --reload
Then open: http://127.0.0.1:8000/docs
```
5. Run the Dashboard
```bash
streamlit run src/dashboard.py
```
## 📊 Model Performance
Trained on the full PaySim dataset with the following results:
MetricScoreFraud: 
Recall 99.39%
Fraud Precision 84.18%
PR-AUC 0.9869
ROC-AUC 0.9989

## 🧠 Key Features Used
Balance difference (origin & destination)
Amount-to-balance ratios
Transaction type encoding
Time-based features (hour, night transactions)
Customer transaction frequency
Risk indicators (high amount + CASH_OUT)


## 🎯 Future Improvements
Model monitoring and drift detection
Docker containerization
User authentication
Deployment to cloud (Render / Railway / AWS)
Improved threshold tuning for better precision-recall balance


## 👤 Author
Johnpaul

Data Science & AI Engineering Enthusiast

## 📄 License
This project is open source and available under the MIT License.
