# 🤖 Fake Social Media Account Detection

A Machine Learning-based web application that detects **fake Instagram accounts** using profile-based features. The application analyzes publicly available account information and predicts whether an account is **Real** or **Fake** using trained machine learning models.

---

## 📌 Project Overview

Fake social media accounts are commonly used for spam, scams, misinformation, and fraudulent activities. This project aims to identify fake Instagram accounts by analyzing profile attributes such as follower count, following count, number of posts, profile picture, biography, and other profile-based features.

The application provides an interactive interface built with **Streamlit**, allowing users to enter account details and receive real-time predictions.

---

## ✨ Features

- Detect Fake and Genuine Instagram Accounts
- Machine Learning-Based Classification
- Random Forest and XGBoost Models
- Data Preprocessing and Feature Engineering
- Interactive Web Interface using Streamlit
- Real-Time Prediction
- Confidence Score for Predictions

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---

## 📂 Project Structure

```text
fake-social-media-account-detection
│
├── data
├── model
├── profile_pics
├── screenshots
├── utils
├── app1.py
├── fetch_profile.py
├── loader.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The project uses two datasets for model training and evaluation:

- **users.csv** – Genuine Instagram Accounts
- **fusers.csv** – Fake Instagram Accounts

The datasets are cleaned, preprocessed, and transformed before training the machine learning models.

---

## 🧠 Machine Learning Models

The following supervised learning algorithms were implemented:

- Random Forest Classifier
- XGBoost Classifier

The best-performing model is used for prediction in the Streamlit application.

---

## ⚙️ Workflow

1. Load the Dataset
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Train Machine Learning Models
5. Evaluate Model Performance
6. Predict Account Type
7. Display Results using Streamlit

---

## ▶️ Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/Nipun-wani/fake-social-media-account-detection.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app1.py
```

---

## 📈 Prediction Output

The application predicts whether an Instagram account is:

- ✅ Genuine Account
- ❌ Fake Account

along with the prediction confidence score.

---

## 📷 Application Screenshots

### Home Page

![Home Page](screenshots/screenshot1.png)

---

### Profile  Information

![Profile  Information](screenshots/screenshot2.png)

---

### Prediction

![Prediction](screenshots/screenshot3.png)

---

### Prediction

![Prediction](screenshots/screenshot4.png)

---

## 🚀 Future Enhancements

- Deep Learning Models
- Dashboard & Analytics
- Explainable AI (SHAP/LIME)
- Cloud Deployment
- User Authentication

---

## 👨‍💻 Author

**Nipun Wani**

GitHub: https://github.com/Nipun-wani

---
