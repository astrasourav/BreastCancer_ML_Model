# 📌 Breast Cancer Prediction using Machine Learning

## 📝 Project Overview
This project focuses on predicting whether a tumor is **malignant** or **benign** using the **Breast Cancer Wisconsin (Diagnostic) dataset**. We tested multiple machine learning models, tuned their hyperparameters, and evaluated their performance to determine the best-performing model.


> **Note:** Feature selection was not performed in this project; all features were used for training the models.



## 📊 Dataset Description
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** Scikit-Learn `https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data`
- **Features:** 30 numerical attributes related to tumor characteristics
- **Target:**
  - `0` - Malignant (Cancerous)
  - `1` - Benign (Non-Cancerous)

## 🎯 Objective
To build a predictive model that accurately classifies tumors as **malignant** or **benign** using machine learning algorithms.

## 🛠️ Models Tested
I evaluated three different models:
1. **Logistic Regression**
2. **Linear SVC (Support Vector Classifier)**
3. **Random Forest Classifier**

Each model was first trained with default hyperparameters and then tuned using **GridSearchCV** for optimal performance.

## 🔍 Performance Comparison
Below is a comparison of the models **before** and **after** hyperparameter tuning:

| Model                 | Accuracy (Before) | Accuracy (After) | Precision | Recall | F1-Score |
|-----------------------|------------------|-----------------|-----------|--------|----------|
| **Logistic Regression** | 0.9825           | 0.9825          | 0.97      | 0.96   | 0.96     |
| **Linear SVC**        | 0.9649           | 0.9824          | 0.97      | 0.96   | 0.96     |
| **Random Forest**     | 0.9561           | 0.9671          | 0.97      | 0.96   | 0.97     |

## 🔧 Hyperparameter Tuning
I fine-tuned the models using **GridSearchCV**:
- **Logistic Regression:** Best `C = 0.1`, `solver = 'liblinear'`
- **Linear SVC:** Best `C = 0.01`, `max_iter = 5000`
- **Random Forest:** Best `n_estimators = 400`, `max_depth = None`, `bootstrap = False`

## 📌 Key Findings
- **Logistic Regression** and **Linear SVC** achieved the highest accuracy of **98.24%**.
- **Random Forest** also performed well, with **96.71%** accuracy.
- Hyperparameter tuning significantly improved the models' performance.

## 🚀 Conclusion
This project demonstrated the effectiveness of machine learning in **early breast cancer detection**. The **Logistic Regression** and **Linear SVC** models proved to be the best, offering high accuracy and reliability.

## 📂 How to Run the Project
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/breast-cancer-prediction.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script to train and evaluate models.

---
💡 **Early detection of breast cancer can save lives!** This project aims to support medical research using machine learning. If you found this helpful, consider ⭐ starring the repo!

📩 Feel free to contribute or suggest improvements!
