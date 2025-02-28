# 🏥 Breast Cancer Prediction using Machine Learning

## 📌 Project Overview
This project aims to build a machine learning model to predict whether a tumor is **malignant** or **benign** using the **Breast Cancer Wisconsin Dataset**. We explored various machine learning models and fine-tuned them using **GridSearchCV** to improve performance.

> **Note:** Feature selection was not performed in this project; all features were used for training the models.

---

## 📊 Dataset Information
- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Features:** 30 numerical attributes related to tumor characteristics
- **Target Variable:**  
  - `0` → Benign  
  - `1` → Malignant  

---

## 🎯 Objective
Our goal is to **accurately classify** whether a tumor is malignant or benign using different machine learning algorithms. We evaluated multiple models and fine-tuned them to achieve the best accuracy.

---

## 🚀 Models Tested
We tested the following machine learning models:
- **Logistic Regression**
- **Support Vector Machine (Linear SVC)**
- **Random Forest Classifier**

Each model was evaluated **before and after hyperparameter tuning** to analyze the improvements.

---

## 📈 Model Performance  

| Model               | Accuracy (Before Tuning) | Accuracy (After Tuning) | Best Parameters |
|---------------------|------------------------|-------------------------|----------------|
| **Logistic Regression**  | 95.61% | 98.24% | `C=0.1, penalty='l2', solver='liblinear'` |
| **Linear SVC**  | 96.49% | 98.24% | `C=0.01, max_iter=5000` |
| **Random Forest** | 95.61% | 96.71% | `bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=400` |

---

## 📊 Precision, Recall, and F1-Score  

### **Logistic Regression (After Tuning)**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Benign) | 0.96 | 0.99 | 0.97 | 67 |
| 1 (Malignant) | 0.98 | 0.94 | 0.96 | 47 |
| **Accuracy** | **96%** |  |  | 114 |

### **Linear SVC (After Tuning)**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Benign) | 0.96 | 0.99 | 0.97 | 67 |
| 1 (Malignant) | 0.98 | 0.94 | 0.96 | 47 |
| **Accuracy** | **96%** |  |  | 114 |

### **Random Forest (After Tuning)**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Benign) | 0.98 | 0.96 | 0.97 | 67 |
| 1 (Malignant) | 0.94 | 0.98 | 0.96 | 47 |
| **Accuracy** | **96%** |  |  | 114 |

---

## 🛠️ Tools & Libraries Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib & Seaborn

---

## 📌 Key Takeaways
✔️ **Hyperparameter tuning significantly improved model performance**, especially for **Logistic Regression and Linear SVC**.  
✔️ **Random Forest performed well but did not improve as much with tuning**.  
✔️ **All features were used without feature selection**, meaning further optimization could be explored.  
✔️ **The models achieved high accuracy, making them effective for breast cancer diagnosis predictions**.  

---

## 🏆 Best Model
Based on accuracy and performance, **Logistic Regression (after tuning) was the best model**, achieving **98.24% accuracy**.

---

## 📜 Conclusion
This project successfully implemented and tuned multiple machine learning models for **breast cancer classification**. Further improvements can be made by experimenting with **feature selection**, **ensemble techniques**, and **deep learning models**.

---

## 📬 Contact
If you have any questions or suggestions, feel free to reach out! 😊  

📧 **Email:** souravkumarr77@gmail.com  
🔗 **LinkedIn:** [Sourav Kumar](https://www.linkedin.com/in/sourav-kumar-30141b174/)  
🔗 **X:** [Sourav Kumar](https://x.com/souravkumarr73)  

---

🔗 **Feel free to contribute or suggest improvements!** 😊  
