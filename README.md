# Multiclass Diabetes Classification with One-vs-All Logistic Regression

This project implements a **multiclass classification system** for diabetes status prediction using **One-vs-All (OvA) Logistic Regression**, trained **from scratch** with gradient descent.  
The objective is to classify individuals into **non-diabetic**, **pre-diabetic**, and **diabetic** categories based on clinical features.

Unlike many diabetes prediction projects that focus on **binary classification**, this work explicitly addresses the **multiclass setting** and analyzes class-level performance and limitations.

---

## Problem Definition

Given a clinical dataset containing numerical health indicators, the task is to predict a patient’s diabetes status as one of three classes:

- **Class 0:** Non-diabetic  
- **Class 1:** Pre-diabetic  
- **Class 2:** Diabetic  

This is formulated as a **multiclass supervised learning problem**.

---

## Methodology

### One-vs-All Logistic Regression
- Implemented **OvA logistic regression** by training one binary classifier per class.
- Each classifier learns to distinguish *one class vs all others*.
- Final prediction is obtained by selecting the class with the highest predicted probability.

### Model Implementation
- Logistic regression implemented **from scratch** (no sklearn model fitting).
- Gradient descent used for optimization.
- Feature normalization applied to improve convergence.
- Cross-entropy loss used as the objective function.

---

## Evaluation

Model performance is evaluated using:
- **Overall accuracy**
- **Confusion matrix** (class-level analysis)
- **Cost (loss) vs iterations plot** to verify convergence

Special attention is given to **class imbalance**, particularly reduced recall for the pre-diabetic class, and its effect on classification quality.

---

## Results & Observations

- The model successfully learns decision boundaries for multiclass prediction.
- Strong performance on majority classes.
- Reduced recall on minority classes highlights limitations of linear models under class imbalance.
- Results motivate future improvements such as:
  - Class weighting
  - Regularization
  - More expressive models

---

## Tools & Technologies

- **Python**
- **NumPy**, **Pandas**

---

## Key Learning Outcomes

- Practical implementation of **multiclass classification**
- Understanding of **One-vs-All strategy**
- Gradient descent optimization from first principles
- Model evaluation beyond accuracy (confusion matrix analysis)
- Limitations of linear models in imbalanced datasets

---

## Possible Extensions

- Compare against sklearn’s OvR logistic regression
- Apply class weighting or resampling
- Extend to softmax (multinomial logistic regression)
- Evaluate with cross-validation

---


## Acknowledgements
Special thanks to [Yasser Hessein](https://www.kaggle.com/yasserhessein) for providing the
[Multiclass Diabetes Dataset](https://www.kaggle.com/datasets/yasserhessein/multiclass-diabetes-dataset),
which made this project possible.
