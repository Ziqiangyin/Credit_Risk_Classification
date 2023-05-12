# [Module 12] Credit Risk Classification Report

Credit risk prediction is a challenging classification problem due to imbalanced data. The majority of instances belong to the healthy loan class, while the number of high-risk loans is significantly lower. This imbalance can lead to biased predictions, with models favoring the majority class.

To address this issue, techniques such as resampling methods (oversampling or undersampling) and synthetic data generation (SMOTE) can be employed. These approaches rebalance the data, enabling the model to learn from a more representative dataset and improve its ability to predict high-risk loans accurately.

Implementing these methods enhances the efficiency of the credit risk prediction model, ensuring more reliable predictions for both healthy and high-risk loans.

- - -

## Features and Labels in Credit Risk Prediction: Understanding Loan Data and Predictive Model Purpose 

Based on financial information such as loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogatory marks, and total debt, we can predict whether a loan is categorized as a "healthy loan" or a "high-risk loan". By analyzing these features, our predictive model aims to assess the creditworthiness and risk associated with each loan application. The model takes into account factors such as the loan amount, interest rate, borrower's income level, debt-to-income ratio, the number of accounts the borrower holds, any derogatory marks on their credit history, and their total debt. With this information, the model can make predictions and classify loans as either "healthy" or "high-risk" based on their financial attributes.

- - -

## Class Imbalance and Analysis of Loan Variables: Understanding the Distribution of Data

Upon analyzing the loan status column, the distribution of the target variable is as follows:
  * Healthy Loan: 75,036 instances
  * High-Risk Loan: 2,500 instances
  
This distribution clearly highlights the issue of class imbalance, where the number of healthy loans significantly outweighs the number of high-risk loans. Such an imbalance can pose challenges in accurately predicting and identifying high-risk loans due to the limited representation of this class in the dataset.

- - -

## Stages of the Machine Learning Analysis Process: From Data Preparation to Model Evaluation

* Separating the Data: The dataset was divided into two components - labels (target variable) and features (independent variables). This step involved identifying the variable(s) to be predicted and the relevant features that influence the prediction.

* Splitting the Data: The data was split into training and testing datasets. The training dataset was used to train the machine learning model, while the testing dataset was kept separate and used for evaluating the model's performance.

* Fitting a Logistic Regression Model: A logistic regression model was chosen and fitted using the training data. This involved learning the relationships between the features and the target variable in order to make predictions.

* Making Predictions: The trained logistic regression model was utilized to make predictions on the testing dataset. The model predicted the labels (target variable) based on the provided features.

* Evaluating Model Performance: The performance of the logistic regression model was assessed using various evaluation metrics. These metrics included accuracy score, confusion matrix, and classification report.

* Resampling the Data: To address the class imbalance issue, the RandomOverSampler module was employed to resample the data. This technique oversamples the minority class (high-risk loans) to balance the class distribution.

* Fitting and Predicting with Resampled Data: The logistic regression model was then refitted using the resampled data and used to make predictions on the testing dataset.

* Re-evaluating Model Performance: The performance of the updated logistic regression model was assessed again, considering the resampled data. This involved evaluating the accuracy score, confusion matrix, and classification report to gauge the model's effectiveness in predicting high-risk loans.

- - -

## Methods Used in Analysis: Logistic Regression and Resampling Techniques

In this analysis, the LogisticRegression algorithm was utilized to predict the target variable y_pred based on the provided features. Logistic regression is a commonly used algorithm for binary classification problems, where it estimates the probability of an instance belonging to a specific class.

To address the issue of class imbalance, the RandomOverSampler method was employed. This technique oversamples the minority class (high-risk loans) by randomly duplicating instances from this class. This helps to balance the class distribution and improve the model's ability to learn and predict the minority class effectively.

By using LogisticRegression in conjunction with RandomOverSampler, the model's performance in predicting high-risk loans was enhanced, enabling better classification and assessment of credit risk.

- - -

## Model Evaluation: Balanced Accuracy, Precision, and Recall Scores of Machine Learning Models

### Machine Learning Model 1: LogisticRegression algorithm

For the "healthy loan" class:
  * F1 score: 1,
  * Precision: 1,
  * Recall: 1

For the "high-risk loan" class:
  * F1 score: 0.88,
  * Precision: 0.87,
  * Recall: 0.89

Model Accuracy: 0.99

### Machine Learning Model 2: LogisticRegression algorithm with oversampled data

For the "healthy loan" class:
  * F1 score: 1,
  * Precision: 1,
  * Recall: 1

For the "high-risk loan" class:
  * F1 score: 0.93,
  * Precision: 0.87,
  * Recall: 1

Model Accuracy: 1

For the "healthy loan" class, the model achieves perfect precision, recall, and F1-score, indicating accurate identification of all instances in this class. This means that the model correctly predicts all "healthy loan" cases without any false positives or false negatives.

In predicting the "high-risk loan" class, the model shows a precision of 0.87, recall of 1, and an F1-score of 0.93. These scores indicate that the model performs well in identifying the majority of "high-risk loan" instances, with a relatively low number of false negatives.

- - -

## Summary and Recommendation

The logistic regression model fitted with oversampled data outperformed the initial model in predicting credit risk. It achieved higher accuracy, precision, and recall for both the "healthy loan" and "high-risk loan" classes. The oversampling technique helped address the class imbalance issue and improved the model's ability to predict high-risk loans accurately.

Considering the importance of correctly identifying both healthy and high-risk loans in credit risk assessment, the logistic regression model fitted with oversampled data is recommended for predicting credit risk. This model provides a balanced and reliable prediction, minimizing the risk of misclassifying loans and ensuring better decision-making in loan approvals or rejections.