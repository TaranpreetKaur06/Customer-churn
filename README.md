# Customer Churn Prediction and Retention Strategy

## Project Overview

Customer churn is one of the biggest challenges for telecom and subscription-based businesses. Churn occurs when customers stop using a company’s services. Since acquiring new customers is significantly more expensive than retaining existing ones, businesses aim to identify customers who are likely to churn and take preventive actions.

This project focuses on building machine learning classification models to predict customer churn and provide business-driven retention strategies based on data analysis and model insights.

---

# Business Problem Understanding

## What is Customer Churn?

Customer churn refers to customers leaving or discontinuing a company’s service.

In telecom companies, churn can occur when customers:

* Cancel subscriptions
* Switch to competitors
* Stop renewing services

---

## Why Churn is a Business Problem

Customer churn directly impacts:

* Revenue
* Customer lifetime value
* Market share
* Business growth

Acquiring a new customer is much more expensive than retaining an existing customer. High churn rates reduce profitability and increase marketing and acquisition costs.

---

## Why Predicting Churn is Important

Predicting churn helps businesses:

* Identify high-risk customers
* Improve customer retention
* Reduce revenue loss
* Increase customer satisfaction
* Optimize marketing campaigns

By identifying likely churners early, businesses can proactively offer retention strategies.

---

## Why Customer Retention is Important

Customer retention helps businesses:

* Maintain stable revenue
* Increase customer loyalty
* Improve long-term profitability
* Reduce customer acquisition costs

Retained customers often spend more over time and contribute to brand reputation.

---

## Why False Negatives Are More Costly

In churn prediction:

* **False Positive:** Predicting a customer will churn when they actually stay.
* **False Negative:** Predicting a customer will stay when they actually churn.

False negatives are more dangerous because:

* The business loses customers unexpectedly
* No retention action is taken
* Revenue loss becomes unavoidable

Therefore, identifying actual churners is highly important.

---

# Dataset Description

Dataset used:
Telco Customer Churn Dataset

Dataset contains customer information such as:

* Customer demographics
* Services subscribed
* Billing information
* Contract information
* Churn status

---

## Important Columns

| Column Name     | Description                        |
| --------------- | ---------------------------------- |
| customerID      | Unique customer identifier         |
| gender          | Male/Female                        |
| SeniorCitizen   | Whether customer is senior citizen |
| Partner         | Whether customer has partner       |
| Dependents      | Whether customer has dependents    |
| tenure          | Number of months customer stayed   |
| PhoneService    | Whether customer has phone service |
| InternetService | Type of internet service           |
| Contract        | Contract type                      |
| PaymentMethod   | Payment method used                |
| MonthlyCharges  | Monthly bill amount                |
| TotalCharges    | Total amount charged               |
| Churn           | Target variable                    |

---

## Numerical Columns

* tenure
* MonthlyCharges
* TotalCharges

---

## Categorical Columns

* gender
* Partner
* Dependents
* PhoneService
* MultipleLines
* InternetService
* Contract
* PaymentMethod
* Churn

---

## Target Variable

* Churn

Where:

* 1 = Customer churned
* 0 = Customer retained

---

## Problem Type

This is a **Classification Problem** because the target variable has categorical outcomes:

* Churn
* No Churn

---

## Why This is Supervised Learning

This is a supervised learning problem because:

* Historical labeled data is available
* Input features and target output are known
* Models learn patterns between features and churn outcome

---

# Data Cleaning and Preprocessing Summary

Several preprocessing steps were performed before model training.

---

## 1. Handling Missing Values

The `TotalCharges` column contained missing values due to incorrect formatting.

Actions performed:

* Converted column datatype to numeric
* Replaced missing values using median imputation

---

## 2. Correcting Data Types

The `TotalCharges` column was stored as object datatype.

Action:

* Converted to numeric datatype using `pd.to_numeric()`

---

## 3. Removing Irrelevant Columns

The `customerID` column was removed because:

* It does not contribute to prediction
* It is only a unique identifier

---

## 4. Encoding Categorical Variables

Machine learning models require numerical input.

Action:

* Applied Label Encoding to convert categorical columns into numeric values

Examples:

* Yes → 1
* No → 0

---

## 5. Feature Scaling

Numerical features were standardized using `StandardScaler`.

Features scaled:

* tenure
* MonthlyCharges
* TotalCharges

Scaling improves model performance and ensures consistent feature ranges.

---

## 6. Train-Test Split

Dataset was split into:

* 80% Training Data
* 20% Testing Data

This helps evaluate model performance on unseen data.

---

# Exploratory Data Analysis (EDA)

EDA was performed to understand churn behavior and identify important business insights.

---

## 1. Overall Churn Rate

### Insight:

A considerable number of customers have churned, indicating customer retention challenges.

### Business Interpretation:

The company should prioritize retention strategies to reduce revenue loss.

---

## 2. Churn by Contract Type

### Insight:

Customers with month-to-month contracts show the highest churn rate.

### Business Interpretation:

Long-term contracts improve customer retention and reduce churn risk.

---

## 3. Churn by Tenure

### Insight:

Customers with shorter tenure are more likely to churn.

### Business Interpretation:

New customers require better onboarding and engagement strategies.

---

## 4. Churn by Monthly Charges

### Insight:

Customers with higher monthly charges tend to churn more frequently.

### Business Interpretation:

Customers may perceive services as expensive or lacking value.

---

## 5. Churn by Payment Method

### Insight:

Electronic check users show higher churn rates.

### Business Interpretation:

Payment behavior may indicate lower customer commitment.

---

## 6. Churn by Internet Service

### Insight:

Fiber optic internet customers showed relatively higher churn rates.

### Business Interpretation:

Possible service quality or pricing issues may exist.

---

## 7. Churn by Senior Citizen Status

### Insight:

Senior citizens showed slightly higher churn patterns.

### Business Interpretation:

Customized support services may improve retention.

---

## 8. Relationship Between Tenure, Charges, and Churn

### Insight:

Customers with:

* Short tenure
* High monthly charges
* Month-to-month contracts

were more likely to churn.

### Business Interpretation:

These customer groups should be targeted for retention campaigns.

---

# Models Used

The following machine learning classification models were implemented:

---

## 1. Logistic Regression

Logistic Regression is a statistical classification algorithm commonly used for binary classification problems.

Advantages:

* Simple and interpretable
* Performs well on structured datasets
* Efficient and fast

---

## 2. Decision Tree Classifier

Decision Tree is a rule-based classification model that splits data into decision nodes.

Advantages:

* Easy to visualize
* Handles nonlinear relationships
* Simple interpretation

---

# Model Evaluation Results

The following evaluation metrics were used:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

# Confusion Matrix Explanation

| Metric         | Meaning                                           |
| -------------- | ------------------------------------------------- |
| True Positive  | Correctly predicted churn customers               |
| True Negative  | Correctly predicted non-churn customers           |
| False Positive | Predicted churn but customer stayed               |
| False Negative | Predicted non-churn but customer actually churned |

---

# Model Comparison and Selection

To choose the most suitable model for churn prediction, we need to consider several evaluation metrics beyond just accuracy. While accuracy gives an overall picture of correct predictions, for imbalanced datasets (where one class, like churn, is less frequent) or scenarios with differing costs for false positives and false negatives, other metrics are more insightful.

## Key Metrics for Churn Prediction

### Recall (Sensitivity) for the Churn Class

Recall measures the proportion of actual churners correctly identified by the model.

In churn prediction, high recall is extremely important because:

* Missing actual churners leads to revenue loss
* Businesses lose opportunities for customer retention

---

### Precision for the Churn Class

Precision measures the proportion of predicted churners who actually churned.

Higher precision reduces unnecessary retention spending on customers who were not likely to churn.

---

### F1-Score for the Churn Class

F1-score balances both:

* Precision
* Recall

It is useful when both false positives and false negatives matter.

---

# Model Performance Summary

## 1. Logistic Regression Model

* Accuracy: 70.28%
* Precision: 0.60
* Recall: 0.55
* F1-score: 0.57

---

## 2. Decision Tree Classifier

* Accuracy: 59.72%
* Precision: 0.45
* Recall: 0.49
* F1-score: 0.47

---

# Final Model Selection

Comparing both models, the **Logistic Regression model** performed significantly better across all major evaluation metrics.

### Reasons for Selecting Logistic Regression

* Higher accuracy
* Better recall for churn prediction
* Better precision
* Higher F1-score
* More effective at identifying actual churners

Recall was considered the most important metric because minimizing false negatives is critical in churn prediction.

A false negative means:

* The company fails to identify a customer who is about to leave
* No retention action is taken
* Revenue loss occurs

Therefore, the Logistic Regression model is the most suitable model for this business problem.

---

# Retention Strategy Recommendations

Based on EDA insights and model findings, the following business recommendations are proposed.

---

## 1. Discounts for High-Risk Customers

Customers with high monthly charges are more likely to churn.

### Recommendation:

* Provide loyalty discounts
* Offer bundled services
* Introduce customized pricing plans

---

## 2. Contract Upgrade Offers

Month-to-month customers showed the highest churn rates.

### Recommendation:

* Encourage long-term contracts
* Offer discounted yearly plans
* Provide contract renewal benefits

---

## 3. Early Engagement for New Customers

Short-tenure customers are highly likely to churn.

### Recommendation:

* Improve onboarding process
* Conduct early customer follow-ups
* Provide welcome offers and tutorials

---

## 4. Payment Method Interventions

Electronic check users showed higher churn behavior.

### Recommendation:

* Encourage automatic payment methods
* Offer incentives for auto-pay enrollment

---

## 5. Better Support for Specific Customer Groups

Fiber optic customers showed relatively high churn.

### Recommendation:

* Improve technical support
* Address service quality issues
* Provide faster complaint resolution

---

## 6. Promote Value-Added Services

Customers without additional services showed higher churn.

### Recommendation:

* Promote online security
* Offer tech support packages
* Bundle protection services

---

## 7. Loyalty Benefits for Long-Term Customers

Long-tenure customers are valuable business assets.

### Recommendation:

* Offer loyalty rewards
* Provide exclusive benefits
* Give renewal discounts

---

## 8. Continuous Monitoring and Feedback

### Recommendation:

* Continuously monitor churn indicators
* Track customer satisfaction
* Collect customer feedback regularly
* Improve retention campaigns using updated data

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

---

# Project Files

Repository contains:

* `README.md`
* `notebook.ipynb`
* `requirements.txt`
* `dataset_source.md`
* `outputs/`

---

# How to Run the Project

## Step 1

Clone the repository:

```bash
git clone <repository-link>
```

---

## Step 2

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 3

Open Jupyter Notebook or Google Colab.

---

## Step 4

Run all cells in:

```bash
notebook.ipynb
```

---

# Submission Guidelines

* Submit only the public GitHub repository link
* No ZIP files
* No PDF files
* No screenshots
* Repository must be public
* Repository must contain:

  * Complete README
  * Executable notebook/code
  * Required project files

---

# Conclusion

This project successfully developed machine learning models to predict customer churn and identify high-risk customers.

Among the implemented models, Logistic Regression achieved the best performance and was selected as the final model because of its stronger recall, precision, and F1-score for churn prediction.

The project also provided actionable business recommendations that can help telecom companies reduce churn, improve customer retention, and increase long-term profitability.
