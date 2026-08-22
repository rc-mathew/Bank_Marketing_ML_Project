\# Bank Marketing Subscription Prediction



An end-to-end machine learning project for predicting whether a bank customer will subscribe to a term deposit following a marketing campaign.



The project demonstrates a structured machine learning workflow covering data ingestion, data quality validation, exploratory data analysis, preprocessing, feature engineering, model development, class imbalance handling, cross-validation, hyperparameter tuning, decision-threshold optimization, model explainability, model persistence, automated testing, and continuous integration.



\## Business Problem



Direct marketing campaigns require banks to decide which customers are most likely to respond positively to a term-deposit offer.



A machine learning model can help rank customers according to their likelihood of subscription, allowing marketing resources to be directed toward more promising prospects.



This project treats the problem as a binary classification task:



\- `0` — customer does not subscribe

\- `1` — customer subscribes



\## Dataset



The project uses the Bank Marketing dataset containing information related to direct marketing campaigns of a Portuguese banking institution.



The modelling dataset contains customer attributes, campaign information and economic indicators used to predict term-deposit subscription.



\## Machine Learning Workflow



The project was developed incrementally using a modular ML workflow:



1\. Data ingestion

2\. Data quality validation

3\. Exploratory data analysis

4\. Preprocessing and feature engineering

5\. Baseline model development

6\. Ensemble model comparison

7\. Class imbalance analysis

8\. Decision-threshold optimization

9\. Stratified cross-validation

10\. Hyperparameter tuning

11\. Model explainability

12\. Model persistence

13\. Automated model tests

14\. GitHub Actions CI



\## Project Structure



```text

Bank\_Marketing\_ML\_Project/

│

├── .github/

│   └── workflows/

│       └── tests.yml

│

├── data/

│   └── raw/

│

├── models/

│

├── notebooks/

│

├── src/

│   └── models/

│

├── tests/

│   └── test\_model.py

│

└── README.md

```



\## Data Validation



Data quality checks are included before modelling to identify potential issues in the dataset and establish a reliable input for the ML pipeline.



\## Exploratory Data Analysis



Exploratory analysis is used to investigate:



\- target distribution

\- numerical variables

\- categorical variables

\- feature relationships

\- potential data-quality issues

\- patterns associated with customer subscription



\## Preprocessing and Feature Engineering



The project separates preprocessing from model training to create a reproducible machine learning workflow.



The preprocessing stage prepares numerical and categorical features for downstream models while reducing the risk of inconsistent transformations between training and inference.



\## Model Development



Multiple classification approaches are evaluated rather than relying on a single algorithm.



The workflow includes baseline modelling followed by ensemble-model training and comparison.



\## Class Imbalance



Bank marketing response data is imbalanced because substantially fewer customers subscribe than decline.



The project therefore evaluates model behaviour beyond accuracy and includes experiments specifically addressing class imbalance.



\## Model Evaluation



Evaluation includes classification metrics appropriate for an imbalanced binary classification problem, including:



\- Accuracy

\- Precision

\- Recall

\- F1-score

\- ROC-AUC

\- PR-AUC



Using multiple metrics provides a more complete view of performance on the minority subscription class.



\## Decision Threshold Optimization



Instead of assuming that `0.50` is always the optimal probability threshold, the project includes threshold analysis to examine the trade-off between precision and recall.



This is particularly relevant to marketing applications where the cost of contacting customers and the value of identifying potential subscribers may differ.



\## Cross-Validation



Stratified cross-validation is used to assess model stability while preserving the class distribution across folds.



This provides a more reliable estimate of generalization performance than relying exclusively on a single train/test split.



\## Hyperparameter Tuning



Hyperparameter tuning is performed to improve model performance systematically while maintaining reproducibility.



\## Model Explainability



Feature-importance analysis is included to investigate which variables contribute most strongly to model predictions.



This helps connect predictive performance with interpretable business insights.



\## Model Persistence



The final preprocessing and prediction pipeline can be serialized as a reusable model artifact.



Keeping preprocessing and prediction logic together reduces the risk of training-serving inconsistencies.

## FastAPI Model Serving

The trained machine learning pipeline is exposed through a FastAPI REST API.

Available endpoints:

- `GET /` — API information
- `GET /health` — service health check
- `POST /predict` — customer subscription prediction

The `/predict` endpoint accepts customer and campaign features
as JSON and passes them through the saved preprocessing and
Random Forest classification pipeline.

### Successful API Test

The REST API was successfully tested through Swagger UI.

Example response:

{
  "prediction": 0,
  "subscription": "no",
  "probability": 0.3315170224485902
}

HTTP Status: 200 OK

This confirms the complete inference workflow:

JSON Request
    ↓
FastAPI
    ↓
Input Validation
    ↓
Feature Preparation
    ↓
Saved Preprocessing Pipeline
    ↓
Random Forest Classifier
    ↓
Prediction Probability
    ↓
JSON Response

\## Automated Testing



The repository contains automated tests implemented with `pytest`.



Current tests verify important model-pipeline behaviour, including:



\- pipeline construction

\- successful fitting

\- binary predictions

\- valid probability outputs

\- binary classifier classes



The local test suite currently passes:



```text

5 passed

```



\## Continuous Integration



GitHub Actions automatically executes the ML test suite when repository changes trigger the workflow.



This provides automated validation that important model-pipeline behaviour continues to work after code changes.



\## Reproducibility



The project uses:



\- modular source code

\- reusable preprocessing

\- persisted model artifacts

\- automated tests

\- version control

\- continuous integration



These practices make the project closer to a production-oriented ML workflow than a notebook-only modelling exercise.



\## Technologies



\- Python

\- Pandas

\- NumPy

\- scikit-learn

\- Matplotlib

\- Jupyter Notebook

\- joblib

\- pytest

\- Git

\- GitHub

\- GitHub Actions



\## Key Learning Outcomes



This project demonstrates practical experience with:



\- end-to-end binary classification

\- imbalanced classification

\- feature engineering

\- ensemble learning

\- model evaluation

\- threshold optimization

\- stratified cross-validation

\- hyperparameter tuning

\- explainable machine learning

\- model persistence

\- automated testing

\- CI for machine learning projects



\## Future Improvements



Potential extensions include:



\- API-based model serving

\- experiment tracking

\- data/model drift monitoring

\- model registry integration

\- containerized deployment

\- additional production monitoring



\## Author



\*\*Reuben C. Mathew\*\*



Machine Learning | Data Science | FinTech

