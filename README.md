# **Housing Price Prediction — Feature Engineering & Modeling Pipeline**

## **Project Overview**

**This project demonstrates advanced feature engineering techniques and machine learning modeling applied to housing price prediction.  
The goal is to improve predictive accuracy by transforming raw data into meaningful features through custom preprocessing pipelines, followed by robust model training.**

---

## **Key Features & Engineering Techniques**

- **Addition Pipeline:** **Adds up columns like total square footage to capture overall size.**  
- **Multiplication Pipeline:** **Multiplies features such as quality and condition to capture their combined effect.**  
- **Subtraction Pipeline:** **Calculates differences, like how old a house is by subtracting the build year from the sale year.**  
- **Weighted Bathrooms Pipeline:** **Combines full and half bathrooms with different weights to get a better bathroom count.**  
- **RBF Kernel Similarity:** **Applies a radial basis function to the year built, helping the model capture non-linear trends related to the age of the house.**

**I wrapped each of these transformations into easy-to-use pipelines with scikit-learn, which keeps everything neat and reusable.**

---

## **How it Works**

- **Handles missing data with median imputation to keep the dataset clean.**  
- **Custom transformers bake domain knowledge directly into the features.**  
- **These steps plug into a bigger pipeline that cleans data and feeds it into a machine learning model.**  
- **Uses a Random Forest regressor — reliable and effective for complex data.**  
- **The whole setup is flexible and easy to expand or adjust.**

---

## **Quick Start**

**1. Clone this repo.**

**2. Install the dependencies:**  
```bash
pip install -r requirements.txt




Quick Start

Clone this repo.

Install the dependencies:

pip install -r requirements.txt


Use the pipelines like this:

from feature_engineering import (
    addition_pipeline,
    multiplication_pipeline,
    subtraction_pipeline,
    weighted_bathrooms_pipeline,
    rbf_kernel_pipeline,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

preprocessor = ColumnTransformer([
    ('add', addition_pipeline(), ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']),
    ('mult', multiplication_pipeline(), ['OverallQual', 'OverallCond']),
    ('sub', subtraction_pipeline(), ['YrSold', 'YearBuilt']),
    ('bath', weighted_bathrooms_pipeline(), ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']),
    ('rbf', rbf_kernel_pipeline(target_year=1960), ['YearBuilt']),
])

model_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42)),
])

model_pipeline.fit(X_train, y_train)
predictions = model_pipeline.predict(X_test)


```




###  Final Test Set Evaluation
- **Final RMSE:** `36,503.50`



## Notes

There are further opportunities for improvement:
- Integrating unsupervised learning techniques like **clustering**
- Dropping **uninformative or noisy features**
- Utilizing **model stacking** or hyperparameter tuning

For this phase, the main focus was on building a clean, modular, and extensible pipeline using custom feature engineering — and that objective was achieved.




## Why This Matters

This project shows how I combine technical skills with real-world thinking to get better predictions:

Bringing domain knowledge directly into the model

Creating clean, reusable code

Using smart feature engineering to boost performance

Building end-to-end pipelines that are easy to maintain





#

I’m excited about new opportunities and collaborations. Reach out anytime!

LinkedIn: https://www.linkedin.com/in/joseph-hawkins-0aa960259/

Email: Hawkinsjoseph2003@gmail.com

GitHub: https://github.com/JosephPHawkins


Thanks for taking a look!
