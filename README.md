Dataset Description

The dataset contains customer information with the following columns:

Gender: Male / Female

Age: Customer age

Salary: Annual salary

Purchase Iphone:

0 → Not Purchased

1 → Purchased

Basic preprocessing and exploratory checks are performed, including:

Null value check

Data description

Target class distribution

 Dataset is loaded and processed in the training script 

iphone

 Technologies Used

Python

pandas

scikit-learn

joblib

 Machine Learning Model

Algorithm: Decision Tree Classifier

Class Weight: Balanced

Train-Test Split: 80% training, 20% testing

Evaluation Metrics:

Accuracy

Confusion Matrix

Classification Report

The trained model is saved as:

iphone_purchase_model.pkl

 How to Run the Project
1️ Install Dependencies
pip install pandas scikit-learn joblib

2️ Train the Model

Run the training script:

python iphone.py


This will:

Load and preprocess data

Train the Decision Tree model

Evaluate performance

Save the trained model

 Training and evaluation logic is implemented here 

iphone

3️Test the Model with New Data

Run:

python test_model.py


Example input used:

Gender: Female (1)
Age: 25
Salary: 50000


Output:

Prediction for new data: 1


 Prediction script reference 

test_model

 Label Encoding Logic

Gender Mapping:

Male → 0

Female → 1

Prediction Output:

0 → Will NOT purchase iPhone

1 → Will purchase iPhone

 Model Use Case

This project can be used for:

Customer purchase behavior analysis

Marketing targeting strategies

Beginner-friendly ML classification demos

 Future Improvements

Add more features (Location, Brand Preference, Credit Score)

Try other algorithms (Random Forest, Logistic Regression)

Build a Flask / Streamlit web app

Perform hyperparameter tuning

Author
Bhaskar
MA in Mass Communication & Journalism
Machine Learning & Data Analytics Enthusiast
