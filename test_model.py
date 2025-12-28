import joblib
import pandas as pd

# Load trained model
model = joblib.load("iphone_purchase_model.pkl")
new_data = pd.DataFrame({
    'Gender': [1],   # Male = 0, Female = 1
    'Age': [25],
    'Salary': [50000]
})
# Make prediction
prediction = model.predict(new_data)
print(f'Prediction for new data: {prediction[0]}')  # 0 = Not Purchase, 1 = Purchase