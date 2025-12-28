import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import joblib
# Load the dataset
data = pd.read_csv('iphone_purchase_records.csv')
df = data.copy()
# Preprocess the data
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df['Purchase Iphone'].value_counts())

#map categorical variables to numerical values
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
print(df.head())

# Define features and target variable
X = df.drop('Purchase Iphone', axis=1)
y = df['Purchase Iphone']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42,class_weight='balanced')
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Save the trained model to a file
joblib.dump(clf, 'iphone_purchase_model.pkl')



