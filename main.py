# Binary classification on adult dataset using logistic regression with plotting of confusion matrix

# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('adult.csv')

# Drop irrelevant columns
data.drop(['education', 'native_country'], axis=1, inplace=True)

# Encode the target variable
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])

# Encode categorical features
data = pd.get_dummies(data, columns=['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(['salary'], axis=1), data['salary'], test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.matshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
