# Binary Classification on Adult Dataset using Logistic Regression with Plotting of Confusion Matrix
This code implements a binary classification model using logistic regression to predict whether a person's salary is greater than $50K or not based on various demographic and socio-economic factors. The confusion matrix is also plotted to evaluate the model's performance.

## Libraries Used
- Pandas
- Scikit-learn
- Matplotlib

## Dataset
The adult dataset is used for training and testing the model. The dataset contains information about 32,561 individuals, and the target variable is "salary".

## Data Preprocessing
Irrelevant columns "education" and "native_country" are dropped.
The target variable "salary" is encoded using LabelEncoder.
Categorical features are encoded using one-hot encoding.

## Model Building and Evaluation
- The dataset is split into training and testing sets.
- A logistic regression model is built and trained on the training set.
- The model's performance is evaluated on the testing set using accuracy score and confusion matrix.
- The confusion matrix is plotted using matplotlib.

## Running the Code
Ensure that the required libraries are installed.
Download the adult dataset as "adult.csv" and place it in the same directory as the code.
Run the code.

## Output
- The accuracy score of the model on the testing set is printed.
- The confusion matrix is plotted and displayed.