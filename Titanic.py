import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

#Import the csv file
titanic_raw = pd.read_csv("train.csv",sep=',')
print(titanic_raw)
print(titanic_raw.columns)
print(titanic_raw.head)

# Use PassengerId as index
titanic_raw = titanic_raw.set_index("PassengerId")
print(titanic_raw)

#Change column names to lowercase
titanic_raw.rename(columns={column: column.lower() for column in titanic_raw.columns}, inplace=True)

titanic_raw.info()


#Drop irrelavant nominal variable and conduct dummy encoding on categorial variables
titanic = titanic_raw.drop(columns=["name", "sibsp","fare", "parch", "ticket", "cabin"])
titanic = pd.get_dummies(
    titanic,
    columns=["sex", "pclass", "embarked"],
    drop_first=True
)
print(titanic)

#Identify missing values
titanic.isna().sum()

#Build correlation matrix
corr_matrix = titanic.corr()

#Regression imputation for missing values for "age"
age_complete = titanic.dropna(subset=['age', 'pclass_2','pclass_3'])
age_missing = titanic[titanic['age'].isna()]

dependent = age_complete[['pclass_2','pclass_3']]
target = age_complete['age']

#Split the complete data into training and test sets
dependent_train, dependent_test, target_train, target_test = train_test_split(dependent, target, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(dependent_train, target_train)

# Predict missing values
imputed_values = model.predict(age_missing[['pclass_2', 'pclass_3']])

# Replace missing values with imputed values
titanic.loc[titanic['age'].isna(), 'age'] = imputed_values

#Define prediction baseline
# Baseline: Survive if female
survived = titanic.pop("survived")
predictions_baseline = 1 - titanic.sex_male
baseline = accuracy_score(survived,predictions_baseline)
print(baseline)
#79%


#Conduct stratified train-test split
X = titanic
y = titanic_raw["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=888)

#Verify the percentage of target vairable is balanced distribtured
print("Original data:")
print(y.value_counts(normalize=True))
print("Training data:")
print(y_train.value_counts(normalize=True))
print("Testing data:")
print(y_test.value_counts(normalize=True))

print("Training set size:", len(X_train))
print("Test set size:", len(X_test))

#Define the grid we will be testing
parameters = {
    'max_depth': [3, 5, 7],
    'max_features': [.5, .8, 1],
    'n_estimators': [25, 30, 35, 40, 50],
}

# Perform the grid search and score it with grid search
random_search = GridSearchCV(RandomForestClassifier(), parameters)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print(best_params)
#{'max_depth': 5, 'max_features': 0.8, 'n_estimators': 25}

train_accuracy = random_search.best_score_
print(train_accuracy)
#83%


#Test this model on test set
predictions_test = random_search.predict(X_test)
test_accuracy = accuracy_score(y_test, predictions_test)
print(test_accuracy)
#80%

#Build final model
final_classifier = RandomForestClassifier(**random_search.best_params_)
final_classifier.fit(titanic, survived)

# Read the test data
titanic_test = pd.read_csv("test.csv",sep=',')

#Conduct pre-prosess for test set
titanic_test = titanic_test.set_index("PassengerId")
titanic_test.rename(columns={column: column.lower() for column in titanic_test.columns}, inplace=True)
titanic_test=titanic_test.drop(columns=["name", "fare","sibsp", "parch", "ticket", "cabin"])
titanic_test = pd.get_dummies(
    titanic_test,
    columns=["sex", "pclass", "embarked"],
    drop_first=True
)

#Impute missing values for test set
complete = titanic_test.dropna(subset=['age', 'pclass_2','pclass_3'])
missing = titanic_test[titanic_test['age'].isna()]

dependent_2 = complete[['pclass_2','pclass_3']]
target_2= complete['age']

#Split the complete data into training and test sets
dependent_train_2, dependent_test_2, target_train_2, target_test_2 = train_test_split(dependent_2, target_2, test_size=0.2, random_state=42)

# Train a linear regression model
model_2 = LinearRegression()
model_2.fit(dependent_train_2, target_train_2)

# Predict missing values
imputed_values_2 = model_2.predict(missing[['pclass_2', 'pclass_3']])

# Replace missing values with imputed values
titanic_test.loc[titanic_test['age'].isna(), 'age'] = imputed_values_2

titanic_test.isna().sum()

#Apply the model on test set and get the prediction
predictions = final_classifier.predict(titanic_test)
print(predictions)

#Save the result to csv file
predictions_df = pd.DataFrame(data={"Survived": predictions}, index=titanic_test.index)
print(predictions_df.head())
predictions_df.to_csv("submission.csv")