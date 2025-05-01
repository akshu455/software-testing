#!/usr/bin/env python3

# Diabetes Prediction Using Support Vector Machine
import pickle
import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# For training the model
# For training the model
def train():
    dataset = pd.read_csv('pima.csv')
  
    X = dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']]

    Y = dataset['Outcome'].values.ravel() # Reshape target variable
    
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
    
    from sklearn.svm import SVC
    model = SVC(kernel='linear')
    svc = model.fit(X_train, Y_train)
    
    # Save Model As Pickle File
    with open('svc.pkl', 'wb') as m:
        pickle.dump(svc, m)
    test(X_test, Y_test)

# Test accuracy of the model
def test(X_test, Y_test):
    with open('svc.pkl', 'rb') as mod:
        p = pickle.load(mod)
    
    pre = p.predict(X_test)
    print(accuracy_score(Y_test, pre))  # Prints the accuracy of the model

def find_data_file(filename):
    if getattr(sys, "frozen", False):
        # The application is frozen.
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen.
        datadir = os.path.dirname(__file__)

    return os.path.join(datadir, filename)

def check_input(data) -> int:
    # Define the required feature names
    features_order = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']
    
    # Convert input data into DataFrame
    df = pd.DataFrame([data])  # No need for `index=[0]`
    
    # Check if all required features exist
    missing_cols = [col for col in features_order if col not in df.columns]
    if missing_cols:
        print(f"Missing Columns: {missing_cols}")
        return -1  # Indicate error
    
    # Select required features
    df = df[features_order]
    
    # Load model
    with open("svc.pkl", 'rb') as model:
        p = pickle.load(model)
    
    # Make prediction
    op = p.predict(df)
    return op[0]



if __name__ == '__main__':
    train()

