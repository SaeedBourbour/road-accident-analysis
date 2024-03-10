import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model(data):
    # Convert categorical columns to dummy variables
    data = pd.get_dummies(data)

    # Define features and target variable
    X = data.drop(columns=['casualty_severity'])
    y = data['casualty_severity']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))

    # Return the trained model
    return clf
