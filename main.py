import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
from Preprocessing import preprocess_data_with_labels,preprocess_data

# Load your data
data = pd.read_csv('Data.csv')

# Preprocess the data
data = preprocess_data(data)

def train_model(data):
        # Check if 'casualty_severity' column exists
    if 'casualty_severity' not in data.columns:
        raise KeyError("Column 'casualty_severity' not found in DataFrame.")

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
    report = classification_report(y_test, y_pred, output_dict=True)
    model_evaluation = pd.DataFrame(report).transpose()

    # Save Model Evaluation to Excel file
    model_evaluation.to_excel('Output/model_evaluation.xlsx')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('Output/confusion_matrix.png')

    # Save Confusion Matrix to Excel file
    cm_df.to_excel('Output/confusion_matrix.xlsx')
    return clf

# Train the model
trained_model = train_model(data)

def save_plot(data, column, plot_title, file_name):
    # Check if 'casualty_severity' column exists
    if column not in data.columns:
        print(f"Warning: Column '{column}' not found in DataFrame. Skipping plot generation.")
        return

    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=data, x=column, hue='casualty_severity')

    # Add data labels
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 10), 
                    textcoords = 'offset points')

    plt.title(plot_title)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.legend(title='Casualty_Severity')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent overlapping labels
    plt.savefig(os.path.join('Plot', file_name))
    plt.close()


def main():
    # Create folders if they don't exist
    if not os.path.exists('Output'):
        os.makedirs('Output')
    if not os.path.exists('Plot'):
        os.makedirs('Plot')

    # Load data
    data = pd.read_csv('Data.csv')

    # Preprocess data with labels
    data = preprocess_data_with_labels(data)

    # Check if 'casualty_severity' column exists
    if 'casualty_severity' not in data.columns:
        raise KeyError("Column 'casualty_severity' not found in DataFrame.")

    #Bar chart - Ratio of casualty_class based on casualty_severity
    save_plot(data, 'casualty_class', 'Ratio of Casualty_Class based on Casualty_Severity', 'casualty_class_severity.png')

    #Donut chart - Ratio of sex_of_casualty
    plt.figure(figsize=(8, 8))
    sex_of_casualty_ratio = data['sex_of_casualty'].value_counts()
    plt.pie(sex_of_casualty_ratio, labels=sex_of_casualty_ratio.index, autopct='%1.1f%%', startangle=90)
    plt.title('Ratio of Sex_of_Casualty')
    plt.legend(title='Sex_of_Casualty', loc='upper right')
    plt.tight_layout()
    plt.savefig('Plot/sex_of_casualty_ratio.png')
    plt.close()

    #Bar chart - Ratio of age_of_casualty based on Casualty_Severity
    save_plot(data, 'age_of_casualty', 'Ratio of Age_of_Casualty based on Casualty_Severity', 'age_casualty_severity.png')

    #Sunburst chart - Ratio of casualty_class per sex_of_casualty
    sunburst_data = data.groupby(['sex_of_casualty', 'casualty_class']).size().reset_index(name='Count')
    fig = px.sunburst(sunburst_data, path=['sex_of_casualty', 'casualty_class'], values='Count')
    fig.update_traces(textinfo='label+percent entry')
    fig.update_layout(title='Ratio of Casualty_Class per Sex_of_Casualty')
    fig.write_image('Plot/sunburst_chart.png')
    #Bar chart - Ratio of age_of_casualty based on casualty_severity
    save_plot(data, 'age_of_casualty', 'Ratio of Age_of_Casualty based on Casualty_Severity', 'age_severity_ratio.png')

    # Train the model
    trained_model = train_model(data)

if __name__ == "__main__":
    main()
