import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fpdf import FPDF

# Data collection - loading the dataset
wine_dataset = pd.read_csv('winequality-red.csv')

# Create PDF document with landscape orientation
pdf = FPDF(orientation='L')
pdf.add_page()
pdf.set_font("Arial", size=12)

# Title
pdf.cell(270, 10, txt="Wine Quality Analysis Report", ln=True, align='C')
pdf.ln(10)

# Number of rows & columns in the dataset
pdf.cell(270, 10, txt="Number of rows & columns in the dataset", ln=True, align='L')
pdf.cell(270, 10, txt=str(wine_dataset.shape), ln=True, align='L')
pdf.ln(5)

# First 5 rows of the dataset
pdf.cell(270, 10, txt="First 5 rows of the dataset", ln=True, align='L')
pdf.multi_cell(0, 10, txt=str(wine_dataset.head()), align='L')
pdf.ln(5)

# Checking for missing values
pdf.cell(270, 10, txt="Missing Values", ln=True, align='L')
pdf.cell(270, 10, txt=str(wine_dataset.isnull().sum()), ln=True, align='L')
pdf.ln(10)

# Data Analysis
# Statistical measures of the dataset
pdf.cell(270, 10, txt="Statistical measures of the dataset", ln=True, align='L')
pdf.multi_cell(0, 10, txt=str(wine_dataset.describe()), align='L')
pdf.ln(10)

# Quality count plot
plt.figure(figsize=(12, 6))
sns.countplot(x='quality', data=wine_dataset, palette='muted')
plt.title('Quality Count')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('quality_count_plot.png', dpi=300)
pdf.image('quality_count_plot.png', x=None, y=None, w=260, h=120)
plt.close()

# Volatile acidity vs quality
plt.figure(figsize=(12, 6))
sns.barplot(x='quality', y='volatile acidity', data=wine_dataset, palette='muted')
plt.title('Volatile Acidity vs Quality')
plt.xlabel('Quality')
plt.ylabel('Volatile Acidity')
plt.tight_layout()
plt.savefig('volatile_acidity_vs_quality.png', dpi=300)
pdf.image('volatile_acidity_vs_quality.png', x=None, y=None, w=260, h=120)
plt.close()

# Citric acid vs quality
plt.figure(figsize=(12, 6))
sns.barplot(x='quality', y='citric acid', data=wine_dataset, palette='muted')
plt.title('Citric Acid vs Quality')
plt.xlabel('Quality')
plt.ylabel('Citric Acid')
plt.tight_layout()
plt.savefig('citric_acid_vs_quality.png', dpi=300)
pdf.image('citric_acid_vs_quality.png', x=None, y=None, w=260, h=120)
plt.close()

# Correlation matrix
plt.figure(figsize=(14, 10))
correlation = wine_dataset.corr()
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
pdf.image('correlation_matrix.png', x=None, y=None, w=260, h=180)
plt.close()

# Separate the data and label
X = wine_dataset.drop('quality', axis=1)
y = wine_dataset['quality']

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

# Model training with best parameters
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Model Accuracy
pdf.cell(270, 10, txt="Model Accuracy: {:.2f}%".format(accuracy * 100), ln=True, align='L')
pdf.ln(10)

# Input data for prediction
input_data = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5)

# Prediction
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)

# Prediction result
if prediction[0] == 1:
    prediction_result = 'Good Quality Wine'
else:
    prediction_result = 'Bad Quality Wine'

pdf.cell(270, 10, txt="Prediction Result: {}".format(prediction_result), ln=True, align='L')

# Save the PDF document
pdf.output("wine_quality_analysis_report.pdf")



