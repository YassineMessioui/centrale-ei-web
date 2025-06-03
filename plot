# Example: Using 4 models on a binary classification problem
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
models = [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(),
    SVC(probability=True),
    GaussianNB()
]
model_names = ['Logistic Regression', 'Random Forest', 'SVC', 'Naive Bayes']

# Train models
for model in models:
    model.fit(X_train, y_train)

# Evaluate and visualize
evaluate_models(models, model_names, X_test, y_test)
