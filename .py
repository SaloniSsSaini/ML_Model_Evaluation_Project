import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_custom_dataset(file_path, target_column):
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def define_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVC": SVC(probability=True)
    }

def define_param_grids():
    return {
        "Logistic Regression": {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']},
        "Random Forest": {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
        "SVC": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    }

def tune_models(models, param_grids, X_train, y_train):
    best_models = {}
    for name, model in models.items():
        print(f"\n🔍 Tuning {name}")
        param = param_grids[name]
        search = (RandomizedSearchCV(model, param, n_iter=10, cv=5, scoring='f1_macro', n_jobs=-1)
                  if name == "Random Forest"
                  else GridSearchCV(model, param, cv=5, scoring='f1_macro', n_jobs=-1))
        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_
        print(f"✅ Best Parameters: {search.best_params_}")
    return best_models

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'F1-Score': f1_score(y_test, y_pred, average='macro')
    }

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png'))
    plt.close()

def plot_roc_curve(model, X_test, y_test, model_name, output_dir):
    if len(np.unique(y_test)) != 2:
        return
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.savefig(os.path.join(output_dir, f'roc_curve_{model_name.replace(" ", "_")}.png'))
    plt.close()

def save_comparison_plot(results_df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df.reset_index().melt(id_vars='index'),
                x='index', y='value', hue='variable')
    plt.title('Model Performance Comparison')
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()

def main():
    file_path = 'diabetes.csv'
    target_column = 'Outcome'
    output_dir = "model_outputs"
    os.makedirs(output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_custom_dataset(file_path, target_column)
    models = define_models()
    param_grids = define_param_grids()
    best_models = tune_models(models, param_grids, X_train, y_train)
    results = {}
    for name, model in best_models.items():
        print(f"\n📊 Evaluating: {name}")
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, name, output_dir)
        plot_roc_curve(model, X_test, y_test, name, output_dir)

    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(output_dir, "model_results.csv"))
    save_comparison_plot(results_df, output_dir)
    best = results_df['F1-Score'].idxmax()
    print(f"\n🏆 Best Model: {best} (F1 = {results_df.loc[best, 'F1-Score']:.4f})")

if __name__ == "__main__":
    main()
