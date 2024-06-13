import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import mlflow

class ModelTrainer:
    """
    Trains an XGBoost model with optional SMOTE oversampling, 
    hyperparameter tuning, and MLflow tracking.
    """

    def __init__(self, use_smote=False, random_state=42, log_feature_importance=False):
        """Initializes the ModelTrainer."""
        self.use_smote = use_smote
        self.random_state = random_state
        self.log_feature_importance = log_feature_importance
        self.model = None
        self.best_params = None

    def _split_data(self, X, y):
        """Splits data into train and test sets."""
        return train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

    def _apply_smote(self, X_train, y_train):
        """Applies SMOTE oversampling to the training data."""
        from imblearn.over_sampling import SMOTE 
        smote = SMOTE(random_state=self.random_state, sampling_strategy="auto")
        return smote.fit_resample(X_train, y_train)

    def _train_model(self, X_train, y_train):
        """Trains the XGBoost model with GridSearchCV for hyperparameter tuning."""
        param_grid = {
            "max_depth": [5, 15, 25],
            "learning_rate": [0.05, 0.1, 0.3, 0.5],
            "n_estimators": [100, 200, 300],
            "subsample": [0.6, 0.8, 1.0],
        }
        xgb_clf = XGBClassifier(n_estimators=100, random_state=self.random_state)
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            xgb_clf, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_ 

    def _evaluate_model(self, X_test, y_test):
        """Evaluates the model and returns performance metrics."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, precision, recall, f1, report, conf_matrix

    def train(self, X, y, mlflow_run_name="XGBoost Model Training"):
        """Trains the model, logs metrics and artifacts to MLflow."""
        X_train, X_test, y_train, y_test = self._split_data(X, y)

        if self.use_smote:
            X_train, y_train = self._apply_smote(X_train, y_train)

        self._train_model(X_train, y_train)

        accuracy, precision, recall, f1, report, conf_matrix = self._evaluate_model(
            X_test, y_test
        )

        # MLflow Tracking
        mlflow.set_tracking_uri(
            "file:///C:/Users/fares/Documents/GitHub/test_technique/mlflow_data"
        )
        experiment_name = "Phishing URLs Detection"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=mlflow_run_name):
            mlflow.log_params(self.best_params)
            mlflow.log_metrics(
                {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                }
            )

            if self.log_feature_importance:
                self._log_feature_importance(X_train)

            self._log_confusion_matrix(conf_matrix)
            mlflow.log_text(report, "classification_report.txt")
            mlflow.sklearn.log_model(self.model, "model")

    def _log_feature_importance(self, X_train):
        """Logs feature importance plot and data to MLflow."""
        feature_importances = self.model.feature_importances_
        importance_df = pd.DataFrame(
            {"feature": X_train.columns, "importance": feature_importances}
        ).sort_values("importance", ascending=False)
        mlflow.log_dict(importance_df.to_dict(), "feature_importances.json")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df.head(20), x="importance", y="feature", ax=ax)
        ax.set_title("Top 20 Feature Importances")
        plt.tight_layout()

        # Ensure directory exists
        os.makedirs("../mlops/artifacts", exist_ok=True)
        feature_importance_path = "../mlops/artifacts/feature_importance.png"
        plt.savefig(feature_importance_path)
        mlflow.log_artifact(feature_importance_path)
        plt.close(fig)

    def _log_confusion_matrix(self, conf_matrix):
        """Logs the confusion matrix as an artifact to MLflow."""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        # Ensure directory exists
        os.makedirs("../mlops/artifacts", exist_ok=True)
        confusion_matrix_path = "../mlops/artifacts/confusion_matrix.png"
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
        mlflow.log_artifact(confusion_matrix_path)
        plt.close(fig)

    def save_model(self, model_path):
        """Saves the trained model to a file."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"Model saved to: {model_path}")
