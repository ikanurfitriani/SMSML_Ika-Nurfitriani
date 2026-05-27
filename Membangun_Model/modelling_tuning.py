# MIT License

# Copyright (c) 2026 Ika Nurfitriani

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub

from sklearn import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    PrecisionRecallDisplay, RocCurveDisplay
)

if __name__ == "__main__":
    DAGSHUB_USERNAME = "ikanurfitriani"
    DAGSHUB_REPO_NAME = "Membangun_Model_MLflow"
    
    print("Menginisialisasi pelacakan eksperimen ke DagsHub Cloud...")
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    mlflow.set_experiment("Credit_Card_Default_Tuning")
    
    TRAIN_PATH = "default-credit-card_preprocessing/credit-card-train_clean.csv"
    TEST_PATH = "default-credit-card_preprocessing/credit-card-test_clean.csv"
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = train_df.drop(columns=['default payment next month'])
    y_train = train_df['default payment next month']
    X_test = test_df.drop(columns=['default payment next month'])
    y_test = test_df['default payment next month']
    
    tuned_params = {
        "n_estimators": 150,
        "max_depth": 8,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": 42
    }
    
    with mlflow.start_run(run_name="Random_Forest_Tuned_Online"):
        print("Melatih model hasil Tuning & mengunggah data ke DagsHub...")
        
        model = RandomForestClassifier(**tuned_params)
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        
        mlflow.log_params(tuned_params)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_prec)
        mlflow.log_metric("test_recall", test_rec)
        mlflow.log_metric("test_f1_score", test_f1)
        
        artifacts = {
            "training_confusion_matrix.png": lambda: sns.heatmap(confusion_matrix(y_train, y_train_pred), annot=True, fmt='d', cmap='Blues'),
            "testing_confusion_matrix.png": lambda: sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Oranges'),
            "training_precision_recall_curve.png": lambda: PrecisionRecallDisplay.from_estimator(model, X_train, y_train),
            "testing_precision_recall_curve.png": lambda: PrecisionRecallDisplay.from_estimator(model, X_test, y_test),
            "training_roc_curve.png": lambda: RocCurveDisplay.from_estimator(model, X_train, y_train),
            "testing_roc_curve.png": lambda: RocCurveDisplay.from_estimator(model, X_test, y_test)
        }
        
        for file_name, plot_func in artifacts.items():
            plt.figure(figsize=(6, 5))
            plot_func()
            plt.title(file_name.replace(".png", "").replace("_", " ").title())
            plt.savefig(file_name, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(file_name)
            if os.path.exists(file_name): os.remove(file_name)
            
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write("=== TUNED TRAINING REPORT ===\n")
            f.write(classification_report(y_train, y_train_pred))
            f.write("\n\n=== TUNED TESTING REPORT ===\n")
            f.write(classification_report(y_test, y_test_pred))
        mlflow.log_artifact(report_path)
        if os.path.exists(report_path): os.remove(report_path)

        estimator_path = "estimator.html"
        html_estimator = utils.estimator_html_repr(model)
        with open(estimator_path, "w", encoding="utf-8") as f:
            f.write(html_estimator)
        mlflow.log_artifact(estimator_path)
        if os.path.exists(estimator_path): os.remove(estimator_path)

        metric_info_path = "metric_info.json"
        metric_data = {
            "experiment_name": "Credit_Card_Default_Tuning_Online",
            "model_type": "RandomForestClassifier_Tuned",
            "metrics": {
                "train_accuracy": round(train_acc, 4),
                "test_accuracy": round(test_acc, 4),
                "test_f1_score": round(test_f1, 4)
            }
        }
        with open(metric_info_path, "w", encoding="utf-8") as f:
            json.dump(metric_data, f, indent=4)
        mlflow.log_artifact(metric_info_path)
        if os.path.exists(metric_info_path): os.remove(metric_info_path)

        mlflow.sklearn.log_model(model, "credit_card_rf_tuned_model")
        
        print(f"\nSukses! Seluruh parameter tuning dan 9 artefak berhasil dikirim ke Cloud DagsHub.")
        print(f"Hasil Akhir -> Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")