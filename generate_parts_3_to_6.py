from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT / "notebooks"

METADATA = {
    "kernelspec": {
        "display_name": "venv",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.14.4",
    },
}


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


COMMON_CLASSIFIER_LOAD = """data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Feature count:", X.shape[1])
print("Classes:", list(data.target_names))
"""


COMMON_CLASSIFIER_MONITOR = """df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = y
df.head()
"""


COMMON_REGRESSION_LOAD = """data_path = os.path.abspath("../src/data/hour.csv")
df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))
"""


COMMON_REGRESSION_SAMPLE = """df["cnt_bin"] = pd.qcut(df["cnt"], q=10, duplicates="drop")

df, _ = train_test_split(
    df,
    train_size=2400,
    random_state=42,
    stratify=df["cnt_bin"]
)

df = df.drop(columns=["cnt_bin"]).reset_index(drop=True)

print("Sampled dataset shape:", df.shape)
"""


COMMON_REGRESSION_PREP = """feature_columns = [
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed"
]
target_column = "cnt"

X = df[feature_columns].values
y = df[target_column].values
"""


COMMON_REGRESSION_MONITOR = """df[feature_columns + [target_column]].head()
"""


def build_part3():
    nb = nbf.v4.new_notebook(metadata=METADATA)
    nb.cells = [
        md(
            "# Homework 1 - Part 3: Linear SVM Classifier\n"
            "### Burak Kurucay - 210104004049\n"
            "---\n\n"
            "This section models a linear SVM classifier on the Breast Cancer Wisconsin dataset.\n"
            "The model is evaluated with 6-fold cross validation using ROC curves, confusion matrices,\n"
            "classification metrics, and runtime analysis.\n"
        ),
        md("## Code:"),
        md("### Import the necessary modules"),
        code(
            """import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
"""
        ),
        md(
            "### Load the dataset\n"
            "Load the data and show the number of samples, features, and class names."
        ),
        code(COMMON_CLASSIFIER_LOAD),
        md("### Monitor the data\nJust to see the dataset loaded."),
        code(COMMON_CLASSIFIER_MONITOR),
        md(
            "### Train By 6-Fold Cross Validation\n"
            "For each fold, fit a linear SVM, compute ROC curves from the decision scores, and choose the\n"
            "classification threshold from the training ROC curve using Youden's J statistic (`TPR - FPR`)."
        ),
        code(
            """kf = KFold(n_splits=6, shuffle=True, random_state=42)

fold_results = []
overall_start_time = time.time()

for fold_idx, (train_index, test_index) in enumerate(kf.split(X), start=1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(random_state=42, dual="auto", max_iter=20000))
    ])

    fold_start_time = time.time()
    model.fit(X_train, y_train)

    train_scores = model.decision_function(X_train)
    test_scores = model.decision_function(X_test)
    fold_end_time = time.time()

    train_fpr, train_tpr, train_thresholds = roc_curve(y_train, train_scores)
    test_fpr, test_tpr, _ = roc_curve(y_test, test_scores)

    best_idx = np.argmax(train_tpr - train_fpr)
    best_threshold = train_thresholds[best_idx]

    train_predictions = (train_scores >= best_threshold).astype(int)
    test_predictions = (test_scores >= best_threshold).astype(int)

    fold_results.append({
        "fold": fold_idx,
        "threshold": best_threshold,
        "train_accuracy": accuracy_score(y_train, train_predictions),
        "test_accuracy": accuracy_score(y_test, test_predictions),
        "train_precision": precision_score(y_train, train_predictions, zero_division=0),
        "test_precision": precision_score(y_test, test_predictions, zero_division=0),
        "train_recall": recall_score(y_train, train_predictions, zero_division=0),
        "test_recall": recall_score(y_test, test_predictions, zero_division=0),
        "train_f1": f1_score(y_train, train_predictions, zero_division=0),
        "test_f1": f1_score(y_test, test_predictions, zero_division=0),
        "train_auc": roc_auc_score(y_train, train_scores),
        "test_auc": roc_auc_score(y_test, test_scores),
        "train_confusion_matrix": confusion_matrix(y_train, train_predictions),
        "test_confusion_matrix": confusion_matrix(y_test, test_predictions),
        "train_fpr": train_fpr,
        "train_tpr": train_tpr,
        "test_fpr": test_fpr,
        "test_tpr": test_tpr,
        "runtime_seconds": fold_end_time - fold_start_time
    })

overall_end_time = time.time()
mean_runtime = np.mean([r["runtime_seconds"] for r in fold_results])
total_runtime = overall_end_time - overall_start_time
"""
        ),
        md("## Results:"),
        md("### Test Fold Confusion Matrices"),
        code(
            """fig, axes = plt.subplots(2, 3, figsize=(7, 5))
axes = axes.flatten()

for i, result in enumerate(fold_results):
    ax = axes[i]
    ax.axis("off")

    cm = result["test_confusion_matrix"]
    table_data = [
        [cm[0][0], cm[0][1]],
        [cm[1][0], cm[1][1]]
    ]

    table = ax.table(
        cellText=table_data,
        rowLabels=["Actual 0", "Actual 1"],
        colLabels=["Pred 0", "Pred 1"],
        loc="center"
    )
    table.scale(1, 1.5)
    ax.set_title(f"Fold {result['fold']}")

plt.tight_layout()
plt.show()
"""
        ),
        md("### Classification Metrics"),
        code(
            """metric_rows = []

for result in fold_results:
    metric_rows.append({
        "Fold": result["fold"],
        "Threshold": result["threshold"],
        "Train Accuracy": result["train_accuracy"],
        "Test Accuracy": result["test_accuracy"],
        "Train Precision": result["train_precision"],
        "Test Precision": result["test_precision"],
        "Train Recall": result["train_recall"],
        "Test Recall": result["test_recall"],
        "Train F1": result["train_f1"],
        "Test F1": result["test_f1"],
        "Train AUC": result["train_auc"],
        "Test AUC": result["test_auc"]
    })

df_metrics = pd.DataFrame(metric_rows)
df_metrics.iloc[:, 1:] = df_metrics.iloc[:, 1:].round(4)
display(df_metrics)

folds = [r["fold"] for r in fold_results]
test_accuracies = [r["test_accuracy"] for r in fold_results]
test_precisions = [r["test_precision"] for r in fold_results]
test_recalls = [r["test_recall"] for r in fold_results]
test_f1s = [r["test_f1"] for r in fold_results]
test_aucs = [r["test_auc"] for r in fold_results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(folds, test_accuracies, marker="o", label="Test Accuracy")
axes[0].plot(folds, test_precisions, marker="o", label="Test Precision")
axes[0].plot(folds, test_recalls, marker="o", label="Test Recall")
axes[0].plot(folds, test_f1s, marker="o", label="Test F1")
axes[0].set_xlabel("Fold")
axes[0].set_ylabel("Score")
axes[0].set_title("Test Classification Metrics per Fold")
axes[0].set_xticks(folds)
axes[0].legend()
axes[0].grid(True)

axes[1].plot(folds, test_aucs, marker="o", color="purple", label="Test AUC")
axes[1].set_xlabel("Fold")
axes[1].set_ylabel("AUC")
axes[1].set_title("Test AUC per Fold")
axes[1].set_xticks(folds)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

df_metrics_summary = pd.DataFrame({
    "Metric": [
        "Mean Train Accuracy",
        "Mean Test Accuracy",
        "Mean Train Precision",
        "Mean Test Precision",
        "Mean Train Recall",
        "Mean Test Recall",
        "Mean Train F1",
        "Mean Test F1",
        "Mean Train AUC",
        "Mean Test AUC"
    ],
    "Value": [
        df_metrics["Train Accuracy"].mean(),
        df_metrics["Test Accuracy"].mean(),
        df_metrics["Train Precision"].mean(),
        df_metrics["Test Precision"].mean(),
        df_metrics["Train Recall"].mean(),
        df_metrics["Test Recall"].mean(),
        df_metrics["Train F1"].mean(),
        df_metrics["Test F1"].mean(),
        df_metrics["Train AUC"].mean(),
        df_metrics["Test AUC"].mean()
    ]
})

df_metrics_summary["Value"] = df_metrics_summary["Value"].round(4)
display(df_metrics_summary)
"""
        ),
        md(
            "### Runtime Performance\n"
            "The runtime of the linear SVM classifier was measured for each fold during the 6-fold cross validation process.  \n"
            "Both per-fold runtimes and the total cross-validation runtime are reported below."
        ),
        code(
            """runtime_rows = []

for result in fold_results:
    runtime_rows.append({
        "Fold": result["fold"],
        "Runtime (seconds)": result["runtime_seconds"]
    })

df_runtime = pd.DataFrame(runtime_rows)
df_runtime["Runtime (seconds)"] = df_runtime["Runtime (seconds)"].round(4)
display(df_runtime)
"""
        ),
        code(
            """df_runtime_summary = pd.DataFrame({
    "Metric": [
        "Average runtime per fold",
        "Total 6-fold CV runtime"
    ],
    "Runtime (seconds)": [
        mean_runtime,
        total_runtime
    ]
})

df_runtime_summary["Runtime (seconds)"] = df_runtime_summary["Runtime (seconds)"].round(4)
display(df_runtime_summary)
"""
        ),
        md(
            "### Selected Fold Performance Results\n"
            "Select the fold whose test AUC is closest to the mean test AUC across all folds."
        ),
        code(
            """mean_test_auc = np.mean([r["test_auc"] for r in fold_results])

selected_fold = min(
    fold_results,
    key=lambda r: abs(r["test_auc"] - mean_test_auc)
)
"""
        ),
        code(
            """df_selected_performance = pd.DataFrame({
    "Metric": [
        "Threshold",
        "Train Accuracy", "Test Accuracy",
        "Train Precision", "Test Precision",
        "Train Recall", "Test Recall",
        "Train F1", "Test F1",
        "Train AUC", "Test AUC"
    ],
    "Value": [
        selected_fold["threshold"],
        selected_fold["train_accuracy"], selected_fold["test_accuracy"],
        selected_fold["train_precision"], selected_fold["test_precision"],
        selected_fold["train_recall"], selected_fold["test_recall"],
        selected_fold["train_f1"], selected_fold["test_f1"],
        selected_fold["train_auc"], selected_fold["test_auc"]
    ]
})

df_selected_performance["Value"] = df_selected_performance["Value"].round(4)
display(df_selected_performance)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, matrix_key, title in [
    (axes[0], "train_confusion_matrix", "Train Confusion Matrix"),
    (axes[1], "test_confusion_matrix", "Test Confusion Matrix")
]:
    ax.axis("off")
    cm = selected_fold[matrix_key]
    table = ax.table(
        cellText=[[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]],
        rowLabels=["Actual 0", "Actual 1"],
        colLabels=["Pred 0", "Pred 1"],
        loc="center"
    )
    table.scale(1, 1.5)
    ax.set_title(title)

plt.tight_layout()
plt.show()
"""
        ),
        md(
            "### Selected Fold ROC Curve\n"
            "ROC is drawn only for the selected fold. The SVM does not output probabilities here, so the curve is\n"
            "computed from `decision_function` scores. We sweep the threshold over these scores and plot TPR against FPR."
        ),
        code(
            """plt.figure(figsize=(6, 5))
plt.plot(
    selected_fold["train_fpr"],
    selected_fold["train_tpr"],
    label=f"Train ROC (AUC = {selected_fold['train_auc']:.4f})"
)
plt.plot(
    selected_fold["test_fpr"],
    selected_fold["test_tpr"],
    label=f"Test ROC (AUC = {selected_fold['test_auc']:.4f})"
)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Selected Fold ({selected_fold['fold']}) ROC Curves")
plt.legend()
plt.grid(True)
plt.show()
"""
        ),
        md(
            "## Comments:\n"
            "- The linear SVM uses the decision scores to build ROC curves, and the threshold is selected from the training fold with Youden's J statistic.\n"
            "- If test AUC and test F1 stay close across folds, the classifier is behaving consistently on unseen data.\n"
            "- The selected fold gives both training and testing results for one representative case of the 6-fold cross validation setup.\n"
        ),
    ]
    return nb


def build_part4():
    nb = nbf.v4.new_notebook(metadata=METADATA)
    nb.cells = [
        md(
            "# Homework 1 - Part 4: Linear SVM Regressor\n"
            "### Burak Kurucay - 210104004049\n"
            "---\n\n"
            "This section models a linear SVM regressor on the Bike Sharing hourly dataset.\n"
            "The model is evaluated with 6-fold cross validation using regression metrics, prediction visualizations,\n"
            "and runtime analysis.\n"
        ),
        md("## Code:"),
        md("### Import the necessary modules"),
        code(
            """import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, r2_score
"""
        ),
        md(
            "### Load the dataset\n"
            "Load the hourly bike sharing dataset and show the number of rows and columns."
        ),
        code(COMMON_REGRESSION_LOAD),
        md(
            "### Sample the dataset\n"
            "The original dataset is large for repeated from-scratch style experiments, so sample 2400 rows while preserving the target distribution by stratifying on binned `cnt` values."
        ),
        code(COMMON_REGRESSION_SAMPLE),
        md(
            "### Prepare the dataset\n"
            "Use `cnt` as the regression target. Exclude `casual` and `registered` to avoid target leakage, and drop identifier/date columns."
        ),
        code(COMMON_REGRESSION_PREP),
        md("### Monitor the data\nJust to see the dataset loaded."),
        code(COMMON_REGRESSION_MONITOR),
        md("### Train By 6-Fold Cross Validation"),
        code(
            """kf = KFold(n_splits=6, shuffle=True, random_state=42)

fold_results = []
overall_start_time = time.time()

for fold_idx, (train_index, test_index) in enumerate(kf.split(X), start=1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", LinearSVR(random_state=42, dual="auto", max_iter=20000))
    ])

    fold_start_time = time.time()
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    fold_end_time = time.time()

    train_rmse = np.sqrt(np.mean((y_train - train_predictions) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - test_predictions) ** 2))

    fold_results.append({
        "fold": fold_idx,
        "train_mae": mean_absolute_error(y_train, train_predictions),
        "test_mae": mean_absolute_error(y_test, test_predictions),
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": r2_score(y_train, train_predictions),
        "test_r2": r2_score(y_test, test_predictions),
        "y_test": y_test,
        "test_predictions": test_predictions,
        "runtime_seconds": fold_end_time - fold_start_time
    })

overall_end_time = time.time()
mean_runtime = np.mean([r["runtime_seconds"] for r in fold_results])
total_runtime = overall_end_time - overall_start_time
"""
        ),
        md("## Results:"),
        md("### Regression Metrics"),
        code(
            """metric_rows = []

for result in fold_results:
    metric_rows.append({
        "Fold": result["fold"],
        "Train MAE": result["train_mae"],
        "Test MAE": result["test_mae"],
        "Train RMSE": result["train_rmse"],
        "Test RMSE": result["test_rmse"],
        "Train R2": result["train_r2"],
        "Test R2": result["test_r2"]
    })

df_metrics = pd.DataFrame(metric_rows)
df_metrics.iloc[:, 1:] = df_metrics.iloc[:, 1:].round(4)
display(df_metrics)

folds = [r["fold"] for r in fold_results]
train_maes = [r["train_mae"] for r in fold_results]
test_maes = [r["test_mae"] for r in fold_results]
train_rmses = [r["train_rmse"] for r in fold_results]
test_rmses = [r["test_rmse"] for r in fold_results]
train_r2s = [r["train_r2"] for r in fold_results]
test_r2s = [r["test_r2"] for r in fold_results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(folds, train_maes, marker="o", label="Train MAE")
axes[0].plot(folds, test_maes, marker="o", label="Test MAE")
axes[0].plot(folds, train_rmses, marker="o", linestyle="--", label="Train RMSE")
axes[0].plot(folds, test_rmses, marker="o", linestyle="--", label="Test RMSE")
axes[0].set_xlabel("Fold")
axes[0].set_ylabel("Error")
axes[0].set_title("Error Metrics per Fold")
axes[0].set_xticks(folds)
axes[0].legend()
axes[0].grid(True)

axes[1].plot(folds, train_r2s, marker="o", label="Train R2")
axes[1].plot(folds, test_r2s, marker="o", label="Test R2")
axes[1].set_xlabel("Fold")
axes[1].set_ylabel("R2 Score")
axes[1].set_title("R2 per Fold")
axes[1].set_xticks(folds)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

df_metrics_summary = pd.DataFrame({
    "Metric": [
        "Mean Train MAE",
        "Mean Test MAE",
        "Mean Train RMSE",
        "Mean Test RMSE",
        "Mean Train R2",
        "Mean Test R2"
    ],
    "Value": [
        df_metrics["Train MAE"].mean(),
        df_metrics["Test MAE"].mean(),
        df_metrics["Train RMSE"].mean(),
        df_metrics["Test RMSE"].mean(),
        df_metrics["Train R2"].mean(),
        df_metrics["Test R2"].mean()
    ]
})

df_metrics_summary["Value"] = df_metrics_summary["Value"].round(4)
display(df_metrics_summary)
"""
        ),
        md(
            "### Runtime Performance\n"
            "The runtime of the linear SVM regressor was measured for each fold during the 6-fold cross validation process.  \n"
            "Both per-fold runtimes and the total cross-validation runtime are reported below."
        ),
        code(
            """runtime_rows = []

for result in fold_results:
    runtime_rows.append({
        "Fold": result["fold"],
        "Runtime (seconds)": result["runtime_seconds"]
    })

df_runtime = pd.DataFrame(runtime_rows)
df_runtime["Runtime (seconds)"] = df_runtime["Runtime (seconds)"].round(4)
display(df_runtime)
"""
        ),
        code(
            """df_runtime_summary = pd.DataFrame({
    "Metric": [
        "Average runtime per fold",
        "Total 6-fold CV runtime"
    ],
    "Runtime (seconds)": [
        mean_runtime,
        total_runtime
    ]
})

df_runtime_summary["Runtime (seconds)"] = df_runtime_summary["Runtime (seconds)"].round(4)
display(df_runtime_summary)
"""
        ),
        md(
            "### Selected Fold Performance Results\n"
            "Select the fold whose test RMSE is closest to the mean test RMSE across all folds."
        ),
        code(
            """mean_test_rmse = np.mean([r["test_rmse"] for r in fold_results])

selected_fold = min(
    fold_results,
    key=lambda r: abs(r["test_rmse"] - mean_test_rmse)
)
"""
        ),
        code(
            """df_selected_performance = pd.DataFrame({
    "Metric": ["Train MAE", "Test MAE", "Train RMSE", "Test RMSE", "Train R2", "Test R2"],
    "Value": [
        selected_fold["train_mae"],
        selected_fold["test_mae"],
        selected_fold["train_rmse"],
        selected_fold["test_rmse"],
        selected_fold["train_r2"],
        selected_fold["test_r2"]
    ]
})

df_selected_performance["Value"] = df_selected_performance["Value"].round(4)
display(df_selected_performance)

plt.figure(figsize=(10, 4))
plt.bar(
    df_selected_performance["Metric"],
    df_selected_performance["Value"],
    color=["#4C72B0", "#55A868", "#CCB974", "#64B5CD", "#8C8C8C", "#DA8BC3"]
)
plt.xlabel("Metric")
plt.ylabel("Value")
plt.title(f"Selected Fold ({selected_fold['fold']}) Metrics")
plt.xticks(rotation=30)
plt.grid(axis="y")
plt.show()

selected_predictions_df = pd.DataFrame({
    "Actual": selected_fold["y_test"][:20],
    "Predicted": np.round(selected_fold["test_predictions"][:20], 4)
})

display(selected_predictions_df)

plt.figure(figsize=(6, 6))
plt.scatter(selected_fold["y_test"], selected_fold["test_predictions"], alpha=0.5)
min_val = min(selected_fold["y_test"].min(), selected_fold["test_predictions"].min())
max_val = max(selected_fold["y_test"].max(), selected_fold["test_predictions"].max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal Fit")
plt.xlabel("Actual cnt")
plt.ylabel("Predicted cnt")
plt.title(f"Selected Fold ({selected_fold['fold']}) Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()
"""
        ),
        md(
            "## Comments:\n"
            "- The linear SVR is trained on scaled features because SVM-based models are sensitive to feature magnitude.\n"
            "- Lower MAE and RMSE values indicate predictions closer to the true hourly bike counts, while higher R2 indicates better fit.\n"
            "- The selected fold reports one representative training/testing case from the 6-fold cross validation setup.\n"
        ),
    ]
    return nb


def build_part5():
    nb = nbf.v4.new_notebook(metadata=METADATA)
    nb.cells = [
        md(
            "# Homework 1 - Part 5: Decision Tree Classifier\n"
            "### Burak Kurucay - 210104004049\n"
            "---\n\n"
            "This section models a decision tree classifier on the Breast Cancer Wisconsin dataset.\n"
            "Two pruning strategies are compared: pre-pruning with `max_depth=4` and post-pruning with\n"
            "`ccp_alpha=0.01`. The notebook also extracts a set of rules from one trained tree.\n"
        ),
        md("## Code:"),
        md("### Import the necessary modules"),
        code(
            """import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.tree import DecisionTreeClassifier, _tree
"""
        ),
        md(
            "### Load the dataset\n"
            "Load the data and show the number of samples, features, and class names."
        ),
        code(COMMON_CLASSIFIER_LOAD),
        md("### Monitor the data\nJust to see the dataset loaded."),
        code(COMMON_CLASSIFIER_MONITOR),
        md(
            "### Define the pruning strategies\n"
            "- Strategy 1 uses pre-pruning by limiting the tree depth with `max_depth=4`.\n"
            "- Strategy 2 uses post-pruning by applying cost-complexity pruning with `ccp_alpha=0.01`."
        ),
        code(
            """strategies = {
    "Max Depth Pruning": DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        random_state=42
    ),
    "Cost Complexity Pruning": DecisionTreeClassifier(
        criterion="gini",
        ccp_alpha=0.01,
        random_state=42
    )
}
"""
        ),
        md("### Train By 6-Fold Cross Validation"),
        code(
            """kf = KFold(n_splits=6, shuffle=True, random_state=42)

strategy_results = {}

for strategy_name, base_model in strategies.items():
    fold_results = []
    overall_start_time = time.time()

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = DecisionTreeClassifier(**base_model.get_params())

        fold_start_time = time.time()
        model.fit(X_train, y_train)
        train_scores = model.predict_proba(X_train)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        fold_end_time = time.time()

        train_fpr, train_tpr, _ = roc_curve(y_train, train_scores)
        test_fpr, test_tpr, _ = roc_curve(y_test, test_scores)

        fold_results.append({
            "fold": fold_idx,
            "train_accuracy": accuracy_score(y_train, train_predictions),
            "test_accuracy": accuracy_score(y_test, test_predictions),
            "train_precision": precision_score(y_train, train_predictions, zero_division=0),
            "test_precision": precision_score(y_test, test_predictions, zero_division=0),
            "train_recall": recall_score(y_train, train_predictions, zero_division=0),
            "test_recall": recall_score(y_test, test_predictions, zero_division=0),
            "train_f1": f1_score(y_train, train_predictions, zero_division=0),
            "test_f1": f1_score(y_test, test_predictions, zero_division=0),
            "train_auc": roc_auc_score(y_train, train_scores),
            "test_auc": roc_auc_score(y_test, test_scores),
            "train_confusion_matrix": confusion_matrix(y_train, train_predictions),
            "test_confusion_matrix": confusion_matrix(y_test, test_predictions),
            "train_fpr": train_fpr,
            "train_tpr": train_tpr,
            "test_fpr": test_fpr,
            "test_tpr": test_tpr,
            "model": model,
            "runtime_seconds": fold_end_time - fold_start_time
        })

    overall_end_time = time.time()

    strategy_results[strategy_name] = {
        "fold_results": fold_results,
        "mean_runtime": np.mean([r["runtime_seconds"] for r in fold_results]),
        "total_runtime": overall_end_time - overall_start_time
    }
"""
        ),
        md("## Results:"),
        md("### Strategy Comparison"),
        code(
            """summary_rows = []

for strategy_name, result_bundle in strategy_results.items():
    fold_results = result_bundle["fold_results"]
    summary_rows.append({
        "Strategy": strategy_name,
        "Mean Train Accuracy": np.mean([r["train_accuracy"] for r in fold_results]),
        "Mean Test Accuracy": np.mean([r["test_accuracy"] for r in fold_results]),
        "Mean Train Precision": np.mean([r["train_precision"] for r in fold_results]),
        "Mean Test Precision": np.mean([r["test_precision"] for r in fold_results]),
        "Mean Train Recall": np.mean([r["train_recall"] for r in fold_results]),
        "Mean Test Recall": np.mean([r["test_recall"] for r in fold_results]),
        "Mean Train F1": np.mean([r["train_f1"] for r in fold_results]),
        "Mean Test F1": np.mean([r["test_f1"] for r in fold_results]),
        "Mean Train AUC": np.mean([r["train_auc"] for r in fold_results]),
        "Mean Test AUC": np.mean([r["test_auc"] for r in fold_results]),
        "Average Runtime (seconds)": result_bundle["mean_runtime"]
    })

df_strategy_summary = pd.DataFrame(summary_rows)
df_strategy_summary.iloc[:, 1:] = df_strategy_summary.iloc[:, 1:].round(4)
display(df_strategy_summary)
"""
        ),
        md("### Select the better pruning strategy"),
        code(
            """best_strategy_name = df_strategy_summary.sort_values(
    by=["Mean Test Accuracy", "Mean Test F1"],
    ascending=False
).iloc[0]["Strategy"]

best_results = strategy_results[best_strategy_name]["fold_results"]
print("Selected strategy:", best_strategy_name)
"""
        ),
        md("### Test Fold Confusion Matrices"),
        code(
            """fig, axes = plt.subplots(2, 3, figsize=(7, 5))
axes = axes.flatten()

for i, result in enumerate(best_results):
    ax = axes[i]
    ax.axis("off")
    cm = result["test_confusion_matrix"]

    table = ax.table(
        cellText=[[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]],
        rowLabels=["Actual 0", "Actual 1"],
        colLabels=["Pred 0", "Pred 1"],
        loc="center"
    )
    table.scale(1, 1.5)
    ax.set_title(f"Fold {result['fold']}")

plt.tight_layout()
plt.show()
"""
        ),
        md("### Classification Metrics"),
        code(
            """metric_rows = []

for result in best_results:
    metric_rows.append({
        "Fold": result["fold"],
        "Train Accuracy": result["train_accuracy"],
        "Test Accuracy": result["test_accuracy"],
        "Train Precision": result["train_precision"],
        "Test Precision": result["test_precision"],
        "Train Recall": result["train_recall"],
        "Test Recall": result["test_recall"],
        "Train F1": result["train_f1"],
        "Test F1": result["test_f1"],
        "Train AUC": result["train_auc"],
        "Test AUC": result["test_auc"]
    })

df_metrics = pd.DataFrame(metric_rows)
df_metrics.iloc[:, 1:] = df_metrics.iloc[:, 1:].round(4)
display(df_metrics)

folds = [r["fold"] for r in best_results]
train_accs = [r["train_accuracy"] for r in best_results]
test_accs = [r["test_accuracy"] for r in best_results]
test_precisions = [r["test_precision"] for r in best_results]
test_recalls = [r["test_recall"] for r in best_results]
test_f1s = [r["test_f1"] for r in best_results]
test_aucs = [r["test_auc"] for r in best_results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(folds, train_accs, marker="o", label="Train Accuracy")
axes[0].plot(folds, test_accs, marker="o", label="Test Accuracy")
axes[0].set_xlabel("Fold")
axes[0].set_ylabel("Score")
axes[0].set_title("Accuracy per Fold")
axes[0].set_xticks(folds)
axes[0].legend()
axes[0].grid(True)

axes[1].plot(folds, test_precisions, marker="o", label="Test Precision")
axes[1].plot(folds, test_recalls, marker="o", label="Test Recall")
axes[1].plot(folds, test_f1s, marker="o", label="Test F1")
axes[1].plot(folds, test_aucs, marker="o", label="Test AUC")
axes[1].set_xlabel("Fold")
axes[1].set_ylabel("Score")
axes[1].set_title("Test Classification Metrics per Fold")
axes[1].set_xticks(folds)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
"""
        ),
        md("### Runtime Performance"),
        code(
            """runtime_rows = []

for result in best_results:
    runtime_rows.append({
        "Fold": result["fold"],
        "Runtime (seconds)": result["runtime_seconds"]
    })

df_runtime = pd.DataFrame(runtime_rows)
df_runtime["Runtime (seconds)"] = df_runtime["Runtime (seconds)"].round(4)
display(df_runtime)

df_runtime_summary = pd.DataFrame({
    "Metric": [
        "Average runtime per fold",
        "Total 6-fold CV runtime"
    ],
    "Runtime (seconds)": [
        strategy_results[best_strategy_name]["mean_runtime"],
        strategy_results[best_strategy_name]["total_runtime"]
    ]
})

df_runtime_summary["Runtime (seconds)"] = df_runtime_summary["Runtime (seconds)"].round(4)
display(df_runtime_summary)
"""
        ),
        md(
            "### Selected Fold Performance Results\n"
            "Select the fold whose test accuracy is closest to the mean test accuracy across all folds for the selected strategy."
        ),
        code(
            """mean_test_acc = np.mean([r["test_accuracy"] for r in best_results])

selected_fold = min(
    best_results,
    key=lambda r: abs(r["test_accuracy"] - mean_test_acc)
)
"""
        ),
        code(
            """df_selected_performance = pd.DataFrame({
    "Metric": [
        "Train Accuracy", "Test Accuracy",
        "Train Precision", "Test Precision",
        "Train Recall", "Test Recall",
        "Train F1", "Test F1",
        "Train AUC", "Test AUC"
    ],
    "Value": [
        selected_fold["train_accuracy"], selected_fold["test_accuracy"],
        selected_fold["train_precision"], selected_fold["test_precision"],
        selected_fold["train_recall"], selected_fold["test_recall"],
        selected_fold["train_f1"], selected_fold["test_f1"],
        selected_fold["train_auc"], selected_fold["test_auc"]
    ]
})

df_selected_performance["Value"] = df_selected_performance["Value"].round(4)
display(df_selected_performance)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, matrix_key, title in [
    (axes[0], "train_confusion_matrix", "Train Confusion Matrix"),
    (axes[1], "test_confusion_matrix", "Test Confusion Matrix")
]:
    ax.axis("off")
    cm = selected_fold[matrix_key]
    table = ax.table(
        cellText=[[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]],
        rowLabels=["Actual 0", "Actual 1"],
        colLabels=["Pred 0", "Pred 1"],
        loc="center"
    )
    table.scale(1, 1.5)
    ax.set_title(title)

plt.tight_layout()
plt.show()
"""
        ),
        md(
            "### Selected Fold ROC Curve\n"
            "ROC is drawn only for the selected fold. The decision tree provides class scores with `predict_proba`,\n"
            "and the curve is obtained by sweeping the threshold on the positive-class score and plotting TPR versus FPR."
        ),
        code(
            """plt.figure(figsize=(6, 5))
plt.plot(
    selected_fold["train_fpr"],
    selected_fold["train_tpr"],
    label=f"Train ROC (AUC = {selected_fold['train_auc']:.4f})"
)
plt.plot(
    selected_fold["test_fpr"],
    selected_fold["test_tpr"],
    label=f"Test ROC (AUC = {selected_fold['test_auc']:.4f})"
)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Selected Fold ({selected_fold['fold']}) ROC Curves")
plt.legend()
plt.grid(True)
plt.show()
"""
        ),
        md("### Extract Rules From the Selected Decision Tree"),
        code(
            """def tree_classifier_rules(model, feature_names, class_names):
    tree = model.tree_
    rules = []

    def walk(node_id, conditions):
        if tree.feature[node_id] != _tree.TREE_UNDEFINED:
            feature_name = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]

            walk(
                tree.children_left[node_id],
                conditions + [f"{feature_name} <= {threshold:.4f}"]
            )
            walk(
                tree.children_right[node_id],
                conditions + [f"{feature_name} > {threshold:.4f}"]
            )
        else:
            class_index = int(np.argmax(tree.value[node_id][0]))
            predicted_class = class_names[class_index]
            samples = int(tree.n_node_samples[node_id])
            rule = "IF " + " AND ".join(conditions) + f" THEN class = {predicted_class} (samples={samples})"
            rules.append(rule)

    walk(0, [])
    return rules


rules = tree_classifier_rules(
    selected_fold["model"],
    data.feature_names,
    data.target_names
)

df_rules = pd.DataFrame({
    "Rule": rules
})

display(df_rules)
"""
        ),
        md(
            "## Comments:\n"
            "- Two pruning strategies are compared: pre-pruning with `max_depth=4` and post-pruning with `ccp_alpha=0.01`.\n"
            "- The better strategy is chosen by mean test accuracy and mean test F1, then its fold-level and selected-fold results are reported.\n"
            "- The extracted rules show how one representative decision tree can be interpreted as a set of human-readable IF-THEN statements.\n"
        ),
    ]
    return nb


def build_part6():
    nb = nbf.v4.new_notebook(metadata=METADATA)
    nb.cells = [
        md(
            "# Homework 1 - Part 6: Decision Tree Regressor\n"
            "### Burak Kurucay - 210104004049\n"
            "---\n\n"
            "This section models a decision tree regressor on the Bike Sharing hourly dataset.\n"
            "The model is evaluated with 6-fold cross validation, and one trained tree is converted into a\n"
            "set of regression rules.\n"
        ),
        md("## Code:"),
        md("### Import the necessary modules"),
        code(
            """import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.metrics import mean_absolute_error, r2_score
"""
        ),
        md(
            "### Load the dataset\n"
            "Load the hourly bike sharing dataset and show the number of rows and columns."
        ),
        code(COMMON_REGRESSION_LOAD),
        md(
            "### Sample the dataset\n"
            "The original dataset is large for repeated tree experiments, so sample 2400 rows while preserving the target distribution by stratifying on binned `cnt` values."
        ),
        code(COMMON_REGRESSION_SAMPLE),
        md(
            "### Prepare the dataset\n"
            "Use `cnt` as the regression target. Exclude `casual` and `registered` to avoid target leakage, and drop identifier/date columns."
        ),
        code(COMMON_REGRESSION_PREP),
        md("### Monitor the data\nJust to see the dataset loaded."),
        code(COMMON_REGRESSION_MONITOR),
        md("### Train By 6-Fold Cross Validation"),
        code(
            """kf = KFold(n_splits=6, shuffle=True, random_state=42)

fold_results = []
overall_start_time = time.time()

for fold_idx, (train_index, test_index) in enumerate(kf.split(X), start=1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=6,
        min_samples_leaf=10,
        random_state=42
    )

    fold_start_time = time.time()
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    fold_end_time = time.time()

    train_rmse = np.sqrt(np.mean((y_train - train_predictions) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - test_predictions) ** 2))

    fold_results.append({
        "fold": fold_idx,
        "train_mae": mean_absolute_error(y_train, train_predictions),
        "test_mae": mean_absolute_error(y_test, test_predictions),
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": r2_score(y_train, train_predictions),
        "test_r2": r2_score(y_test, test_predictions),
        "model": model,
        "y_test": y_test,
        "test_predictions": test_predictions,
        "runtime_seconds": fold_end_time - fold_start_time
    })

overall_end_time = time.time()
mean_runtime = np.mean([r["runtime_seconds"] for r in fold_results])
total_runtime = overall_end_time - overall_start_time
"""
        ),
        md("## Results:"),
        md("### Regression Metrics"),
        code(
            """metric_rows = []

for result in fold_results:
    metric_rows.append({
        "Fold": result["fold"],
        "Train MAE": result["train_mae"],
        "Test MAE": result["test_mae"],
        "Train RMSE": result["train_rmse"],
        "Test RMSE": result["test_rmse"],
        "Train R2": result["train_r2"],
        "Test R2": result["test_r2"]
    })

df_metrics = pd.DataFrame(metric_rows)
df_metrics.iloc[:, 1:] = df_metrics.iloc[:, 1:].round(4)
display(df_metrics)

folds = [r["fold"] for r in fold_results]
train_maes = [r["train_mae"] for r in fold_results]
test_maes = [r["test_mae"] for r in fold_results]
train_rmses = [r["train_rmse"] for r in fold_results]
test_rmses = [r["test_rmse"] for r in fold_results]
train_r2s = [r["train_r2"] for r in fold_results]
test_r2s = [r["test_r2"] for r in fold_results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(folds, train_maes, marker="o", label="Train MAE")
axes[0].plot(folds, test_maes, marker="o", label="Test MAE")
axes[0].plot(folds, train_rmses, marker="o", linestyle="--", label="Train RMSE")
axes[0].plot(folds, test_rmses, marker="o", linestyle="--", label="Test RMSE")
axes[0].set_xlabel("Fold")
axes[0].set_ylabel("Error")
axes[0].set_title("Error Metrics per Fold")
axes[0].set_xticks(folds)
axes[0].legend()
axes[0].grid(True)

axes[1].plot(folds, train_r2s, marker="o", label="Train R2")
axes[1].plot(folds, test_r2s, marker="o", label="Test R2")
axes[1].set_xlabel("Fold")
axes[1].set_ylabel("R2 Score")
axes[1].set_title("R2 per Fold")
axes[1].set_xticks(folds)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

df_metrics_summary = pd.DataFrame({
    "Metric": [
        "Mean Train MAE",
        "Mean Test MAE",
        "Mean Train RMSE",
        "Mean Test RMSE",
        "Mean Train R2",
        "Mean Test R2"
    ],
    "Value": [
        df_metrics["Train MAE"].mean(),
        df_metrics["Test MAE"].mean(),
        df_metrics["Train RMSE"].mean(),
        df_metrics["Test RMSE"].mean(),
        df_metrics["Train R2"].mean(),
        df_metrics["Test R2"].mean()
    ]
})

df_metrics_summary["Value"] = df_metrics_summary["Value"].round(4)
display(df_metrics_summary)
"""
        ),
        md("### Runtime Performance"),
        code(
            """runtime_rows = []

for result in fold_results:
    runtime_rows.append({
        "Fold": result["fold"],
        "Runtime (seconds)": result["runtime_seconds"]
    })

df_runtime = pd.DataFrame(runtime_rows)
df_runtime["Runtime (seconds)"] = df_runtime["Runtime (seconds)"].round(4)
display(df_runtime)

df_runtime_summary = pd.DataFrame({
    "Metric": [
        "Average runtime per fold",
        "Total 6-fold CV runtime"
    ],
    "Runtime (seconds)": [
        mean_runtime,
        total_runtime
    ]
})

df_runtime_summary["Runtime (seconds)"] = df_runtime_summary["Runtime (seconds)"].round(4)
display(df_runtime_summary)
"""
        ),
        md(
            "### Selected Fold Performance Results\n"
            "Select the fold whose test RMSE is closest to the mean test RMSE across all folds."
        ),
        code(
            """mean_test_rmse = np.mean([r["test_rmse"] for r in fold_results])

selected_fold = min(
    fold_results,
    key=lambda r: abs(r["test_rmse"] - mean_test_rmse)
)
"""
        ),
        code(
            """df_selected_performance = pd.DataFrame({
    "Metric": ["Train MAE", "Test MAE", "Train RMSE", "Test RMSE", "Train R2", "Test R2"],
    "Value": [
        selected_fold["train_mae"],
        selected_fold["test_mae"],
        selected_fold["train_rmse"],
        selected_fold["test_rmse"],
        selected_fold["train_r2"],
        selected_fold["test_r2"]
    ]
})

df_selected_performance["Value"] = df_selected_performance["Value"].round(4)
display(df_selected_performance)

plt.figure(figsize=(10, 4))
plt.bar(
    df_selected_performance["Metric"],
    df_selected_performance["Value"],
    color=["#4C72B0", "#55A868", "#CCB974", "#64B5CD", "#8C8C8C", "#DA8BC3"]
)
plt.xlabel("Metric")
plt.ylabel("Value")
plt.title(f"Selected Fold ({selected_fold['fold']}) Metrics")
plt.xticks(rotation=30)
plt.grid(axis="y")
plt.show()

selected_predictions_df = pd.DataFrame({
    "Actual": selected_fold["y_test"][:20],
    "Predicted": np.round(selected_fold["test_predictions"][:20], 4)
})

display(selected_predictions_df)

plt.figure(figsize=(6, 6))
plt.scatter(selected_fold["y_test"], selected_fold["test_predictions"], alpha=0.5)
min_val = min(selected_fold["y_test"].min(), selected_fold["test_predictions"].min())
max_val = max(selected_fold["y_test"].max(), selected_fold["test_predictions"].max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal Fit")
plt.xlabel("Actual cnt")
plt.ylabel("Predicted cnt")
plt.title(f"Selected Fold ({selected_fold['fold']}) Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()
"""
        ),
        md("### Extract Rules From the Selected Decision Tree"),
        code(
            """def tree_regressor_rules(model, feature_names):
    tree = model.tree_
    rules = []

    def walk(node_id, conditions):
        if tree.feature[node_id] != _tree.TREE_UNDEFINED:
            feature_name = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]

            walk(
                tree.children_left[node_id],
                conditions + [f"{feature_name} <= {threshold:.4f}"]
            )
            walk(
                tree.children_right[node_id],
                conditions + [f"{feature_name} > {threshold:.4f}"]
            )
        else:
            prediction = float(tree.value[node_id][0][0])
            samples = int(tree.n_node_samples[node_id])
            rule = "IF " + " AND ".join(conditions) + f" THEN predicted cnt = {prediction:.4f} (samples={samples})"
            rules.append(rule)

    walk(0, [])
    return rules


rules = tree_regressor_rules(selected_fold["model"], feature_columns)

df_rules = pd.DataFrame({
    "Rule": rules
})

display(df_rules)
"""
        ),
        md(
            "## Comments:\n"
            "- The decision tree regressor is kept shallow enough to avoid an excessively large rule set while still modeling nonlinear relationships.\n"
            "- Lower MAE and RMSE mean better prediction quality, and R2 shows how much target variance is explained by the tree.\n"
            "- The extracted rules convert one representative regression tree into a readable set of IF-THEN statements for interpretation.\n"
        ),
    ]
    return nb


def main():
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    notebooks = {
        "part3.ipynb": build_part3(),
        "part4.ipynb": build_part4(),
        "part5.ipynb": build_part5(),
        "part6.ipynb": build_part6(),
    }

    for filename, notebook in notebooks.items():
        output_path = NOTEBOOKS_DIR / filename
        with output_path.open("w", encoding="utf-8") as f:
            nbf.write(notebook, f)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
