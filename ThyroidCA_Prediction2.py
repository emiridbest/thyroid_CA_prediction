# -*- coding: utf-8 -*-

import os
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer, FunctionTransformer
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET = "Recurred"
NUMERIC_FEATURES = ["Age"]
ORDINAL_FEATURES = ["T", "N", "M", "Stage", "Risk", "Response", "Focality"]
NOMINAL_FEATURES = [
    "Gender", "Smoking", "Hx Smoking", "Hx Radiothreapy",
    "Thyroid Function", "Physical Examination", "Adenopathy", "Pathology"
]
FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES

# Clinical ordering for ordinal features
ORDERED_DEFS = {
    "T": ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"],
    "N": ["N0", "N1a", "N1b"],
    "M": ["M0", "M1"],
    "Stage": ["I", "II", "III", "IVA", "IVB"],
    "Risk": ["Low", "Intermediate", "High"],
    "Response": ["Excellent", "Indeterminate", "Biochemical Incomplete", "Structural Incomplete"],
    "Focality": ["Uni-Focal", "Multi-Focal"],
}

RANDOM_STATE = 42
N_FOLDS = 5
N_JOBS = -1
N_BOOTSTRAP = 1000

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Validate and clean input data."""
    warnings_list = []

    # Check for required columns
    missing_cols = [c for c in [*FEATURES, TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for missing values
    missing_counts = df[FEATURES + [TARGET]].isnull().sum()
    if missing_counts.any():
        warnings_list.append(f"Found missing values:\n{missing_counts[missing_counts > 0]}")

    # Check target values
    unique_targets = df[TARGET].unique()
    expected_targets = ["No", "Yes"]
    unexpected = set(unique_targets) - set(expected_targets)
    if unexpected:
        raise ValueError(f"Unexpected target values: {unexpected}")

    # Check ordinal feature values
    for feat in ORDINAL_FEATURES:
        if feat in df.columns:
            unique_vals = df[feat].dropna().unique()
            expected_vals = ORDERED_DEFS[feat]
            unexpected = set(unique_vals) - set(expected_vals)
            if unexpected:
                warnings_list.append(f"Unexpected values in '{feat}': {unexpected}")

    # Drop duplicates
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        warnings_list.append(f"Found and removed {n_duplicates} duplicate rows")
        df = df.drop_duplicates()

    return df, warnings_list

# ============================================================================
# PREPROCESSORS
# ============================================================================

def build_preprocessor_general():
    """Build preprocessor for general models (SMOTE-compatible)."""
    ordinal_categories = [ORDERED_DEFS[c] for c in ORDINAL_FEATURES]
    transformers = [
        ("num", MinMaxScaler(), NUMERIC_FEATURES),
        ("ord", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ), ORDINAL_FEATURES),
        ("nom", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), NOMINAL_FEATURES),
    ]
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )


def build_preprocessor_catnb():
    """Build preprocessor for CategoricalNB (discrete features only)."""
    transformers = []
    if NUMERIC_FEATURES:
        transformers.append((
            "num_bins",
            KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile"),
            NUMERIC_FEATURES
        ))
    cat_features = ORDINAL_FEATURES + NOMINAL_FEATURES
    if cat_features:
        transformers.append((
            "cat_ord",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            cat_features
        ))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )

# ============================================================================
# MODELS
# ============================================================================

def get_models():
    """Define all classifiers with hyperparameters."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=10,
            min_samples_split=5
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            max_depth=20,
            min_samples_split=5
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            learning_rate=0.1
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        ),
        "Logistic Regression": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=100,
            gamma=0.01,
            probability=True,
            random_state=RANDOM_STATE
        ),
        "Categorical Naive Bayes": CategoricalNB(alpha=1.0),
    }

# ============================================================================
# EVALUATION
# ============================================================================

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=N_BOOTSTRAP, alpha=0.05):
    """Calculate bootstrap confidence interval for a metric."""
    scores = []
    n_samples = len(y_true)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_pred[indices])
        scores.append(score)

    scores = np.array(scores)
    ci_lower = np.percentile(scores, alpha/2 * 100)
    ci_upper = np.percentile(scores, (1 - alpha/2) * 100)

    return ci_lower, ci_upper


def evaluate_on_test(model, X_test, y_test, model_name: str, out_dir: str):
    """
    Comprehensive evaluation on test set with visualizations.

    Args:
        model: Trained pipeline
        X_test: Test features
        y_test: Test labels
        model_name: Name for plots
        out_dir: Directory for saving plots
    """
    y_pred = model.predict(X_test)

    # Get probability scores
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred.astype(float)

    # Calculate test metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "avg_precision": average_precision_score(y_test, y_score),
    }

    # Bootstrap confidence intervals
    ci_acc = bootstrap_ci(y_test.values, y_pred, accuracy_score)
    ci_f1 = bootstrap_ci(y_test.values, y_pred, f1_score)
    ci_roc_auc = bootstrap_ci(y_test.values, y_score, roc_auc_score)
    ci_avg_precision = bootstrap_ci(y_test.values, y_score, average_precision_score)
    metrics["roc_auc_ci_lower"] = ci_roc_auc[0]
    metrics["roc_auc_ci_upper"] = ci_roc_auc[1]
    metrics["accuracy_ci_lower"] = ci_acc[0]
    metrics["accuracy_ci_upper"] = ci_acc[1]
    metrics["f1_ci_lower"] = ci_f1[0]
    metrics["f1_ci_upper"] = ci_f1[1]

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#f6f5f5')
    ax.set_facecolor('#f6f5f5')
    # Remove spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['roc_auc']:.3f})", lw=1, c='#990000')
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve - {model_name}", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"roc_{model_name.replace(' ', '_')}.png"), dpi=300)
    plt.close()

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#f6f5f5')
    ax.set_facecolor('#f6f5f5')
    # Remove spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(rec, prec, label=f"{model_name} (AP = {metrics['avg_precision']:.3f})", lw=1, c='#990000')
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve - {model_name}", fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pr_{model_name.replace(' ', '_')}.png"), dpi=300)
    plt.close()

    # Calibration Curve
    if hasattr(model, "predict_proba"):
        # get probabilities for the positive class
        y_prob = model.predict_proba(X_test)[:, 1]

        # choose fewer, quantile-based bins to stabilize small test sets
        n_bins = max(5, min(10, int(np.sqrt(len(y_test)))))
        frac_pos, mean_pred = calibration_curve(
            y_test, y_prob, n_bins=n_bins, strategy="quantile"
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#f6f5f5')
        ax.set_facecolor('#f6f5f5')

        # Tweak spines and plot
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.plot(mean_pred, frac_pos, marker='o', lw=1, label=model_name, c="#990000")
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label="Perfect Calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title(f"Calibration Curve - {model_name}", fontsize=14, fontweight='bold')
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"calibration_{model_name.replace(' ', '_')}.png"), dpi=300)
        plt.close()

    # Confusion Matrix (in %)
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12) * 100.0

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Reds", cbar_kws={'label': 'Percentage'})
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(f"Confusion Matrix (%) - {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cm_{model_name.replace(' ', '_')}.png"), dpi=300)
    plt.close()

    return metrics

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(data_path: str, out: str = "results", seed: int = RANDOM_STATE):
    """
    Run complete ML pipeline for thyroid cancer recurrence prediction.

    Args:
        data_path: Path to Thyroid_Diff.csv
        out: Output directory for results
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    os.makedirs(out, exist_ok=True)

    print("="*80)
    print("THYROID CANCER RECURRENCE PREDICTION PIPELINE")
    print("="*80)

    # Load and validate data
    print("\n[1/7] Loading and validating data...")
    df = pd.read_csv(data_path)
    df, warnings_list = validate_data(df)

    if warnings_list:
        print("\nData Validation Warnings:")
        for warning in warnings_list:
            print(f"  ⚠ {warning}")

    print(f"\nData shape: {df.shape}")
    print(f"Features: {len(FEATURES)}")
    print(f"Target distribution:\n{df[TARGET].value_counts()}")

    # Prepare features and target
    X = df[FEATURES].copy()
    y = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)

    print(f"\nClass balance: {y.value_counts()[0]} negative, {y.value_counts()[1]} positive")
    print(f"Imbalance ratio: {y.value_counts()[0]/y.value_counts()[1]:.2f}:1")

    # ========================================================================
    # CREATE CORRELATION HEATMAP
    # ========================================================================
    print("\n[2/7] Creating correlation heatmap...")

    # Create a copy for correlation analysis
    df_corr = df.copy()

    # Encode categorical variables for correlation
    # Ordinal features - use ordinal encoding
    for feat in ORDINAL_FEATURES:
        if feat in df_corr.columns and feat in ORDERED_DEFS:
            categories = ORDERED_DEFS[feat]
            df_corr[feat] = pd.Categorical(df_corr[feat], categories=categories, ordered=True)
            df_corr[feat] = df_corr[feat].cat.codes

    # Nominal features - use label encoding for correlation
    le = {}
    for feat in NOMINAL_FEATURES:
        if feat in df_corr.columns:
            le[feat] = LabelEncoder()
            df_corr[feat] = le[feat].fit_transform(df_corr[feat].astype(str))

    # Target encoding
    df_corr[TARGET] = df_corr[TARGET].map({"No": 0, "Yes": 1})

    # Calculate correlation matrix
    corr_matrix = df_corr[FEATURES + [TARGET]].corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create the heatmap with diverging colormap
    plt.figure(figsize=(16, 14))

    # Use RdBu_r: Red (positive) -> White (0) -> Blue (negative)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',  # Red-Blue reversed
        center=0,  # White at zero
        vmin=-1,
        vmax=1,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
        annot_kws={'fontsize': 8}
    )

    plt.title("Correlation Heatmap - Thyroid Cancer Recurrence Predictors\n(Red: Positive, Blue: Negative, White: Zero)",
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Correlation heatmap saved to: {os.path.join(out, 'correlation_heatmap.png')}")

    # Print top correlations with target
    target_corr = corr_matrix[TARGET].drop(TARGET).sort_values(ascending=False)
    print("\nTop 10 correlations with Recurrence:")
    for feat, corr in target_corr.head(10).items():
        print(f"  {feat:30s}: {corr:6.3f}")

    # Train-test split
    print("\n[3/7] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Build pipelines
    print("\n[4/7] Building ML pipelines...")
    preproc_general = build_preprocessor_general()
    preproc_catnb = build_preprocessor_catnb()
    cast_to_int = FunctionTransformer(lambda X: np.nan_to_num(X, nan=-1).astype(int))

    models = get_models()
    pipelines = {}

    # CategoricalNB gets special treatment
    pipelines["Categorical Naive Bayes"] = Pipeline(steps=[
        ("preprocess_cat", preproc_catnb),
        ("cast", cast_to_int),
        ("ros", RandomOverSampler(random_state=seed)),
        ("clf", models["Categorical Naive Bayes"]),
    ])

    # All other models use SMOTE
    for name, clf in models.items():
        if name == "Categorical Naive Bayes":
            continue
        pipelines[name] = Pipeline(steps=[
            ("preprocess", preproc_general),
            ("smote", SMOTE(k_neighbors=3, random_state=seed)),
            ("clf", clf),
        ])

    print(f"Built {len(pipelines)} pipelines")

    # Cross-validation
    print("\n[5/7] Running 5-fold cross-validation...")
    print("(SMOTE/ROS applied inside each fold - no data leakage)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "avg_precision": "average_precision",
    }

    cv_rows = []
    for name, pipe in pipelines.items():
        print(f"\n  Training {name}...")
        scores = cross_validate(
            pipe, X_train, y_train,
            cv=skf,
            scoring=scoring,
            return_train_score=False,
            n_jobs=N_JOBS
        )

        row = {"model": name}
        for m in scoring.keys():
            row[f"cv_{m}_mean"] = float(np.mean(scores[f"test_{m}"]))
            row[f"cv_{m}_std"] = float(np.std(scores[f"test_{m}"]))
        cv_rows.append(row)

        print(f"    ROC-AUC: {row['cv_roc_auc_mean']:.4f} ± {row['cv_roc_auc_std']:.4f}")
        print(f"    Accuracy: {row['cv_accuracy_mean']:.4f} ± {row['cv_accuracy_std']:.4f}")
        print(f"    F1 Score: {row['cv_f1_mean']:.4f} ± {row['cv_f1_std']:.4f}")

    cv_df = pd.DataFrame(cv_rows).sort_values("cv_roc_auc_mean", ascending=False)
    cv_df.to_csv(os.path.join(out, "cv_results.csv"), index=False)
    print(f"\n✓ CV results saved to: {os.path.join(out, 'cv_results.csv')}")

    # Test set evaluation
    print("\n[6/7] Evaluating on held-out test set...")
    test_rows = []
    roc_panel = []

    for name, pipe in pipelines.items():
        print(f"\n  Evaluating {name}...")
        pipe.fit(X_train, y_train)
        metrics = evaluate_on_test(pipe, X_test, y_test, model_name=name, out_dir=out)
        test_rows.append({"model": name, **metrics})

        print(f"    ROC-AUC: {metrics['roc_auc']:.4f} (95% CI : [{metrics['roc_auc_ci_lower']:.4f}, {metrics['roc_auc_ci_upper']:.4f}])")
        print(f"    Accuracy: {metrics['accuracy']:.4f} (95% CI: [{metrics['accuracy_ci_lower']:.4f}, {metrics['accuracy_ci_upper']:.4f}])")
        print(f"    F1 Score: {metrics['f1']:.4f} (95% CI: [{metrics['f1_ci_lower']:.4f}, {metrics['f1_ci_upper']:.4f}])")

        # Collect ROC data for combined plot
        if hasattr(pipe, "predict_proba"):
            y_score = pipe.predict_proba(X_test)[:, 1]
            print(len(np.unique(y_score)), "if")
        elif hasattr(pipe, "decision_function"):
            y_score = pipe.decision_function(X_test)
            print(len(np.unique(y_score)), "elif")
        else:
            y_score = pipe.predict(X_test).astype(float)
            print(len(np.unique(y_score)), "else")

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_panel.append((name, fpr, tpr, metrics["roc_auc"]))

    test_df = pd.DataFrame(test_rows).sort_values("roc_auc", ascending=False)
    test_df.to_csv(os.path.join(out, "test_results.csv"), index=False)
    print(f"\n✓ Test results saved to: {os.path.join(out, 'test_results.csv')}")

    # Combined ROC plot
    print("\n[7/7] Creating combined visualizations...")
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_panel)))

    for (name, fpr, tpr, auc_), color in zip(roc_panel, colors):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={auc_:.3f})", color=color)

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    plt.axhline(y=0.85, color='green', linestyle=':', linewidth=2,
                label='Clinical Benchmark (AUC≈0.85)', alpha=0.7)

    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curves - All Models (Hold-out Test Set)", fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "roc_all_models.png"), dpi=300)
    plt.close()

    # Save configuration
    with open(os.path.join(out, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "n_folds": N_FOLDS,
            "test_size": 0.20,
            "numeric_features": NUMERIC_FEATURES,
            "ordinal_features": ORDINAL_FEATURES,
            "nominal_features": NOMINAL_FEATURES,
            "target": TARGET,
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "class_balance_train": y_train.value_counts().to_dict(),
            "class_balance_test": y_test.value_counts().to_dict(),
        }, f, indent=2)

    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS - RANKED BY TEST ROC-AUC")
    print("="*80)

    for i, row in enumerate(test_df.itertuples(index=False), 1):
        print(f"\n{i}. {row.model}")
        print(f"   ROC-AUC: {row.roc_auc:.4f}")
        print(f"   Accuracy: {row.accuracy:.4f} (95% CI: [{row.accuracy_ci_lower:.4f}, {row.accuracy_ci_upper:.4f}])")
        print(f"   Precision: {row.precision:.4f}")
        print(f"   Recall: {row.recall:.4f}")
        print(f"   F1 Score: {row.f1:.4f}")
        print(f"   Avg Precision: {row.avg_precision:.4f}")

    print("\n" + "="*80)
    print(f"✓ All artifacts saved in: {os.path.abspath(out)}")
    print("="*80)

    return test_df, cv_df

# To run
# run_pipeline('/content/Thyroid_Diff.csv', out='/content/results', seed=42)