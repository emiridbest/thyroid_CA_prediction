# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 00:36:49 2025

@author: ADMIN
"""
# http://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import missingno as msno

# Load the dataset
data = pd.read_csv("Thyroid_Diff.csv")

# Display the first few rows
print(data.head())

# Display dataset information
print(data.info())


# Color scheme for visualizations
colors = [ "#cc0000", "#990000", "#000000", "#550000"]
color_scheme = sns.color_palette(palette=colors)

sns.palplot(color_scheme, size=2.5)
plt.text(0.1, -0.6, "Colour Scheme to be used.", {"font": "serif", "size": 18, "weight": "bold"}, alpha=0.8)
plt.show()

# Check for null values
color = color_scheme
fig, ax = plt.subplots(figsize=(12, 4), dpi=70)
fig.patch.set_facecolor('#f6f5f5')
ax.set_facecolor('#f6f5f5')

msno.bar(data, sort='descending',
         color=color,
         ax=ax, fontsize=8,
         labels='off', filter='top')

ax.text(-1, 1.35, 'Visualization of Nullity of The Dataset', {'font': 'Serif', 'size': 24, 'color': 'black'}, alpha=0.9)
ax.text(-1, 1.2, 'Overall there are 383 datapoints present in \nthe given dataset.', {'font': 'Serif', 'size': 12, 'color': 'black'}, alpha=0.7)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                   ha='center', **{'font': 'Serif', 'size': 14, 'weight': 'normal', 'color': '#512b58'}, alpha=1)
ax.set_yticklabels('')
ax.spines['bottom'].set_visible(True)

plt.show()

# Display dataset statistics
print(data.describe().T)

# Check for null values
print(data.isnull().sum())

# Check for duplicated rows
print("Number of duplicated rows:", data.duplicated().sum())

# Drop duplicates
data = data.drop_duplicates()



# 1. Bar Plot for Gender Distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(y="Gender", hue="Recurred", data=data, palette=color_scheme, width=0.4)
ax.set_title("Gender Distribution", fontsize=12)
ax.set_ylabel("Gender", fontsize=10)
ax.set_xlabel("Count", fontsize=10)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.legend(title="Recurred", loc="lower right")
ax.set_xticks([])
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_facecolor('#f6f5f5')
plt.show()

# 2. Smoking vs Recurrence
plt.figure(figsize=(8, 5))
ax = sns.countplot(y="Smoking", hue="Recurred", data=data, palette=color_scheme, width=0.4)
ax.set_title("Smoking vs Recurrence", fontsize=12)
ax.set_ylabel("Smoking", fontsize=10)
ax.set_xlabel("Count", fontsize=10)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.legend(title="Recurred", loc="lower right")
ax.set_xticks([])
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_facecolor('#f6f5f5')
plt.show()

# 3. Age Distribution

plt.figure(figsize=(8, 5))
ax = sns.kdeplot(data=data, x="Age", hue="Recurred",  palette=color_scheme, fill=True, alpha=0.65)
ax.set_title("Age Distribution", fontsize=12)
ax.set_xlabel("Age", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_xticks([])
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_facecolor('#f6f5f5')
plt.show()

# 4. Age Distrubution Distplot
plt.figure(figsize=(8, 5))
ax = sns.displot(data=data, x="Age", hue="Recurred", kind="hist", kde=True, palette=color_scheme, bins=14)
ax.fig.suptitle("Age Distribution", fontsize=12, )
ax.fig.set_facecolor('#f6f5f5')
for spine in ['top', 'right', 'bottom', 'left']:
    ax.axes[0, 0].spines[spine].set_visible(False)

plt.show()



# 5. Box Plot for Age by Recurrence
plt.figure(figsize=(8, 5))
ax = sns.boxplot(x="Recurred", y="Age", hue="Recurred", data=data, palette=color_scheme, width=0.4)
ax.set_title("Age by Recurrence", fontsize=12)
ax.set_xlabel("Recurred", fontsize=10)
ax.set_ylabel("Age", fontsize=10)
ax.set_xticks([])
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_facecolor('#f6f5f5')
plt.show()

# 6. Smoking  vs Recurrence
plt.figure(figsize=(8, 5))
ax = sns.countplot(y="Smoking", hue="Recurred", data=data, palette=color_scheme, width=0.4)
ax.set_title("Smoking Actively vs Recurrence", fontsize=12)
ax.set_ylabel("Smoking", fontsize=10)
ax.set_xlabel("Count", fontsize=10)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.legend(title="Recurred", loc="lower right")
ax.set_xticks([])
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_facecolor('#f6f5f5')
plt.show()

# 7. Stage Distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(x="Stage", data=data, hue="Recurred", palette=color_scheme, width=0.4)
ax.set_title("Stage Distribution", fontsize=12)
ax.set_xlabel("Stage", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.legend(title="Recurred", loc="upper right")
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_facecolor('#f6f5f5')
plt.show()



# 8. Recurrence by Response
plt.figure(figsize=(8, 5))
ax =sns.countplot(x="Response", hue="Recurred", data=data, palette=color_scheme, width=0.4)
ax.set_title("Recurrence by Response", fontsize=12)
ax.set_xlabel("Response", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.legend(title="Recurred", loc="upper right")
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_facecolor('#f6f5f5')
plt.xticks(rotation=10)
plt.show()

# 9. Thyroid Function vs Recurrence
plt.figure(figsize=(8, 5))
ax = sns.countplot(x="Thyroid Function", hue="Recurred", data=data, palette=color_scheme, width=0.4)
ax.set_title("Thyroid Function vs Recurrence", fontsize=12)
ax.set_xlabel("Thyroid Function", fontsize=10)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.legend(title="Recurred", loc="upper right")
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_facecolor('#f6f5f5')
plt.xticks(rotation=10)
plt.show()

# 10. Tumor Size vs Recurrence
plt.figure(figsize=(8, 5))
ax = sns.countplot(x="T", hue="Recurred", data=data, palette=color_scheme, width=0.4)
ax.set_title("Tumor Size vs Recurrence", fontsize=12)
ax.set_xlabel("Tumor Size", fontsize=10)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.legend(title="Recurred", loc="upper right")
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_facecolor('#f6f5f5')
plt.show()

# 11. Nodal Involvement vs Recurrence
plt.figure(figsize=(8, 5))
ax = sns.countplot(y="N", hue="Recurred", data=data, palette=color_scheme, width=0.4)
ax.set_title("Nodal Involvement vs Recurrence", fontsize=12)
ax.set_ylabel("Nodal Involvement", fontsize=10)
ax.set_xlabel("Count", fontsize=10)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.legend(title="Recurred", loc="lower right")
ax.set_xticks([])
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_facecolor('#f6f5f5')
plt.show()

# 12. Risk Distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(y="Risk", data=data, palette=color_scheme, hue="Recurred", width=0.4)
ax.set_title("Risk Distribution", fontsize=16)
ax.set_xlabel("Count", fontsize=12)
ax.set_ylabel("Risk", fontsize=12)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.legend(title="Recurred", loc="upper right")
ax.set_xticks([])
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_facecolor('#f6f5f5')
plt.show()

# 13. Adenopathy Distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(x="Adenopathy", data=data, palette=color_scheme, hue="Recurred", width=0.4)
ax.set_title("Adenopathy Distribution", fontsize=16)
ax.set_ylabel("Adenopathy", fontsize=12)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.legend(title="Recurred", loc="upper right")
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks([])
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_facecolor('#f6f5f5')
plt.show()

# 14. Pathology Distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(x="Pathology", data=data, palette=color_scheme, hue="Recurred", width=0.4)
ax.set_title("Pathology Distribution", fontsize=16)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.legend(title="Recurred", loc="upper right")
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_facecolor('#f6f5f5')
plt.yticks([])
plt.show()

data["T"] = data["T"].replace({"T1a": 0, "T1b": 1, "T2": 2, "T3a": 3, "T3b": 4, "T4a": 5, "T4b": 6})
data["Thyroid Function"] = data["Thyroid Function"].replace({"Euthyroid": 0, "Subclinical Hypothyroidism": 1, "Clinical Hypothyroidism": 2, "Subclinical Hyperthyroidism": 3, "Clinical Hyperthyroidism": 4})
data["N"] = data["N"].replace({"N0": 0, "N1a": 1, "N1b": 2})
data["Response"] = data["Response"].replace({"Excellent": 0, "Indeterminate": 1, "Biochemical Incomplete": 2, "Structural Incomplete": 3 })
data["Physical Examination"] = data["Physical Examination"].replace({"Normal": 0, "Single nodular goiter-right": 1, "Single nodular goiter-left": 2, "Multinodular goiter": 3, "Diffuse goiter": 4 })
data["Pathology"] = data["Pathology"].replace({"Micropapillary": 0, "Papillary": 1, "Follicular": 2, "Hurthel cell": 3 })
data["Risk"] = data["Risk"].replace({"Low": 0, "Intermediate": 1, "High": 2})
data["Adenopathy"] = data["Adenopathy"].replace({"No": 0, "Right": 1, "Left": 2, "Posterior": 3, "Bilateral": 4, "Extensive": 5 })
data["Recurred"] = data["Recurred"].replace({"No": 0, "Yes": 1})
data["Stage"] = data["Stage"].replace({"I": 0, "II": 1, "III": 2, "IVA": 3, "IVB": 4})
data["Focality"] = data["Focality"].replace({"Uni-Focal": 0, "Multi-Focal": 1})

# Encode categorical columns
le = LabelEncoder()

categorical_cols = data.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Display value counts for the target column
data.info()
data.head(5)


# Correlation heatmap
plt.figure(figsize=(12, 7))
ax = sns.heatmap(data.corr(), annot=True, cmap=color_scheme)
ax.set_title("Correlation Heatmap", fontsize=12)
plt.show()



# scale age column
data["Age"] = MinMaxScaler().fit_transform(data[["Age"]])
data.describe()
# check for imbalance
data["Recurred"].value_counts()



X = data.drop(["Recurred" ],axis=1)

#Prediction Column
y = data["Recurred"]
print(y.value_counts())

# 15. Class Distribution before SMOTE
Target = pd.DataFrame({"Recurrence": y})

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=Target["Recurrence"].value_counts().index, y=Target["Recurrence"].value_counts().values,
                 hue=Target["Recurrence"].value_counts().index, palette=color_scheme, width=0.4)

ax.set_title("Class Distribution For Target Variable", fontsize=16)
ax.bar_label(ax.containers[0], fontsize=20)
ax.bar_label(ax.containers[1], fontsize=20)
ax.legend(title="Recurred", loc="upper right")
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_facecolor('#f6f5f5')
plt.yticks([])
plt.show()

# Apply SMOTE
os = SMOTE(random_state=0)
X, y = os.fit_resample(X, y)
print(y.value_counts())

# 16. Class Distribution after SMOTE
Target = pd.DataFrame({"Recurrence": y})

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=Target["Recurrence"].value_counts().index, y=Target["Recurrence"].value_counts().values,
                 hue=Target["Recurrence"].value_counts().index, palette=color_scheme, width=0.4)
ax.set_title("Class Distribution For Target Variable", fontsize=16)
ax.bar_label(ax.containers[0], fontsize=20)
ax.bar_label(ax.containers[1], fontsize=20)
ax.legend(title="Recurred", loc="upper right")
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_facecolor('#f6f5f5')
plt.yticks([])
plt.show()
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100,random_state=42 ),
    "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100,random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=4, weights="distance", algorithm="brute"),
    "Logistic Regression": LogisticRegression(penalty="l2",C=1.0,solver="liblinear"),
    "Support Vector Machine": SVC(kernel="rbf", C=100, gamma=0.01),
    "Categorical Naive Bayes": CategoricalNB(alpha=0, force_alpha=True),

}
# Comparison dictionary
model_comparison = {}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Importance', y='Feature', data=importances, palette=color_scheme)
ax.set_xticks([])
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_facecolor('#f6f5f5')
plt.title('Feature Importance from Random Forest', fontsize=16)
plt.tight_layout()
plt.show()


X = X.drop(["Hx Radiothreapy","Hx Smoking", "Risk" ],axis=1)




for name, classifier in classifiers.items():
    print(f"\nTraining {name}")

    # Fit and predict
    model = classifier.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_metrics = {
        "Accuracy": accuracy_score(y_train, y_train_pred),
        "Precision": precision_score(y_train, y_train_pred),
        "Recall": recall_score(y_train, y_train_pred),
        "F1 Score": f1_score(y_train, y_train_pred),
        "AUC": roc_auc_score(y_train, y_train_pred),
        "Confusion Matrix Train": confusion_matrix(y_train, y_train_pred),
    }

    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
        "F1 Score": f1_score(y_test, y_test_pred),
        "AUC": roc_auc_score(y_test, y_test_pred),
        "Confusion Matrix Test": confusion_matrix(y_test, y_test_pred)
    }

    # Calculate percentage confusion matrix
    cm = test_metrics["Confusion Matrix Test"]
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        cm_percentage,  
        annot=True,
        cmap=color_scheme,
        fmt='.1f',
        cbar_kws={'format': '%.0f%%'} ,
        annot_kws={'fontsize': 48}
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"\n Confusion Matrix {name} (%)", fontsize=30)  
    plt.show()
    # Store results
    model_comparison[name] = {
        "Train Metrics": train_metrics,
        "Test Metrics": test_metrics,
        "Confusion Matrix": cm 
    }

# Print comparison
for model_name, results in model_comparison.items():
    print(f"\n{model_name} Performance:")
    print("Train Metrics:")
    for metric, value in results["Train Metrics"].items():
        print(f"{metric}: {value}")
    print("\nTest Metrics:")
    for metric, value in results["Test Metrics"].items():
        print(f"{metric}: {value}")



