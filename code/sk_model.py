import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from imblearn.over_sampling import SMOTE

def prepare_data(data):
    """Prepare thyroid_diff data."""
    try:
        
        
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
    
        
        data["Age"] = MinMaxScaler().fit_transform(data[["Age"]])
    
        return  data
    except Exception as e:
        raise Exception(f"Data preparation failed: {str(e)}")

def train_sk_model():
    """Train RandomForest model for price prediction."""
    try:
        # Prepare data
        data = pd.read_csv("dataset/Thyroid_Diff.csv")
        data = data.drop_duplicates()
        data = prepare_data(data)
        X = data.drop(["Recurred","Hx Radiothreapy","Hx Smoking"],axis=1)

        #Prediction Column
        y = data["Recurred"]
        # Apply SMOTE
        os = SMOTE(random_state=0)
        X, y = os.fit_resample(X, y)
        # Use RandomForest for more stable predictions
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
         # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        # Fit the model
        model = rf.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate scores
        train_score =  accuracy_score(y_train, y_train_pred)
        test_score = accuracy_score(y_test, y_test_pred)
        
        return model, (train_score, test_score)
    except Exception as e:
        raise Exception(f"Model training failed: {str(e)}")

