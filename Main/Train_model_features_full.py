import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

def evaluate_dataset(csv_file):
    print("\nReview data set:", csv_file)
    df = pd.read_csv(csv_file)
    
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])
    
    X = df.drop(["Label"], axis=1)
    y = df["Label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    
    print("Shape:", X_train.shape, y_train.shape)

    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_shuffled, y_train_shuffled)

    models = [
        KNeighborsClassifier(),
        LogisticRegression(),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        CatBoostClassifier(verbose=0),
        ExtraTreesClassifier()
    ]
    
    estimators = [
        ('lr', LogisticRegression()),
        ('svc', SVC(probability=True)),
        ('catboost', CatBoostClassifier(verbose=0))
    ]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=10, n_jobs=-1)
    models.append(stacking_model)
    
    scores = []
    train_times = []
    names = []
    
    for model in models:
        start = time.time()
        try:
            score = cross_val_score(estimator=model, X=X_train_balanced, y=y_train_balanced,
                                    scoring="accuracy", cv=10, n_jobs=-1).mean()
            scores.append(score)
        except Exception as e:
            print(f"Error model {model.__class__.__name__}: {e}")
            scores.append(None)
        end = time.time()
        train_times.append(end - start)
        names.append(model.__class__.__name__)
    
    df_results = pd.DataFrame({'Model': names, 'Score': scores, 'Time (s)': train_times})
    print("Model evaluation results:")
    print(df_results)

if __name__ == '__main__':
    datasets = [r"C:\Users\LENOVO\Downloads\BC15.csv", r"C:\Users\LENOVO\Downloads\Huongthom.csv", r"C:\Users\LENOVO\Downloads\Nep87.csv", r"C:\Users\LENOVO\Downloads\Q5.csv", r"C:\Users\LENOVO\Downloads\TBR36.csv", r"C:\Users\LENOVO\Downloads\TBR45.csv", r"C:\Users\LENOVO\Downloads\TH35.csv", r"C:\Users\LENOVO\Downloads\Thienuu.csv", r"C:\Users\LENOVO\Downloads\Xi23.csv"]

    for csv_file in datasets:
        if not pd.io.common.file_exists(csv_file):
            print(f"Tập tin {csv_file} không tồn tại. Vui lòng kiểm tra lại đường dẫn!")
        else:
            evaluate_dataset(csv_file)