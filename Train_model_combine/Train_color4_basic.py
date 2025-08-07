import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import time
import warnings
warnings.filterwarnings('ignore')

def evaluate_models(X_train_balanced, y_train_balanced):
    """Đánh giá các mô hình sử dụng cross-validation."""
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
    scores = []
    train_times = []
    names = []
    for model in models:
        start = time.time()
        try:
            score = cross_val_score(estimator=model, X=X_train_balanced, y=y_train_balanced, scoring="accuracy", cv=10, n_jobs=-1).mean()
            scores.append(score)
        except Exception as e:
            print(f"Error model {model.__class__.__name__}: {e}")
            scores.append(None)
        end = time.time()
        train_times.append(end - start)
        names.append(model.__class__.__name__)
    return pd.DataFrame({'Model': names, 'Score': scores, 'Time': train_times})

def evaluate_dataset(csv_file, dataset_name):
    """Đánh giá mô hình trên một tập dữ liệu từ file CSV."""
    print(f"Evaluating dataset: {dataset_name}")

    df = pd.read_csv(csv_file)
    df["Label"] = df["Label"].replace({'positive': 0, 'negative': 1})
    X = df.drop(["Label"], axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_shuffled, y_train_shuffled)

    print(f"Before SMOTE: {np.bincount(y_train_shuffled)}")
    print(f"After SMOTE: {np.bincount(y_train_balanced)}")

    df_results = evaluate_models(X_train_balanced, y_train_balanced)
    print(f"\nModel evaluation results for {dataset_name}:")
    print(df_results)
    print("\n")

csv_files = [
    {'name': 'BC-15', 'csv': 'BC15_features.csv'},
    {'name': 'Huong_thom-1', 'csv': 'huongthom_features.csv'},
    {'name': 'Nep-87', 'csv': 'nep_features.csv'},
    {'name': 'Q-5', 'csv': 'q5_features.csv'},
    {'name': 'TBR-36', 'csv': 'tbr36_features.csv'},
    {'name': 'TBR-45', 'csv': 'tbr45_features.csv'},
    {'name': 'TH3-5', 'csv': 'th35_features.csv'},
    {'name': 'Thien_uu-8', 'csv': 'thienuu_features.csv'},
    {'name': 'Xi-23', 'csv': 'xi_features.csv'}
]

for dataset in csv_files:
    evaluate_dataset(dataset['csv'], dataset['name'])