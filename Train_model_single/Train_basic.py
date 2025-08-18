import numpy as np
import pandas as pd
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

csv_files = [
    'basic_bc15.csv',
    'basic_huongthom.csv',
    'basic_nep87.csv',
    'basic_q5.csv',
    'basic_tbr36.csv',
    'basic_tbr45.csv',
    'basic_th35.csv',
    'basic_thienuu.csv',
    'basic_xi23.csv'
]

def train_and_evaluate_models(csv_file):
    """Đọc dữ liệu từ CSV, xử lý và đánh giá các mô hình."""
    df = pd.read_csv(csv_file)
    df["Label"] = df["Label"].replace({'positive': 0, 'negative': 1})

    X = df.drop(["Label"], axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nDataset: {csv_file}")
    print(type(X_train), type(y_train))
    print(X_train.shape, y_train.shape)

    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_shuffled, y_train_shuffled)

    print("Before SMOTE:", np.bincount(y_train_shuffled))
    print("After SMOTE:", np.bincount(y_train_balanced))

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

    df_results = pd.DataFrame({'Model': names, 'Score': scores, 'Time': train_times})
    print(df_results)
    return df_results

for csv_file in csv_files:
    train_and_evaluate_models(csv_file)