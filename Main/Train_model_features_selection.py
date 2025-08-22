import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
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
from sklearn.feature_selection import SelectFdr, RFE, SelectFromModel, RFECV, chi2, f_classif

def evaluate_dataset(csv_file, feature_selection_method='filter'):
    print('\n')
    print("Review data set:", csv_file)
    df = pd.read_csv(csv_file)
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])
    X = df.drop(["Label"], axis=1)
    y = df["Label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    if feature_selection_method == 'filter':
        X_train_abs = np.abs(X_train)
        X_test_abs  = np.abs(X_test)
        selector = SelectFdr(score_func=chi2, alpha=0.05)  
        X_train_selected = selector.fit_transform(X_train_abs, y_train)
        X_test_selected  = selector.transform(X_test_abs)

    elif feature_selection_method == 'anova':
        selector = SelectFdr(score_func=f_classif, alpha=0.05)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected  = selector.transform(X_test)

    elif feature_selection_method == 'wrapper':
        svm_estimator = SVC(kernel='linear', random_state=42)
        rfe = RFE(estimator=svm_estimator, n_features_to_select=80)
        X_train_selected = rfe.fit_transform(X_train, y_train)
        X_test_selected  = rfe.transform(X_test)

    elif feature_selection_method == 'embedded_rf':
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        selector = SelectFromModel(rf, threshold='median', prefit=True)
        X_train_selected = selector.transform(X_train)
        X_test_selected  = selector.transform(X_test)

    elif feature_selection_method == 'embedded_dt':
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        selector_dt = SelectFromModel(dt, threshold='median', prefit=True)
        X_train_selected = selector_dt.transform(X_train)
        X_test_selected  = selector_dt.transform(X_test)

    elif feature_selection_method == 'rfecv':
        svm_estimator = SVC(kernel='linear', random_state=42)
        rfecv = RFECV(estimator=svm_estimator, step=1, cv=5, scoring='accuracy')
        X_train_selected = rfecv.fit_transform(X_train, y_train)
        X_test_selected  = rfecv.transform(X_test)

    else:
        X_train_selected = X_train
        X_test_selected = X_test
    
    print("Shape :", X_train_selected.shape)

    X_train_shuffled, y_train_shuffled = shuffle(X_train_selected, y_train, random_state=42)
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
    
    results_df = pd.DataFrame({'Model': names, 'Score': scores, 'Time (s)': train_times})
    print("Model evaluation results: ")
    print(results_df)
if __name__ == '__main__':
    datasets = [r"C:\Users\LENOVO\Downloads\BC15.csv", r"C:\Users\LENOVO\Downloads\Huongthom.csv", r"C:\Users\LENOVO\Downloads\Nep87.csv", r"C:\Users\LENOVO\Downloads\Q5.csv", r"C:\Users\LENOVO\Downloads\TBR36.csv", r"C:\Users\LENOVO\Downloads\TBR45.csv", r"C:\Users\LENOVO\Downloads\TH35.csv", r"C:\Users\LENOVO\Downloads\Thienuu.csv", r"C:\Users\LENOVO\Downloads\Xi23.csv"]
    feature_method = 'embedded_rf'
    
    for csv_file in datasets:
        evaluate_dataset(csv_file, feature_selection_method=feature_method)

