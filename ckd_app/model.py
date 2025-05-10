# model.py
import os
import pandas as pd
import joblib  
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'voting_classifier.pkl')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'models', 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(settings.BASE_DIR, 'models', 'label_encoder.pkl')
DATA_PATH = os.path.join(settings.BASE_DIR, 'data', 'ckd_data.csv')

def train_model():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=['Unnamed: 0'], axis=1, errors='ignore')
    data = data.drop(columns=['id'])
    data = data.fillna(data.mean())  

    label_encoder = LabelEncoder()
    data['classification'] = label_encoder.fit_transform(data['classification'])  

    X = data.drop('classification', axis=1)
    y = data['classification']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    log_reg = LogisticRegression()
    rf = RandomForestClassifier(random_state=42)
    xgb_model = xgb.XGBClassifier(eval_metric='logloss')

    voting_clf = VotingClassifier(estimators=[
        ('log_reg', log_reg),
        ('rf', rf),
        ('xgb', xgb_model)
    ], voting='soft')

    cv_scores = cross_val_score(voting_clf, X_resampled, y_resampled, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean cross-validation accuracy: {cv_scores.mean() * 100:.2f}%")

    voting_clf.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(voting_clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    y_pred = voting_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return voting_clf, scaler, label_encoder

def predict(input_data):
    try:
        voting_clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    except FileNotFoundError:
        return "Model not found. Please train the model first."

    input_data = pd.DataFrame([input_data])
    input_data_scaled = scaler.transform(input_data)
    prediction = voting_clf.predict(input_data_scaled)
    return label_encoder.inverse_transform(prediction)
