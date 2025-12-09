import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    data = pd.read_csv('sample_data.csv')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['candidate_profile'] + " " + data['job_description'])
    y = data['fit_score']
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Model trained.")

def predict(candidate_text, job_text):
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    X = vectorizer.transform([candidate_text + " " + job_text])
    score = model.predict(X)[0]
    return max(0, min(100, score))

if __name__ == '__main__':
    train_model()
