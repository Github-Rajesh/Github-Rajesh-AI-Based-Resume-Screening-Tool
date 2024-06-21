import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess_text

# Dummy data loading and model training for demonstration
data = {
    'resume': ["Data Scientist with experience in Python and Machine Learning.", 
               "Software Engineer with experience in Java and Cloud Computing."],
    'job_description': ["Looking for a Data Scientist skilled in Python and Machine Learning.", 
                        "Hiring a Software Engineer proficient in Java and Cloud Computing."],
    'label': [1, 1]  # Binary labels: 1 for match, 0 for no match
}

df = pd.DataFrame(data)

# Preprocess text
df['resume'] = df['resume'].apply(preprocess_text)
df['job_description'] = df['job_description'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=3000)
X_resumes = vectorizer.fit_transform(df['resume']).toarray()
X_jobs = vectorizer.transform(df['job_description']).toarray()

# Combine features for model input
X = X_resumes + X_jobs
y = df['label']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

def predict_match(resume, job_description):
    resume = preprocess_text(resume)
    job_description = preprocess_text(job_description)
    X_resume = vectorizer.transform([resume]).toarray()
    X_job = vectorizer.transform([job_description]).toarray()
    X = X_resume + X_job
    return model.predict(X)[0]
