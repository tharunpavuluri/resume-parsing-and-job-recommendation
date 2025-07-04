from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from resume_parser import parse_resume
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load job data from CSV
def load_job_data(file_path='jobs_data.csv'):
    try:
        job_data = pd.read_csv(file_path, encoding='latin1')
        required_columns = ['Position', 'Company', 'Job_Description', 'Location']
        if not all(col in job_data.columns for col in required_columns):
            raise ValueError(f"CSV missing required columns: {', '.join(required_columns)}")
        return job_data.fillna("").to_dict(orient='records')
    except Exception as e:
        print(f"[Error] Loading job data: {e}")
        return []

# Load job listings at startup
job_listings = load_job_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        return redirect(url_for('index'))

    file = request.files['resume']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    allowed_extensions = {'pdf', 'docx'}
    if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return render_template('index.html', error="Unsupported file format. Please upload a PDF or DOCX.")

    # Parse resume and extract skills
    resume_skills = parse_resume(file)
    print("Extracted Skills:", resume_skills)

    if not resume_skills:
        return render_template('index.html', error="No skills found or unsupported file content.")

    # Match extracted skills to jobs
    matched_jobs = match_jobs(resume_skills)

    if not matched_jobs:
        return render_template('results.html', jobs=[], message="No matching jobs found.")

    dropdown_locations = sorted(set(job['location'] for job in matched_jobs if job['location']))

    return render_template('results.html', jobs=matched_jobs, dropdown_locations=dropdown_locations)

def match_jobs(resume_skills, threshold=0.4):
    """Matches resume skills with job descriptions using TF-IDF + Cosine Similarity."""
    resume_text = ' '.join(resume_skills).lower()
    job_descriptions = [job['Job_Description'].lower() for job in job_listings]

    if not job_descriptions:
        return []

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    job_tfidf = vectorizer.fit_transform(job_descriptions)
    resume_tfidf = vectorizer.transform([resume_text])

    # Compute similarity
    similarities = cosine_similarity(resume_tfidf, job_tfidf)[0]

    # Filter jobs based on similarity threshold
    matched_jobs = []
    for i, score in enumerate(similarities):
        if score >= threshold:
            job = job_listings[i]
            matched_jobs.append({
                'title': job.get('Position', 'N/A'),
                'company': job.get('Company', 'N/A'),
                'location': job.get('Location', 'Not Specified'),
                'score': round(score * 100, 2)
            })

    return sorted(matched_jobs, key=lambda x: x['score'], reverse=True)

if __name__ == '__main__':
    app.run(debug=True)
