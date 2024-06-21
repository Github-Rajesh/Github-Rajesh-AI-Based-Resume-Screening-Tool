from flask import Flask, render_template, request
from model import predict_match
from utils import extract_text_from_pdf
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    job_description = request.form['job_description']
    resume_file = request.files['resume']
    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    resume_file.save(resume_path)

    resume_text = extract_text_from_pdf(resume_path)
    match = predict_match(resume_text, job_description)
    return render_template('result.html', match=match)

if __name__ == '__main__':
    app.run(debug=True)
