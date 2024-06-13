from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from pdfminer.high_level import extract_text

current_dir = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(current_dir, 'template')
tfidf = TfidfVectorizer(stop_words='english', max_features=7351)

app = Flask(__name__, template_folder = template_path)

# Load the trained classifier
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

#Load the pre-fitted TfidfVectorizer
with open('tfidf.pkl', 'rb') as t:
    tfidf = pickle.load(t)

# Map category ID to category name
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

def predict_category(resume_text):
    # Transform the input resume text using the trained TfidfVectorizer
    input_features = tfidf.transform([resume_text])
    # Make the prediction using the loaded classifier
    prediction_id = clf.predict(input_features)[0]
    # Map category ID to category name
    category_name = category_mapping.get(prediction_id, "Unknown")
    return category_name

@app.route('/', methods=['GET', 'POST'])
def upload_resume():
    if request.method == 'POST':
        if 'resume' not in request.files:
            return render_template('index.html', error='No file part')
        
        resume_file = request.files['resume']

        if resume_file.filename == '':
            return render_template('index.html', error='No selected file')

        if resume_file:
            #save file to a temporary location
            temp_path = os.path.join(current_dir, 'temp.pdf')
            resume_file.save(temp_path)

            #Extract the text from the saved file
            resume_text = extract_text(temp_path)
            #remove the temporary file
            os.remove(temp_path)

            category_name = predict_category(resume_text)
            return render_template('index.html', category=category_name, filename=resume_file.filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)