from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pickle
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model and TF-IDF vectorizer
model = pickle.load(open(r"senti\backend\resume.pickle", "rb"))
tfidf = pickle.load(open(r"senti\backend\tfidf.pickle", "rb"))

# Resume category mapping
category_mapping = {
    15: "JAVA DEV", 23: "TESTING", 8: "DEVOPS ENGINEER", 20: "PYTHON DEVELOPER",
    24: "WEB DESIGNER", 12: "HR", 3: "BLOCKCHAIN", 10: "ETL DEVELOPER",
    18: "OPERATIONS MANAGER", 6: "DATA SCIENCE", 22: "SALES",
    16: "MECHANICAL ENGINEER", 1: "ARTS", 7: "DATABASE",
    11: "ELECTRICAL ENGINEER", 14: "HEALTH AND FITNESS", 19: "PMO",
    4: "BUSINESS ANALYST", 9: "DOTNET DEVELOPER", 2: "AUTOMATION TESTING",
    17: "NETWORK SECURITY ENGINEER", 21: "SAP DEVELOPER", 5: "CIVIL ENGINEER",
    0: "ADVOCATE"
}

def clean_resume(txt):
    """Clean resume text by removing unwanted characters and URLs."""
    clean_text = re.sub(r"http\S+", " ", txt)  # Remove URLs
    clean_text = re.sub(r"\b(RT|CC)\b", " ", clean_text)  # Remove retweets (RT) and copy-comments (CC)
    clean_text = re.sub(r"#\S+", " ", clean_text)  # Remove hashtags
    clean_text = re.sub(r"@\S+", " ", clean_text)  # Remove mentions
    clean_text = re.sub(r"[\"!#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]", " ", clean_text)  # Remove special characters
    clean_text = re.sub(r"[\x00-\x1F\x7F]", " ", clean_text)  # Remove non-printable characters
    clean_text = re.sub(r"\s+", " ", clean_text).strip()  # Remove extra spaces
    return clean_text

@app.route("/predict", methods=["POST"])
def predict():
    """Predicts the category of a given resume."""
    try:
        data = request.get_json()
        
        if "resume" not in data:
            return jsonify({"error": "Missing 'resume' key in request"}), 400
        
        resume_text = data["resume"]
        cleaned_resume = clean_resume(resume_text)
        
        # Transform text using TF-IDF
        input_feature = tfidf.transform([cleaned_resume])
        prediction_id = model.predict(input_feature)[0]

        category_name = category_mapping.get(prediction_id, "Unknown")

        return jsonify({"prediction": category_name})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
