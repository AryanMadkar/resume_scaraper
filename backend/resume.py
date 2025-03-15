import pickle 
import re
from senti.backend.dummy import resume
model = pickle.load(open(r"F:\dektop241205\healthappv2\senti\resume.pickle","rb"))
tfidf = pickle.load(open(r"F:\dektop241205\healthappv2\senti\tfidf.pickle","rb"))



def cleanResume(txt):
    cleanText = re.sub(r"http\S+", " ", txt)  # Remove URLs
    cleanText = re.sub(r"\b(RT|CC)\b", " ", cleanText)  # Remove retweets (RT) and copy-comments (CC)
    cleanText = re.sub(r"#\S+", " ", cleanText)  # Remove hashtags
    cleanText = re.sub(r"@\S+", " ", cleanText)  # Remove mentions
    cleanText = re.sub(r"[\"!#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]", " ", cleanText)  # Remove special characters
    cleanText = re.sub(r"[\x00-\x1F\x7F]", " ", cleanText)  # Remove non-printable characters
    cleanText = re.sub(r"\s+", " ", cleanText).strip()  # Remove extra spaces
    return cleanText

cleanedresume = cleanResume(resume)
inputfeture = tfidf.transform([cleanResume(resume)])  # ✅ Correct
prediction_id = model.predict(inputfeture)[0]

catageory_mapping = {
    15:"JAVA DEV",
    23:"TESTING",
    8:"DEVOPS ENGINEER",
    20:"PYTHON DEVELOPER",
    24:"WEB DESIGNER",
    12:"HR",
    12:"HADOOP",
    3:"BLOCKCHAIN",
    10:"ETL DEVELOPER",
    18:"OPERATIONS MANAGER",
    6:"DATA SCIENCE",
    22:"SALES",
    16:"MECHANICAL ENGINEER",
    1:"ARTS",
    7:"DATABASE",
    11:"ELECTRICAL ENGINEER",
    14:"HEALTH AND FITNESS",
    19:"PMO",
    4:"BUSINESS ANALYST",
    9:"DOTNET DEVELOPER",
    2:"AUTOMATION TESTING",
    17:"NETWORK ECURITY ENGINEER",
    21:"SAP DEVELOPER",
    5:"CIVIL ENGINEER",
    0:"ADVOCATE"
}

categeory_name = catageory_mapping.get(prediction_id,"unknown")
print("prediction: ",categeory_name)