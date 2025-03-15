import { useState } from "react";
import axios from "axios";
import * as pdfjsLib from "pdfjs-dist/build/pdf";
import "pdfjs-dist/build/pdf.worker.entry"; // Required for pdf.js
import { motion } from "framer-motion";
import { Upload, FileCheck } from "lucide-react";

const ResumeUpload = () => {
  const [file, setFile] = useState(null);
  const [resumeText, setResumeText] = useState("");
  const [prediction, setPrediction] = useState("");

  // Handle file selection
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(selectedFile);
      extractTextFromPDF(selectedFile);
    } else {
      alert("Please upload a valid PDF file.");
    }
  };

  // Extract text from PDF using pdf.js
  const extractTextFromPDF = async (file) => {
    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = async () => {
      const typedArray = new Uint8Array(reader.result);
      const pdf = await pdfjsLib.getDocument(typedArray).promise;
      let text = "";

      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        text += content.items.map((item) => item.str).join(" ") + "\n";
      }

      setResumeText(text);
    };
  };

  // Send extracted text to Flask backend
  const handleSubmit = async () => {
    if (!resumeText) {
      alert("No resume text extracted. Please upload a valid PDF.");
      return;
    }

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", {
        resume: resumeText,
      });

      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error sending data:", error);
      alert("Error sending resume. Check the console for details.");
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-blue-500 to-purple-600 p-6"
    >
      <div className="w-full max-w-lg p-6 bg-white/20 backdrop-blur-lg rounded-2xl shadow-xl border border-white/10">
        <h2 className="text-2xl font-bold text-white text-center mb-4">
          Upload Your Resume
        </h2>

        <label
          className="w-full flex flex-col items-center justify-center border-2 border-dashed border-gray-300 bg-white/20 rounded-lg p-6 cursor-pointer hover:border-blue-400 transition-all"
        >
          <input
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            className="hidden"
          />
          {file ? (
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.3 }}
              className="flex items-center gap-2 text-green-500"
            >
              <FileCheck size={24} />
              <span className="text-white">{file.name}</span>
            </motion.div>
          ) : (
            <div className="flex flex-col items-center text-gray-300">
              <Upload size={32} />
              <span className="mt-2 text-sm">Drag & drop or click to upload</span>
            </div>
          )}
        </label>

        <button
          onClick={handleSubmit}
          className="w-full mt-4 bg-blue-500 text-white py-2 rounded-lg hover:bg-blue-600 transition-all"
        >
          Submit Resume
        </button>

        {prediction && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="mt-4 p-4 bg-white/20 rounded-lg text-center text-white font-medium"
          >
            <strong>Prediction:</strong> {prediction}
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

export default ResumeUpload;
