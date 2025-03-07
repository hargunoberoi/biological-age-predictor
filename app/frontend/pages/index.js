import { useState } from "react";
import Head from "next/head";
import axios from "axios";

// API configuration
const API_URL = "http://localhost:8000";

// Define the questions and their properties
const questions = [
  {
    id: "height",
    label: "What is your height in centimeters?",
    type: "number",
    placeholder: "Enter height in cm (e.g. 175)",
    key: "Height (cm)",
    validation: (value) => value > 0 && value < 250,
    errorMessage: "Please enter a valid height between 1 and 250 cm",
  },
  {
    id: "weight",
    label: "What is your weight in kilograms?",
    type: "number",
    placeholder: "Enter weight in kg (e.g. 70)",
    key: "Weight (kg)",
    validation: (value) => value > 0 && value < 300,
    errorMessage: "Please enter a valid weight between 1 and 300 kg",
  },
  {
    id: "bloodPressure",
    label: "What is your blood pressure (systolic/diastolic)?",
    type: "text",
    placeholder: "Enter as systolic/diastolic (e.g. 120/80)",
    key: "Blood Pressure (s/d)",
    validation: (value) => {
      const pattern = /^\d{2,3}\/\d{2,3}$/;
      return pattern.test(value);
    },
    errorMessage:
      "Please enter a valid blood pressure in format systolic/diastolic (e.g. 120/80)",
  },
  {
    id: "cholesterol",
    label: "What is your cholesterol level in mg/dL?",
    type: "number",
    placeholder: "Enter cholesterol in mg/dL (e.g. 180)",
    key: "Cholesterol Level (mg/dL)",
    validation: (value) => value >= 0 && value < 500,
    errorMessage:
      "Please enter a valid cholesterol level between 0 and 500 mg/dL",
  },
  {
    id: "bmi",
    label: "What is your BMI?",
    type: "number",
    placeholder: "Enter BMI (e.g. 22.5)",
    step: "0.1",
    key: "BMI",
    validation: (value) => value > 0 && value < 50,
    errorMessage: "Please enter a valid BMI between 1 and 50",
  },
  {
    id: "glucose",
    label: "What is your blood glucose level in mg/dL?",
    type: "number",
    placeholder: "Enter blood glucose in mg/dL (e.g. 90)",
    key: "Blood Glucose Level (mg/dL)",
    validation: (value) => value >= 0 && value < 500,
    errorMessage: "Please enter a valid glucose level between 0 and 500 mg/dL",
  },
  {
    id: "boneDensity",
    label: "What is your bone density in g/cm²?",
    type: "number",
    placeholder: "Enter bone density in g/cm² (e.g. 0.6)",
    step: "0.01",
    key: "Bone Density (g/cm²)",
    validation: (value) => value > 0 && value < 2,
    errorMessage: "Please enter a valid bone density between 0 and 2 g/cm²",
  },
  {
    id: "vision",
    label: "What is your vision sharpness (0.1-1.0)?",
    type: "number",
    placeholder: "Enter vision sharpness (e.g. 0.8)",
    step: "0.1",
    key: "Vision Sharpness",
    validation: (value) => value >= 0 && value <= 1,
    errorMessage: "Please enter a valid vision sharpness between 0 and 1",
  },
  {
    id: "hearing",
    label: "What is your hearing ability in dB?",
    type: "number",
    placeholder: "Enter hearing ability in dB (e.g. 20)",
    key: "Hearing Ability (dB)",
    validation: (value) => value >= 0 && value < 150,
    errorMessage: "Please enter a valid hearing ability between 0 and 150 dB",
  },
  {
    id: "activity",
    label: "What is your physical activity level?",
    type: "select",
    options: ["Low", "Moderate", "High"],
    placeholder: "Select activity level",
    key: "Physical Activity Level",
    validation: (value) => ["Low", "Moderate", "High"].includes(value),
    errorMessage: "Please select a valid activity level",
  },
  {
    id: "disease",
    label: "Do you have any chronic diseases?",
    type: "select",
    options: ["None", "Hypertension", "Diabetes", "Heart Disease"],
    placeholder: "Select chronic disease status",
    key: "Chronic Diseases",
    validation: (value) =>
      ["None", "Hypertension", "Diabetes", "Heart Disease"].includes(value),
    errorMessage: "Please select a valid option",
  },
  {
    id: "medication",
    label: "What is your medication use?",
    type: "select",
    options: ["None", "Occasional", "Regular"],
    placeholder: "Select medication use",
    key: "Medication Use",
    validation: (value) => ["None", "Occasional", "Regular"].includes(value),
    errorMessage: "Please select a valid option",
  },
  {
    id: "cognitive",
    label: "Rate your cognitive function (0-100):",
    type: "number",
    placeholder: "Enter cognitive function score (e.g. 85)",
    key: "Cognitive Function",
    validation: (value) => value >= 0 && value <= 100,
    errorMessage:
      "Please enter a valid cognitive function score between 0 and 100",
  },
  {
    id: "education",
    label: "What is your education level?",
    type: "select",
    options: ["None", "High School", "Undergraduate", "Postgraduate"],
    placeholder: "Select education level",
    key: "Education Level",
    validation: (value) =>
      ["None", "High School", "Undergraduate", "Postgraduate"].includes(value),
    errorMessage: "Please select a valid education level",
  },
];

export default function Home() {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [currentAnswer, setCurrentAnswer] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);

  const currentQuestion = questions[currentQuestionIndex];

  // Add a function to handle Enter key press
  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault(); // Prevent default form submission
      handleNext();
    }
  };

  const handleNext = () => {
    // Validate the current answer
    if (!currentQuestion.validation(currentAnswer)) {
      setError(currentQuestion.errorMessage);
      return;
    }

    // Save the answer
    const newAnswers = {
      ...answers,
      [currentQuestion.key]: currentAnswer,
    };
    setAnswers(newAnswers);

    // Move to the next question or predict if it's the last question
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
      setCurrentAnswer("");
      setError("");
    } else {
      // Submit all answers for prediction
      predictAge(newAnswers);
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
      setCurrentAnswer(answers[questions[currentQuestionIndex - 1].key] || "");
      setError("");
    }
  };

  const handleChange = (e) => {
    const value = e.target.value;
    setCurrentAnswer(value);
    setError("");
  };

  const predictAge = async (allAnswers) => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/predict`, {
        features: allAnswers,
      });
      setPrediction(response.data.predicted_age);
    } catch (error) {
      console.error("Error predicting age:", error);
      setError("Error predicting age. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setCurrentQuestionIndex(0);
    setAnswers({});
    setCurrentAnswer("");
    setError("");
    setPrediction(null);
  };

  // Show the prediction result screen if we have a prediction
  if (prediction !== null) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 p-6">
        <Head>
          <title>Age Predictor</title>
          <meta
            name="description"
            content="Predict your biological age based on health metrics"
          />
          <link rel="icon" href="/favicon.ico" />
        </Head>

        <div className="max-w-md w-full space-y-8 bg-white p-8 rounded-xl shadow-lg">
          <div className="text-center">
            <h2 className="text-3xl font-extrabold text-gray-900 mb-2">
              Your Predicted Age
            </h2>
            <div className="mt-8 mb-8">
              <div className="text-6xl font-bold text-primary">
                {Math.round(prediction)} years
              </div>
              <p className="mt-4 text-gray-600">
                This is based on the health metrics you provided.
              </p>
            </div>
            <button
              onClick={resetForm}
              className="mt-6 w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Start Over
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-6">
      <Head>
        <title>Age Predictor</title>
        <meta
          name="description"
          content="Predict your biological age based on health metrics"
        />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="max-w-md w-full space-y-8 bg-white p-8 rounded-xl shadow-lg">
        <div>
          <div className="text-center mb-4">
            <span className="text-6xl">⌛</span>
          </div>
          <h2 className="text-2xl font-extrabold text-gray-900 text-center mb-2">
            Age Predictor
          </h2>
          <p className="text-sm text-gray-600 text-center">
            Question {currentQuestionIndex + 1} of {questions.length}
          </p>
          <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
            <div
              className="bg-primary h-2.5 rounded-full"
              style={{
                width: `${
                  ((currentQuestionIndex + 1) / questions.length) * 100
                }%`,
              }}
            ></div>
          </div>
        </div>

        <div className="mt-8">
          <div className="mb-6">
            <label className="block text-gray-700 text-lg font-medium mb-2">
              {currentQuestion.label}
            </label>

            {currentQuestion.type === "select" ? (
              <select
                value={currentAnswer}
                onChange={handleChange}
                className="mt-1 block w-full py-3 px-4 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              >
                <option value="" disabled>
                  {currentQuestion.placeholder}
                </option>
                {currentQuestion.options.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type={currentQuestion.type}
                value={currentAnswer}
                onChange={handleChange}
                onKeyDown={handleKeyDown}
                placeholder={currentQuestion.placeholder}
                step={currentQuestion.step}
                className="mt-1 block w-full py-3 px-4 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              />
            )}

            {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
          </div>

          <div className="flex justify-between mt-8">
            <button
              onClick={handlePrevious}
              disabled={currentQuestionIndex === 0}
              className={`py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium ${
                currentQuestionIndex === 0
                  ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                  : "bg-white text-gray-700 hover:bg-gray-50"
              }`}
            >
              Previous
            </button>

            <button
              onClick={handleNext}
              className="py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              {currentQuestionIndex === questions.length - 1
                ? "Predict Age"
                : "Next"}
            </button>
          </div>
        </div>

        {loading && (
          <div className="fixed inset-0 bg-gray-600 bg-opacity-75 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg shadow-xl max-w-sm w-full flex flex-col items-center">
              <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-primary mb-4"></div>
              <p className="text-gray-700">Predicting your age...</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
