import { useState, useRef } from "react";
import Head from "next/head";
import axios from "axios";

// API configuration
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
  const [formData, setFormData] = useState({});
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [imageLoading, setImageLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [showQuestions, setShowQuestions] = useState(false);
  const [autofilledFields, setAutofilledFields] = useState([]);
  const fileInputRef = useRef(null);

  const validateField = (id, value) => {
    const question = questions.find((q) => q.id === id);
    if (!question) return true;
    return question.validation(value);
  };

  const handleChange = (e) => {
    const { id, value } = e.target;
    setFormData({
      ...formData,
      [id]: value,
    });

    // Clear error for this field if valid
    if (validateField(id, value)) {
      const newErrors = { ...errors };
      delete newErrors[id];
      setErrors(newErrors);
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith("image/png")) {
      setErrors({
        ...errors,
        image: "Please upload a PNG image",
      });
      return;
    }

    setImageFile(file);

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(file);

    // Clear image error if exists
    const newErrors = { ...errors };
    delete newErrors.image;
    setErrors(newErrors);
  };

  const handleAutofill = async () => {
    if (!imageFile) {
      setErrors({
        ...errors,
        image: "Please upload an image first",
      });
      return;
    }

    setImageLoading(true);
    try {
      // Log the API URL being used
      console.log("Using API URL:", `${API_URL}/analyze-image`);

      // Create form data for file upload
      const formData = new FormData();
      formData.append("image", imageFile);

      // Call OpenAI O1 model API (replace with actual endpoint)
      const response = await axios.post(`${API_URL}/analyze-image`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      // Update form with the received data
      if (response.data) {
        // Keep track of which fields were filled by autofill
        const filledFields = [];

        // Map the API response fields to form field IDs
        const updatedData = {};

        // Process each field from the API response
        if (response.data.height) {
          updatedData.height = response.data.height;
          filledFields.push("height");
        }
        if (response.data.weight) {
          updatedData.weight = response.data.weight;
          filledFields.push("weight");
        }
        if (response.data.bloodPressure) {
          updatedData.bloodPressure = response.data.bloodPressure;
          filledFields.push("bloodPressure");
        }
        if (response.data.cholesterol) {
          updatedData.cholesterol = response.data.cholesterol;
          filledFields.push("cholesterol");
        }
        if (response.data.bmi) {
          updatedData.bmi = response.data.bmi;
          filledFields.push("bmi");
        }
        if (response.data.glucose) {
          updatedData.glucose = response.data.glucose;
          filledFields.push("glucose");
        }
        if (response.data.boneDensity) {
          updatedData.boneDensity = response.data.boneDensity;
          filledFields.push("boneDensity");
        }
        if (response.data.vision) {
          updatedData.vision = response.data.vision;
          filledFields.push("vision");
        }
        if (response.data.hearing) {
          updatedData.hearing = response.data.hearing;
          filledFields.push("hearing");
        }
        if (response.data.activity) {
          updatedData.activity = response.data.activity;
          filledFields.push("activity");
        }
        if (response.data.sleepDuration) {
          updatedData.sleep = response.data.sleepDuration;
          filledFields.push("sleep");
        }
        if (response.data.smokingStatus) {
          updatedData.smoking = response.data.smokingStatus;
          filledFields.push("smoking");
        }
        if (response.data.alcoholConsumption) {
          updatedData.alcohol = response.data.alcoholConsumption;
          filledFields.push("alcohol");
        }
        if (response.data.medicationCount) {
          updatedData.medication = response.data.medicationCount;
          filledFields.push("medication");
        }
        if (response.data.heartRate) {
          updatedData.heartRate = response.data.heartRate;
          filledFields.push("heartRate");
        }

        setFormData((prevData) => ({
          ...prevData,
          ...updatedData,
        }));

        // Update the list of autofilled fields
        setAutofilledFields(filledFields);

        // Expand the questions section to show the filled fields
        setShowQuestions(true);
      }
    } catch (error) {
      console.error("Error analyzing image:", error);
      // Log more detailed error information
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error("Response data:", error.response.data);
        console.error("Response status:", error.response.status);
        console.error("Response headers:", error.response.headers);
      } else if (error.request) {
        // The request was made but no response was received
        console.error("Request made but no response received:", error.request);
      } else {
        // Something happened in setting up the request that triggered an Error
        console.error("Error setting up request:", error.message);
      }

      setErrors({
        ...errors,
        image: `Error analyzing image: ${error.message}. Please try again.`,
      });
    } finally {
      setImageLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validate all fields
    const newErrors = {};
    let hasErrors = false;

    questions.forEach((question) => {
      const value = formData[question.id];
      if (!value || !question.validation(value)) {
        newErrors[question.id] = question.errorMessage;
        hasErrors = true;
      }
    });

    if (hasErrors) {
      setErrors(newErrors);
      return;
    }

    // Prepare data for API
    const apiData = {};
    questions.forEach((question) => {
      apiData[question.key] = formData[question.id];
    });

    // Submit data
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/predict`, {
        features: apiData,
      });
      setPrediction(response.data.predicted_age);
    } catch (error) {
      console.error("Error predicting age:", error);
      setErrors({
        ...errors,
        submit: "Error predicting age. Please try again.",
      });
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({});
    setErrors({});
    setPrediction(null);
    setImageFile(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = null;
    }
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
            <p className="text-gray-600 mb-6">
              Based on your health metrics, your predicted biological age is:
            </p>
            <div className="text-6xl font-bold text-indigo-600 mb-8">
              {Math.round(prediction * 10) / 10} years
            </div>
            <button
              onClick={resetForm}
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Start Over
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-gray-50 p-6">
      <Head>
        <title>Age Predictor</title>
        <meta
          name="description"
          content="Predict your biological age based on health metrics"
        />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="flex-grow flex flex-col items-center justify-center">
        <div className="max-w-3xl w-full bg-white rounded-xl shadow-lg p-8">
          <div className="text-center mb-8">
            <h1 className="text-5xl font-extrabold text-gray-1000">⏳</h1>
            <h1 className="text-3xl font-extrabold text-gray-1000">
              Biological Age Predictor
            </h1>
            <p className="mt-2 text-gray-600">
              Fill out the form below to predict your biological age based on
              health metrics
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Collapsible questions section */}
            <div className="border border-gray-200 rounded-md overflow-hidden">
              <button
                type="button"
                onClick={() => setShowQuestions(!showQuestions)}
                className="w-full flex justify-between items-center p-4 bg-gray-50 hover:bg-gray-100 focus:outline-none"
              >
                <span className="font-medium">Health Metrics Form Fields</span>
                <svg
                  className={`w-5 h-5 transition-transform ${
                    showQuestions ? "transform rotate-180" : ""
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M19 9l-7 7-7-7"
                  ></path>
                </svg>
              </button>

              {showQuestions && (
                <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-6">
                  {questions.map((question) => (
                    <div key={question.id} className="space-y-2">
                      <label
                        htmlFor={question.id}
                        className="block text-sm font-medium text-gray-700"
                      >
                        {question.label}
                      </label>

                      {question.type === "select" ? (
                        <select
                          id={question.id}
                          value={formData[question.id] || ""}
                          onChange={handleChange}
                          className={`block w-full rounded-md border ${
                            errors[question.id]
                              ? "border-red-300"
                              : formData[question.id]
                              ? "border-green-300"
                              : "border-gray-300"
                          } shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2`}
                        >
                          <option value="" disabled>
                            {question.placeholder}
                          </option>
                          {question.options.map((option) => (
                            <option key={option} value={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type={question.type}
                          id={question.id}
                          placeholder={question.placeholder}
                          step={question.step}
                          value={formData[question.id] || ""}
                          onChange={handleChange}
                          className={`block w-full rounded-md border ${
                            errors[question.id]
                              ? "border-red-300"
                              : formData[question.id]
                              ? "border-green-300"
                              : "border-gray-300"
                          } shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2`}
                        />
                      )}

                      {!formData[question.id] && (
                        <p className="mt-1 text-sm text-red-600 font-medium">
                          not updated
                        </p>
                      )}

                      {errors[question.id] && (
                        <p className="mt-1 text-sm text-red-600">
                          {errors[question.id]}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Summary of filled fields */}
            {Object.keys(formData).length > 0 && (
              <div className="p-4 border border-gray-200 rounded-md bg-gray-50">
                <h3 className="text-md font-medium text-gray-700 mb-2">
                  Entered Data:
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                  {questions
                    .filter((q) => formData[q.id])
                    .map((question) => (
                      <div key={question.id} className="flex justify-between">
                        <span className="font-medium">
                          {question.label.replace("?", "")}:
                        </span>
                        <span>{formData[question.id]}</span>
                      </div>
                    ))}
                </div>
              </div>
            )}

            <div className="border-t border-gray-200 pt-6">
              <div className="flex flex-col md:flex-row gap-4 items-start">
                <div className="w-full md:w-1/2">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Upload Image for Autofill
                  </label>
                  <div className="flex items-center space-x-2">
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/png"
                      onChange={handleImageChange}
                      className="block w-full text-sm text-gray-500
                        file:mr-4 file:py-2 file:px-4
                        file:rounded-md file:border-0
                        file:text-sm file:font-semibold
                        file:bg-indigo-50 file:text-indigo-700
                        hover:file:bg-indigo-100"
                    />
                    <button
                      type="button"
                      onClick={handleAutofill}
                      disabled={!imageFile || imageLoading}
                      className={`py-2 px-4 rounded-md text-sm font-medium text-white 
                        ${
                          !imageFile || imageLoading
                            ? "bg-gray-400"
                            : "bg-indigo-600 hover:bg-indigo-700"
                        } 
                        focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500`}
                    >
                      {imageLoading ? "Processing..." : "Autofill"}
                    </button>
                  </div>
                  {errors.image && (
                    <p className="mt-1 text-sm text-red-600">{errors.image}</p>
                  )}
                </div>

                {imagePreview && (
                  <div className="w-full md:w-1/2">
                    <p className="block text-sm font-medium text-gray-700 mb-2">
                      Image Preview
                    </p>
                    <img
                      src={imagePreview}
                      alt="Preview"
                      className="h-32 object-contain border border-gray-300 rounded-md"
                    />
                  </div>
                )}
              </div>
            </div>

            {errors.submit && (
              <div className="text-center text-red-600 text-sm mt-4">
                {errors.submit}
              </div>
            )}

            <div className="pt-4">
              <button
                type="submit"
                disabled={loading}
                className="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                {loading ? "Processing..." : "Predict My Age"}
              </button>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}
