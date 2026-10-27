# Smart Hospitality Management System

An AI-powered hospitality management project that combines machine learning, data analysis, and web APIs to support smarter hospitality operations and data-driven decision-making.

**GitHub Repository:** [Anwar6777/smart-hospitality-management-system](https://github.com/Anwar6777/smart-hospitality-management-system)

**Live Application:** [Frontend](https://smart-hospitality-management-system-c7fh.onrender.com/)

**Backend API:** [FastAPI Server](https://smart-hospitality-management-system.onrender.com/)

**Interactive API Documentation:** [Swagger UI](https://smart-hospitality-management-system.onrender.com/docs)

---

## Table of Contents

1. [Project Overview and Objectives](#1-project-overview-and-objectives)
2. [Features and Machine Learning Modules](#2-features-and-machine-learning-modules)
3. [Technology Stack](#3-technology-stack)
4. [Project Folder Structure](#4-project-folder-structure)
5. [Installation and Setup](#5-installation-and-setup)
6. [How to Run the Project](#6-how-to-run-the-project)
7. [API Documentation and Example Requests](#7-api-documentation-and-example-requests)
8. [Dataset and Model Details](#8-dataset-and-model-details)
9. [Deployment Information](#9-deployment-information)
10. [Future Enhancements](#10-future-enhancements)
11. [Author](#11-author)

---

## 1. Project Overview and Objectives

### About the Project

The **Smart Hospitality Management System (SHMS)** is a machine-learning-based project designed to help hospitality businesses make better operational decisions using data.

The system brings together multiple machine learning modules covering booking cancellation prediction, demand forecasting, food classification, and sentiment analysis. A FastAPI backend exposes prediction functionality through REST API endpoints, allowing a web interface or another client application to interact with the models.

The project demonstrates how machine learning can be integrated into a web-based application to transform hospitality-related data into useful predictions and insights.

### Objectives

The primary objectives are to:

- Predict the likelihood of booking cancellations and support better booking management.
- Forecast future demand to help hospitality businesses plan their operations.
- Classify food images using machine learning or deep learning techniques.
- Analyze customer sentiment to understand feedback and customer experiences.
- Integrate machine learning models with a backend API.
- Provide an accessible interface for interacting with the system.
- Demonstrate the practical application of Python, data analysis, machine learning, and web development.

### Problem Statement

Hospitality businesses must make decisions about bookings, demand, food services, and customer satisfaction. Making these decisions without sufficient data analysis can lead to inefficient planning and missed opportunities.

This project explores how machine learning models and API-based software architecture can assist with these challenges. Predictions are intended to support human decision-making rather than replace operational judgment.

### Proposed Solution

The system organizes different machine learning capabilities into separate modules. The backend accepts requests, processes inputs, invokes the appropriate prediction functionality, and returns results through an API response.

Conceptually, the workflow is:

```text
             User / Web Interface
                      |
                      v
               FastAPI Backend
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
      Booking      Demand      Customer
    Cancellation  Forecasting  Sentiment
          |           |           |
          +-----------+-----------+
                      |
                      v
             Prediction Results

          Food Classification
          operates as another
             ML module
```

---

## 2. Features and Machine Learning Modules

### 2.1 Booking Cancellation Prediction

The cancellation module focuses on predicting booking cancellation outcomes using historical booking information and a trained machine learning model.

**Purpose:**
- Identify bookings that may be at risk of cancellation.
- Support booking and capacity planning.
- Help hospitality teams investigate cancellation patterns.
- Provide predictions that can inform operational decisions.

**Typical workflow:**

1. Receive booking-related input.
2. Validate and preprocess the input.
3. Apply the trained cancellation model.
4. Return the prediction through the API.

The exact input fields and response schema are defined by the deployed API.

### 2.2 Demand Forecasting

The demand-forecasting module estimates future demand based on historical data.

The project has referenced the following model options:

- `random_forest`
- `lstm`
- `lstm_improved`

Random Forest is a tree-based ensemble learning approach. LSTM (Long Short-Term Memory) is a recurrent neural network architecture commonly used for sequential data and time-series forecasting.

**Purpose:**
- Estimate future demand over a selected forecasting horizon.
- Support staffing and resource planning.
- Help hospitality businesses prepare for changes in demand.
- Compare different forecasting approaches.

The API includes a forecasting endpoint that accepts a forecasting horizon and model selection.

### 2.3 Food Classification

The food-classification module focuses on identifying food categories from image inputs using a trained classification model.

**Purpose:**
- Process food images.
- Predict a food category supported by the trained model.
- Demonstrate image preprocessing and model inference.
- Provide a foundation for future food-service automation.

The supported food categories, image requirements, model architecture, and prediction response should be confirmed from the model files and API implementation.

### 2.4 Sentiment Analysis

The sentiment-analysis module is designed to analyze text and produce a sentiment prediction.

**Purpose:**
- Analyze customer feedback or review text.
- Identify sentiment patterns in hospitality-related feedback.
- Support customer-experience analysis.
- Provide information that may help identify areas for service improvement.

The repository includes a sentiment-analysis module. The exact labels, supported languages, preprocessing pipeline, and model architecture depend on the implementation.

### 2.5 Backend API Integration

FastAPI provides an interface between the machine learning modules and the web application.

Its responsibilities include:

- Receiving HTTP requests.
- Validating structured request data.
- Invoking prediction services.
- Returning structured responses.
- Exposing interactive API documentation.

### 2.6 Modular Architecture

Keeping the ML modules separate makes the project easier to maintain. Each module can be developed, tested, and improved independently, provided its interface remains compatible with the backend.

---

## 3. Technology Stack

The project uses Python-based machine learning and a web API architecture.

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | REST API backend |
| Uvicorn | ASGI server for running FastAPI |
| NumPy | Numerical operations |
| Pandas | Data manipulation and preprocessing |
| Scikit-learn | Machine learning utilities and models |
| Joblib | Saving and loading compatible trained models |
| TensorFlow / Keras | Deep learning functionality where used by a module |
| SQLAlchemy | Database interaction through an ORM |
| PostgreSQL driver (`psycopg`) | PostgreSQL connectivity |
| Pillow | Image processing support |
| HTML, CSS, JavaScript or the configured frontend framework | User-interface functionality, according to the frontend implementation |
| Render | Application deployment |

**Note:** Not every library is necessarily used by every module. The actual dependency versions should be taken from the repository's dependency files and the deployment configuration.

---

## 4. Project Folder Structure

The repository's visible top-level structure includes the following directories:

```text
smart-hospitality-management-system/
│
├── cancellation/
│   └── Booking cancellation prediction module
│
├── demand-forecasting/
│   └── Demand forecasting module
│
├── food-classification/
│   └── Food image classification module
│
├── sentiment-analysis/
│   └── Customer sentiment analysis module
│
└── .gitignore
```

The deployed backend also uses a FastAPI application with an entry point referenced as `app.main:app`.

A conceptual view of the backend's responsibilities is:

```text
Backend Application
│
├── app.main
│   └── FastAPI application entry point
│
├── API routes
│   └── HTTP endpoint definitions
│
├── Configuration
│   └── Application settings
│
├── Services
│   └── Prediction and business logic
│
├── Models
│   └── Trained model files
│
└── Data / Preprocessing
    └── Input preparation and transformations
```

The second diagram is a conceptual organization, not a verified listing of every file or directory. Use the actual source tree when documenting individual filenames.

---

## 5. Installation and Setup

### Prerequisites

Install the following before running the project locally:

- Python compatible with the project's pinned dependencies.
- Git.
- `pip`, Python's package installer.
- A code editor such as Visual Studio Code.
- Access to the trained model files and datasets required by the modules.

Deep learning dependencies can require additional memory and compatible package versions.

### Step 1: Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/Anwar6777/smart-hospitality-management-system.git
cd smart-hospitality-management-system
```

### Step 2: Create a Virtual Environment

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

If a `requirements.txt` file is present in the backend's working directory, install its dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the dependency file is located elsewhere, navigate to that directory or provide its correct path.

Do not install arbitrary latest versions of machine learning packages when the project depends on specific versions. Check the existing requirements and model compatibility first.

### Step 4: Configure Environment Variables

If the application uses environment variables, create a local `.env` file in the location expected by the settings implementation.

Configure the actual variables required by the project, such as database connection details, CORS origins, or other service credentials, where applicable.

For example, a database-backed deployment may require a connection string:

```text
DATABASE_URL=<your-database-connection-string>
```

This is an illustrative example, not a guarantee that the application uses this exact variable name. Confirm the required settings in the project's configuration code.

**Security reminders:**

- Never commit passwords, API keys, database credentials, or private tokens.
- Keep `.env` files out of version control.
- Use the deployment platform's environment-variable settings for production secrets.
- Do not place secrets in frontend code.

### Step 5: Check Model Files

Ensure that the trained model files required by the selected module are available at the paths expected by the application.

If a model is stored outside Git or is downloaded separately, follow the project's existing model-loading configuration. Do not rename model files without updating the corresponding paths in the code.

---

## 6. How to Run the Project

### Run the FastAPI Backend Locally

From the directory containing the `app` package, run:

```bash
uvicorn app.main:app --reload
```

The development server should be available at:

- **API base URL:** `http://127.0.0.1:8000`
- **Swagger documentation:** `http://127.0.0.1:8000/docs`
- **ReDoc documentation:** `http://127.0.0.1:8000/redoc`

These URLs assume the server starts on Uvicorn's default local host and port.

If the application cannot be imported, verify that your terminal is in the correct directory, the dependencies are installed, and the `app.main:app` import path matches the source code.

### Test the API

1. Start the backend.
2. Open `http://127.0.0.1:8000/docs`.
3. Select an endpoint.
4. Expand the endpoint and click **Try it out**.
5. Enter the request parameters or JSON body specified by the API schema.
6. Click **Execute**.
7. Review the HTTP status code and response body.

### Run with a Deployment Port

For a deployment environment that provides a `PORT` variable, the server can be started with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

On Windows PowerShell, use the syntax appropriate to that shell if running locally:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port $env:PORT
```

---

## 7. API Documentation and Example Requests

### Interactive Documentation

The deployed FastAPI service provides Swagger UI:

[Open the API documentation](https://smart-hospitality-management-system.onrender.com/docs)

Swagger is the recommended place to inspect the current request schemas, required fields, response structures, and error responses before sending prediction requests.

### API Endpoint Overview

| Module | Endpoint | Method | Purpose |
|---|---|---|---|
| Booking cancellation | `/api/booking/predict` | POST | Submit booking information for cancellation prediction |
| Demand forecasting | `/api/demand/forecast` | GET | Request a demand forecast |
| Sentiment analysis | `/api/sentiment/predict` | POST | Submit text for sentiment analysis |
| Food classification | Check `/docs` | As defined by the API | Identify the implemented image-classification route |

The first three endpoint paths are based on the project's known backend routes. Confirm their current request schemas in Swagger. The food-classification route is intentionally not guessed.

### Example 1: Demand Forecasting

Request a seven-step forecast using the Random Forest option:

```bash
curl -X GET "https://smart-hospitality-management-system.onrender.com/api/demand/forecast?horizon=7&model=random_forest" \
  -H "Accept: application/json"
```

Other model options referenced by the project include:

```text
lstm
lstm_improved
```

For example, the query parameters can be changed to:

```text
/api/demand/forecast?horizon=7&model=lstm
```

The meaning of each forecast step, the output fields, and any limitations on the horizon should be checked in the API documentation.

### Example 2: Booking Cancellation Prediction

Endpoint:

```text
POST /api/booking/predict
```

Use the following steps:

1. Open the deployed Swagger documentation.
2. Locate `/api/booking/predict`.
3. Click **Try it out**.
4. Use the example JSON or required schema displayed by the API.
5. Execute the request and review the prediction.

The exact JSON body is not included here because the required field names and data types must match the implemented Pydantic request schema.

### Example 3: Sentiment Analysis

Endpoint:

```text
POST /api/sentiment/predict
```

The request should use the JSON structure specified by the endpoint's Swagger schema. Submit the text or other required fields, then inspect the returned sentiment result.

Do not assume the response uses particular sentiment labels unless those labels are confirmed by the implementation.

### Example 4: Python Client

The following example demonstrates how to call the demand-forecasting endpoint from Python:

```python
import requests

url = (
    "https://smart-hospitality-management-system.onrender.com"
    "/api/demand/forecast"
)

params = {
    "horizon": 7,
    "model": "random_forest",
}

try:
    response = requests.get(
        url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()
    result = response.json()

    print("Forecast response:")
    print(result)

except requests.RequestException as error:
    print("API request failed:", error)

except ValueError:
    print("The server did not return valid JSON.")
```

Install the HTTP client if needed:

```bash
pip install requests
```

### Common API Errors

| Problem | Possible cause | Suggested check |
|---|---|---|
| `404 Not Found` | Incorrect route or deployment path | Verify the route in Swagger |
| `422 Unprocessable Entity` | Missing or invalid request fields | Follow the endpoint schema |
| `500 Internal Server Error` | Model-loading or prediction failure | Review backend logs |
| CORS error | Frontend origin is not permitted | Check backend CORS configuration |
| Unexpected end of JSON input | Empty or non-JSON response | Inspect HTTP status and response text before parsing JSON |
| Connection timeout | Server startup delay or overloaded service | Check service health and deployment logs |

---

## 8. Dataset and Model Details

### Dataset Overview

The system uses data appropriate to its individual machine learning tasks. Each module may require a different dataset structure and preprocessing workflow.

| Module | Data type | Intended task |
|---|---|---|
| Booking cancellation | Booking-related tabular data | Predict cancellation outcomes |
| Demand forecasting | Historical demand or time-series data | Estimate future demand |
| Food classification | Food images and category labels | Classify images |
| Sentiment analysis | Text data and sentiment labels, where applicable | Analyze sentiment |

The exact dataset names, original sources, licensing conditions, sample counts, feature columns, train/test splits, and class distributions should be documented from the dataset files and training scripts. They are not inferred here.

### Data Preprocessing

Depending on the module, preprocessing may include:

- Handling missing or invalid values.
- Encoding categorical features.
- Scaling numerical features when required.
- Formatting chronological observations for forecasting.
- Resizing or normalizing images.
- Cleaning and tokenizing text.
- Applying the same transformations during training and inference.

The preprocessing pipeline used at prediction time must remain compatible with the pipeline used when training the model.

### Model Overview

**Booking cancellation:** A trained classification model estimates the cancellation outcome from booking features.

**Demand forecasting:** The module provides model options identified as Random Forest, LSTM, and improved LSTM. Their suitability depends on data quality, temporal validation, and forecast accuracy.

**Food classification:** A trained image-classification model predicts among its supported food classes.

**Sentiment analysis:** A trained text-analysis model predicts sentiment based on its implemented labels and preprocessing pipeline.

### Model Evaluation

For classification tasks, useful evaluation measures include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For forecasting or numerical prediction tasks, useful measures include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE), where appropriate

Metrics should be reported only after evaluating the actual trained model on a suitable validation or test dataset. No unverified accuracy score is claimed in this README.

### Model Storage and Compatibility

Models may be saved in serialized formats such as Joblib or in TensorFlow/Keras formats, depending on the module.

For reliable inference:

1. Keep the model file and its expected preprocessing configuration together.
2. Use compatible library versions.
3. Validate the input shape and data types.
4. Handle missing or malformed inputs.
5. Monitor model-loading errors and memory usage.
6. Re-evaluate models when training data or production conditions change.

Large deep learning models can require substantial memory. On resource-limited hosting plans, loading multiple models into one process may cause startup failures or out-of-memory errors. Load models selectively or use a hosting environment that meets the actual resource requirements.

---

## 9. Deployment Information

The project has been deployed using Render.

### Deployment Links

| Component | URL |
|---|---|
| Frontend | [smart-hospitality-management-system-c7fh.onrender.com](https://smart-hospitality-management-system-c7fh.onrender.com/) |
| Backend API | [smart-hospitality-management-system.onrender.com](https://smart-hospitality-management-system.onrender.com/) |
| Swagger UI | [Backend `/docs`](https://smart-hospitality-management-system.onrender.com/docs) |

Availability may vary depending on deployment status and hosting resources.

### Backend Deployment

A typical Render configuration for the FastAPI backend uses:

**Build command** — adapt to the actual location of the dependency file:

```bash
pip install -r requirements.txt
```

**Start command:**

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

The build and start commands must match the backend's actual root directory and dependency-file location.

### Frontend-to-Backend Integration

The frontend must send API requests to the deployed backend URL or the configured local backend URL.

For production deployment:

- Configure the correct API base URL in the frontend.
- Ensure the backend permits requests from the frontend origin.
- Avoid accidentally using `localhost` in production API calls.
- Ensure request URLs do not contain extra quotation marks or malformed path segments.
- Check that API responses are valid JSON before parsing them.
- Verify that the frontend handles non-success HTTP responses.

### Database and Environment Configuration

If the deployed application uses a database, configure its connection settings through the hosting platform's environment variables.

Confirm that the database is reachable from the deployed service and that credentials are not exposed in source control.

### Deployment Troubleshooting

**CORS errors:** Confirm that the frontend's exact origin is allowed by the backend.

**Model-loading failures:** Verify model paths, dependency versions, and access to any separately stored model files.

**Out-of-memory errors:** Review the size of serialized models, TensorFlow imports, concurrent requests, and the hosting plan's memory limit.

**Frontend receives HTML instead of JSON:** Check whether the request reached the correct backend API route or was redirected to a frontend page.

**Cold starts or slow requests:** Monitor service startup time and inference latency, especially for deep learning models.

---

## 10. Future Enhancements

The following are proposed improvements, not claims that these features already exist.

### 10.1 Improve Prediction Accuracy

- Perform systematic hyperparameter tuning.
- Compare multiple baseline and advanced models.
- Use appropriate validation strategies.
- Monitor model drift and changes in prediction quality.

### 10.2 Enhance Demand Forecasting

- Add confidence intervals where supported.
- Compare forecasts across different time horizons.
- Evaluate seasonal patterns and special events.
- Provide visual comparisons between actual and predicted demand.

### 10.3 Improve Cancellation Management

- Add risk categories based on calibrated model outputs.
- Provide explanations for important prediction factors.
- Track model performance as new booking outcomes become available.

### 10.4 Expand Sentiment Analysis

- Improve handling of short reviews and informal text.
- Evaluate performance across different sentiment classes.
- Add multilingual support if suitable training data is available.
- Summarize recurring customer feedback themes.

### 10.5 Extend Food Classification

- Expand supported categories using appropriately labeled data.
- Add image-quality validation.
- Report prediction confidence where supported by the model.
- Optimize inference latency and memory use.

### 10.6 Improve the Web Application

- Improve responsive design and accessibility.
- Add clearer validation and error messages.
- Provide interactive charts and downloadable reports.
- Improve loading states and display of model predictions.

### 10.7 Strengthen Reliability and Security

- Add automated unit and integration tests.
- Validate and limit API inputs.
- Add structured logging and monitoring.
- Use secure configuration management.
- Document API schemas and error responses.
- Add health checks and deployment verification.

### 10.8 Improve Model Deployment

- Reduce unnecessary model-loading overhead.
- Cache models safely when appropriate.
- Separate resource-intensive inference workloads when needed.
- Add model versioning and reproducible training workflows.

---

## 11. Author

**Anwar6777**

GitHub: [@Anwar6777](https://github.com/Anwar6777)

Project repository: [Smart Hospitality Management System](https://github.com/Anwar6777/smart-hospitality-management-system)

This project demonstrates the integration of machine learning, data preprocessing, prediction services, and web API development in a hospitality-focused application.

---

## License

A license was not verified from the repository information used to prepare this README. Add a `LICENSE` file and update this section when the project's license has been selected.

## Disclaimer

Predictions generated by machine learning models are estimates based on their training data and implemented assumptions. They should be evaluated for accuracy, fairness, and suitability before being used for consequential business decisions.

---

**If you find this project useful, you can explore the source code, report issues, and suggest improvements through the GitHub repository.**
