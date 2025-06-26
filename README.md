Bati-credit-risk-model 

Credit risk model for Bati Bank Credit Risk Modeling for Bati Bank This project is part of Bati Bank’s strategic initiative to enable Buy Now, Pay Later (BNPL) services in partnership with an eCommerce platform. The goal is to build a Credit Scoring Model that assigns a risk probability, credit score, and loan recommendation to new applicants based on transaction behavior. 

📌 Project Overview Organization: Bati Bank 

Objective: Build a credit scoring model to identify high-risk and low-risk users using transaction data 

Business Value: Improve loan approval decisions, manage credit risk, and comply with Basel II standards 

📁 Project Structure plaintext Copy Edit credit-risk-model/ ├── .github/workflows/ci.yml # GitHub Actions for CI/CD ├── data/ # Add this folder to .gitignore │ ├── raw/ # Raw input data │ └── processed/ # Cleaned and transformed data ├── notebooks/ │ └── 1.0-eda.ipynb # Exploratory data analysis ├── src/ │ ├── init.py │ ├── data_processing.py # Feature engineering functions │ ├── train.py # Model training script │ ├── predict.py # Inference/prediction script │ └── api/ │ ├── main.py # FastAPI application │ └── pydantic_models.py # Data validation with Pydantic ├── tests/ │ └── test_data_processing.py # Unit tests ├── Dockerfile ├── docker-compose.yml ├── requirements.txt ├── .gitignore └── README.md 📊 Credit Scoring Business Understanding 

Basel II Accord and the Need for an Interpretable Model The Basel II Capital Accord requires financial institutions to adopt a risk-sensitive framework that links capital requirements to the underlying credit risk of lending activities. Under this framework, banks must quantify credit risk using either standardized or internal ratings-based approaches. This increases the demand for interpretable, auditable, and well-documented models, especially in regulatory environments. For Bati Bank, partnering with an eCommerce platform to offer "Buy Now, Pay Later" (BNPL) services, this means our credit scoring model must not only be accurate but also transparent. Stakeholders (including regulators) need to understand how decisions are made and ensure the model does not introduce bias or operational risk. 

The Role and Risk of Using a Proxy Variable Since the dataset lacks a direct label indicating loan default, we must engineer a proxy variable to classify customers as "high-risk" or "low-risk." In this project, behavioral metrics such as fraud flags, RFM patterns, or late payment behavior can serve as proxies. While necessary, this approach introduces business risks: 

Mislabeling: Customers might be incorrectly categorized, leading to either over-rejection (losing good customers) or over-lending (exposing the bank to unnecessary risk). 

Bias propagation: If the proxy variable embeds historical biases (e.g., based on region or channel), the model could reinforce them. 

Model drift: As customer behavior evolves, the proxy may become outdated, degrading model performance over time. Thus, continuous monitoring and periodic re-evaluation of the proxy are essential. 

Model Choice: Interpretability vs. Predictive Power In regulated financial services, there's a critical trade-off between model interpretability and predictive power: 

Simple, Interpretable Models (e.g., Logistic Regression with Weight of Evidence encoding): 

✅ Pros: 

Easy to explain to regulators and business teams 

Transparent decision boundaries 

Fast training and deployment 

❌ Cons: 

Limited ability to capture complex patterns 

Lower performance on high-dimensional or nonlinear data 

Complex, High-Performance Models (e.g., Gradient Boosting, XGBoost, LightGBM): 

✅ Pros: 

Higher accuracy and better handling of feature interactions 

Robust to missing values and outliers 

❌ Cons: 

Harder to interpret 

Risk of overfitting and lack of transparency 

Regulatory scrutiny may delay approval 

To align with Basel II and ensure trustworthy lending decisions, we may start with an interpretable model for deployment and monitor performance, then explore complex models under strict governance using explainability tools (like SHAP). 

📚 Learning Objectives Skills Feature Engineering 

ML Modeling and Hyperparameter Tuning 

CI/CD & FastAPI Deployment 

MLOps with MLflow and CML 

Python Unit Testing & Logging 

Knowledge Credit Risk & Basel II 

Proxy Variable Design 

Model Interpretability vs. Performance 

Regulatory-Compliant ML Systems 

 
