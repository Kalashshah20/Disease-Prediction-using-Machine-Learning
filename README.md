# **🩺 Disease Prediction using Machine Learning**
───────────────────────────────────────────────────────────────────────────────
*Developed an end-to-end disease prediction pipeline to assist early diagnosis and clinical decision-making, combining robust preprocessing, exploratory analysis, and model benchmarking.*  

## **🔎 Project Overview**
───────────────────────────────────────────────────────────────────────────────
This project builds a machine learning-based disease prediction system using patient symptoms and vitals. The pipeline includes data cleaning, feature engineering, exploratory data analysis, model training (Decision Trees, SVM, Neural Networks), evaluation with cross-validation, and a prototype inference interface for symptom-based predictions.

## **✅ Key Contributions**
───────────────────────────────────────────────────────────────────────────────
- Implemented an **end-to-end pipeline** in Python (Jupyter & VS Code) covering data ingestion → preprocessing → modeling → evaluation → prototype interface.  
- Conducted thorough **EDA** (box plots, histograms, correlation heatmaps) and addressed distribution skews and missing values to guide feature selection and modeling choices.  
- Benchmarking and hyperparameter tuning of **Decision Tree, SVM, and Neural Network** models using cross-validation; evaluated with accuracy, precision, recall, and F1-score.  
- Delivered a **prototype prediction interface** for symptom input and model inference; documented limitations, ethical/privacy considerations, and future work (larger datasets, explainability).

## **📂 Dataset**
───────────────────────────────────────────────────────────────────────────────
- **Type:** Patient records containing symptoms, vitals (e.g., age, gender, blood pressure, heart rate, cholesterol), and labels for disease presence.  
- **Format:** CSV / tabular (preprocessed into feature matrix + labels).  
- **Preprocessing highlights:** Missing value imputation, normalization/standardization, categorical encoding, class-balance checks.  
- **Privacy:** All handling follows anonymization and privacy best practices as documented in the project report.

## **🧠 Model Architecture & Approach**
───────────────────────────────────────────────────────────────────────────────
- **Algorithms explored:** Decision Tree, Support Vector Machine (SVM), Feedforward Neural Network.  
- **Validation:** k-fold cross-validation for robust performance estimates.  
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score (reported per-class where relevant).  
- **Selection strategy:** Hyperparameter tuning (grid search / random search) and model selection based on balanced metric performance.

## **🧪 Techniques Used**
───────────────────────────────────────────────────────────────────────────────
- Data cleaning: Imputation, outlier handling.  
- Feature engineering: Normalization, categorical encoding, derived features from vitals.  
- EDA: Boxplots, histograms, correlation heatmaps to spot skew and multicollinearity.  
- Model tuning: Cross-validation, hyperparameter search, and performance trade-off analysis.  
- Prototype: Simple UI/script for symptom input → model inference.

## **🚀 Getting Started**
───────────────────────────────────────────────────────────────────────────────

# 1) Data Collection
- Gather a dataset containing patient symptoms, medical history and diagnosis information. 

# 2) Identify the Technolgies to use
- In this case Visual Studio Code & Jupyter Notebook

# 3) Select which Libraries and ML Models to use
- pandas
- numpy
- scikit-learn
- xgboost
- tensorflow       
- matplotlib
- seaborn
- jupyterlab

# 4) Install Libraries

# 5) Prepare dataset folder and example file location

- Place your dataset CSV 
- Expected columns (example): age, gender, heart_rate, blood_pressure, cholesterol, symptom_1, symptom_2, ..., label
- IMPORTANT: Ensure patient data is anonymized and you have data usage permissions.

# 6) Run preprocessing to generate cleaned files and train/test splits

# 7) Train the model 

# 8) Evaluate model performance on test set

# 9) Visualize EDA & results using notebooks

## **📊 Results (example)

───────────────────────────────────────────────────────────────────────────────

- Metrics reported: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.

 -Sample (example): Accuracy: 0.85 | Precision: 0.82 | Recall: 0.80 | F1: 0.81
(See results/ for full metric tables and plots.)

## **🛠️ Tools & Libraries

───────────────────────────────────────────────────────────────────────────────

- Core: Python, Jupyter, VS Code

- Data & ML: pandas, numpy, scikit-learn, xgboost, tensorflow/torch

- Visualization: matplotlib, seaborn


## **💡 Limitations & Ethical Considerations

───────────────────────────────────────────────────────────────────────────────

- Model performance depends heavily on dataset size, quality, and class balance.

- Patient privacy and data security are critical — anonymize data, follow institutional guidelines, and restrict access.

## **🔭 Future Work

───────────────────────────────────────────────────────────────────────────────

- Expand dataset (more samples, diverse sources) and re-evaluate generalization.

- Deploy as a web/mobile app (Streamlit/Flask) with clinician-friendly UI.


## **🙏 Acknowledgements

───────────────────────────────────────────────────────────────────────────────
Thanks to faculty & industry mentors and peers of NMIMS

## **👤 Author

───────────────────────────────────────────────────────────────────────────────
Kalash Shah
