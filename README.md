# regression_ml_app

Insurance Cost Prediction App

This is a Streamlit-based web application that predicts insurance costs based on user inputs such as age, sex, BMI, number of children, smoking habits, and region. The app uses a trained machine learning model to make predictions.

Features
	•	User-friendly interface built with Streamlit.
	•	Predicts insurance costs using a pre-trained regression model.
	•	Accepts key inputs such as age, sex, BMI, smoking status, number of children, and region.
	•	Displays the predicted cost with proper formatting.

How It Works
	1.	Users input their details in the form provided.
	2.	The app processes the input data, encodes categorical variables, and transforms them using preloaded encoders.
	3.	The pre-trained regression model (reg_model.joblib) predicts the insurance cost.
	4.	The prediction is displayed dynamically on the app.

Requirements

Install the required Python packages:

pip install streamlit pandas scikit-learn joblib

How to Run the App
	1.	Clone the repository:

    git clone https://github.com/Tee82-debug/regression_ml_app.git
    cd regression_ml_app


	2.	Make sure the required files are in the same directory:
 
    	•	reg_model.joblib (trained regression model)
    	•	encoder.joblib (one-hot encoder for regions)
    	•	le_sex.joblib (label encoder for sex)
    	•	le_smk.joblib (label encoder for smoker status)
	
 3.	Run the Streamlit app:

    streamlit run app.py


	4.	Open the app in your browser (usually at http://localhost:8501/).

    Inputs
    	•	Age: Numeric input (minimum age: 18).
    	•	Sex: Select box with options Male or Female.
    	•	BMI: Numeric input for Body Mass Index (minimum BMI: 15).
    	•	Number of Children: Numeric input for the number of children.
    	•	Smoker: Select box with options Yes or No.
    	•	Region: Select box with options northeast, northwest, southeast, or southwest.

Outputs
    	•	Predicted insurance cost in USD (formatted to two decimal places).
    	•	If the predicted cost is negative, it displays $0.00.

Technologies Used
    	•	Python: Programming language.
    	•	Streamlit: Web app framework.
    	•	Pandas: Data manipulation.
    	•	Scikit-learn: Machine learning for encoding and predictions.
    	•	Joblib: Model and encoder loading.
