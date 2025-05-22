from flask import Flask,render_template,request
app=Flask(__name__)
import numpy as np
import pickle
import pandas as pd


with open('regresseion_model.pkl','rb') as f:
    reg_model = pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)
status_map={"Unemployed":0,"Self-Employed":1,"Employed":2}    
edu_map={"High school":0,"Associate":1,"Bachelor":2,"Master":3,"Doctorate":4}
features_to_loglp=["LoanAmount","MonthlyIncome","NetWorth"]
num_cols_to_standarize=['Age', 'CreditScore', 'LoanAmount', 'LoanDuration',
'CreditCardUtilizationRate', 'LengthOfCreditHistory',
'MonthlyIncome', 'NetWorth', 'InterestRate']

@app.route('/')
def home():
    return render_template('form.html',prediction_text='')

@app.route('/prediction',methods=["post"])
@app.route('/prediction', methods=["POST"])
def predict():
    input_data = {
        "Age": int(request.form.get('Age')),
        "CreditScore": int(request.form.get('CreditScore')),
        "EmploymentStatus": request.form.get('EmploymentStatus'),
        "Educationlevel": request.form.get('Educationlevel'),
        "LoanAmount": float(request.form.get('LoanAmount')),
        "LoanDuration": int(request.form.get('LoanDuration')),
        "CreditCardUtilizationRate": float(request.form.get('CreditCardUtilizationRate')),
        "BankruptcyHistory": int(request.form.get('BankruptcyHistory')),
        "PreviousLoanDefaults": int(request.form.get('PreviousLoanDefaults')),
        "LengthOfCreditHistory": int(request.form.get('LengthofCreditHistory')),
        "MonthlyIncome": float(request.form.get('MonthlyIncome')),
        "NetWorth": float(request.form.get('NetWorth')),
        "InterestRate": float(request.form.get('InterestRate'))
    }

    input_df = pd.DataFrame([input_data])

    # Encoding categorical features
    input_df['EmploymentStatus'] = input_df['EmploymentStatus'].map(status_map)
    input_df['Educationlevel'] = input_df['Educationlevel'].map(edu_map)

    # Log transformation
    input_df[features_to_loglp] = input_df[features_to_loglp].apply(np.log1p)

    # Scaling
    input_df[num_cols_to_standarize] = scaler.transform(input_df[num_cols_to_standarize])

    # Prediction
    risk_score = reg_model.predict(input_df)[0]
    risk_score = round(risk_score, 2)

    return render_template('form.html', prediction_text=f"Risk Score: {risk_score}")


if __name__=='__main__':
     app.run(debug=True)