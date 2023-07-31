import streamlit as st
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

def main():
    st.title('Car Price Predictor')
    st.write('This app predicts the price of a car you want to sell. Sell your Car in best price to filling the details below:')

    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    selected_company = st.selectbox('Select the company:', ['Select Company'] + companies)
    selected_model = st.selectbox('Select the model:', car_models)
    selected_year = st.selectbox('Select Year of Purchase:', years)
    selected_fuel = st.selectbox('Select the Fuel Type:', fuel_types)
    kilo_driven = st.text_input('Enter the Number of Kilometres that the car has travelled:')

    if st.button('Predict Price'):
        if selected_company == 'Select Company':
            st.error('Please select a company')
        elif not kilo_driven.isdigit():
            st.error('Please enter a valid number of kilometers')
        else:
            prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                          data=np.array([selected_model, selected_company, selected_year, kilo_driven, selected_fuel]).reshape(1, 5)))
            st.success(f"Prediction: ₹{np.round(prediction[0],2)}")

if __name__ == '__main__':
    main()
    
    
##############################################################################################################    
# Flask Code
##############################################################################################################
  
# from flask import Flask,render_template,request,redirect
# from flask_cors import CORS,cross_origin
# import pickle
# import pandas as pd
# import numpy as np

# app=Flask(__name__)
# cors=CORS(app)
# model=pickle.load(open('LinearRegressionModel.pkl','rb'))
# car=pd.read_csv('Cleaned_Car_data.csv')

# @app.route('/',methods=['GET','POST'])
# def index():
#     companies=sorted(car['company'].unique())
#     car_models=sorted(car['name'].unique())
#     year=sorted(car['year'].unique(),reverse=True)
#     fuel_type=car['fuel_type'].unique()

#     companies.insert(0,'Select Company')
#     return render_template('index.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)


# @app.route('/predict',methods=['POST'])
# @cross_origin()
# def predict():

#     company=request.form.get('company')

#     car_model=request.form.get('car_models')
#     year=request.form.get('year')
#     fuel_type=request.form.get('fuel_type')
#     driven=request.form.get('kilo_driven')

#     prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
#                               data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
#     print(prediction)

#     return str(np.round(prediction[0],2))



# if __name__=='__main__':
#     app.run()