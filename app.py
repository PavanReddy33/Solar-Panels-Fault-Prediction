from flask import Flask, render_template, request
import re
import pandas as pd
import copy
import pickle
import joblib
import numpy as np

cleaner = joblib.load('processed1')
model = pickle.load(open('best_random_forest.pkl','rb'))


#define flask
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')



@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data = pd.read_csv(f)
        y2 = pd.DataFrame(model.predict(cleaner.transform(data)),columns=['class'])
        data['class']= y2 
        data.to_csv('results.csv',index=False)       
        return render_template("new.html", Y = data.to_html(justify = 'center'))




# prediction function
def ValuePredictor(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1, 13)
	loaded_model = pickle.load(open('best_random_forest.pkl','rb'))
	result = loaded_model.predict(to_predict)
	return result[0]

@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
		to_predict_list = request.form.to_dict()
		to_predict_list = list(to_predict_list.values())
		to_predict_list = list(map(float, to_predict_list))
		result = ValuePredictor(to_predict_list)	
		if result== "NF":
			prediction ='Congrats! Your Solar Panel Is Not Faulty'
		else:
			prediction ='Sorry! Your Solar Panel Is Faulty'		
		return render_template("result.html", prediction = prediction)


if __name__=='__main__':
    app.run(debug = True)


