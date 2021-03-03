import numpy as np
import pickle
from flask import Flask,render_template,request
app=Flask(__name__)

model=pickle.load(open('titanic.pkl','rb'))

@app.route("/")
@app.route("/home")
def Hello():
	return render_template("home.html")

def age_discrete(x):
    if x<19:return 0
    elif x<30:return 1
    elif x<45:return 2
    else: return 3

def family_discrete(a):
    if a==0: return 0
    elif a<4: return 1
    else: return 2

def fare_discrete(f):
    if f<11: return 0
    elif f<45: return 1
    else: return 2 

@app.route('/predict',methods=['POST'])
def predict():    
    int_features = [int(x) for x in request.form.values()]
    features =np.array(int_features)
    features[2]=age_discrete(features[2])
    features[4]=family_discrete(features[4])
    features[3]=fare_discrete(features[3])
    features=features.reshape(1,-1)
    prediction=model.predict_proba(features)[0,1]*100
    return render_template('home.html', prediction_text='Your chances of survival were {}%'.format(round(prediction,2)))

if __name__=='__main__':
	app.run(host='0.0.0.0',port=8080)
