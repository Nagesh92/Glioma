from flask import Flask,request,render_template
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    float_val = [float(a) for a in request.form.values()]
    final_val = np.array(float_val)
    predictions = model.predict(final_val)

    if predictions==0:
        output="Presence of Low Grade Glioma in the Human brain"
    
    else:
        output="Presence of GlioBlastoma Multiforme in the Human Brain"



