from flask import Flask, render_template, request, redirect, session
from joblib import load
import re
import collections
import numpy as np

app = Flask(__name__)


@app.route("/")
def base():
    return render_template('base.html',title="SPAM Prediction")

def Frequencies(text):
    
    variables = ['make', 'address', 'all', '3d', 'our', 
                 'over', 'remove', 'internet', 'order', 
                 'mail', 'receive', 'will', 'people', 'report', 
                 'addresses', 'free', 'business', 'email', 'you', 
                 'credit', 'your', 'font', '000', 'money', 'hp', 'hpl', 
                 'george', '650', 'lab', 'labs', 'telnet', '857', 'data', 
                 '415', '85', 'technology', '1999', 'parts', 'pm', 'direct', 
                 'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 
                 'conference', ';', '(', '[', '!', '$', '#', 
                 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']

    values100 = []
    words = re.findall(r"[\w']+", text)
    words_count = collections.Counter(words)



    for i in range(48):
        if variables[i] in words_count.keys() : values100.append(words_count[variables[i]]/len(words)*100)
        else : values100.append(0.0)



    for i in range(48,54):
        values100.append(100*text.count(variables[i])/(len(text.replace(" ",""))))



    uppercase = sum(1 for letters in text if letters.isupper())
    uppercase_words = re.findall(r"[A-Z]+", text)  

    if uppercase != 0:
        values100.append(uppercase/len(uppercase_words))
        values100.append(len(max(uppercase_words, key=len)))
        values100.append(uppercase)
    else:
        values100.extend((0,0,0))
    
    return values100


@app.route('/predict',methods=["POST"])
def prediction():
    mail = request.form['email']
    model = load('model_saved.joblib')
    finalpredict = str(model.predict([Frequencies(mail)])[0])    
    return render_template('results.html', title = "Prediction", data = finalpredict)







if __name__ == '__main__':
    app.run(host='127.0.0.1',port=4000, debug=True)


