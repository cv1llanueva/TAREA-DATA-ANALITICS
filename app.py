from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# Importar los modelos
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
en = pickle.load(open('encoder.pkl','rb'))

# crear flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    G = request.form['Gender']
    A = request.form['Age']
    Ai = request.form['Anual_Income']
    SS = request.form['Spending_Score']
    

    feature_list = [G, A,Ai,SS]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = en.transform(single_pred)
    scaled_features[:,2:] = sc.transform(scaled_features[:,2:])
    prediction = model.predict(scaled_features)

    diccionario = {1: "Grupo 1", 2: "Grupo 2", 3: "Grupo 3", 4: "Grupo 4", 5: "Grupo 5"}

    if prediction[0] in diccionario:
        crop = diccionario[prediction[0]]
        result =("El cluster al que pertenece es: {} ".format(crop))
    else:
        result =("Sorry, Hoy no se come")
    return render_template('index.html',result = result)




# python main
if __name__ == "__main__":
    app.run(debug=True)