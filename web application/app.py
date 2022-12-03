from flask import Flask, render_template, request, redirect, request, send_file, session, url_for
import pickle 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
import io
from flask_session import Session
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

#loading the model
model = pickle.load(open('model.pkl','rb')) #read mode

#calling the main html page to input the feature values
@app.route("/")
def home():
    return render_template('index.html')

#creating a predict page to predict based on above given values in the form
@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        
        #access the data from form
        age = int(request.form["age"])
        job = int(request.form["job"])
        marital = int(request.form["marital"])
        education = int(request.form["education"])
        default = int(request.form["default"])
        housing = int(request.form["housing"])
        loan = int(request.form["loan"])

        contact = int(request.form["contact"])
        month = int(request.form["month"])
        day = int(request.form["Day of week"])
        duration = int(request.form["duration"])
        campaign = int(request.form["campaign"])
        poutcome = int(request.form["poutcome"])

        cons_price_idx = float(request.form["cons_price_idx"])
        cons_conf_idx = float(request.form["cons_conf_idx"])
        poutcome = float(request.form["poutcome"])

        session['age'] = age
        session['duration'] = duration

        #get data into a list for prediction
        input_cols = [[age, job, marital, education, default, housing, loan,contact, month, day, duration, campaign, poutcome, cons_price_idx, cons_conf_idx]]
        
        #loading scalar object to scale the given data 

        with open('scale.pkl','rb') as file:
            scalar = pickle.load(file)

        #loading the dataframe to plot
        with open('df.pkl','rb') as file:
            df = pickle.load(file)

        session['df'] = df

        #performing the scalar operation on the given input record
        test_record = scalar.transform(input_cols)

        #predicting the given preprocessed data on model
        prediction = model.predict(input_cols)

        #storing the prediction
        output = round(prediction[0], 2)
        if output == 1:
            output = 'Yes'
        if output == 0:
            output = 'No'
            
        return render_template("show_graph.html", prediction_text='Your predicted campaign optin status  is  {}'.format(output))

#helps to visualize the predicted output

@app.route('/visualize')
def visualize():

    matplotlib.pyplot.switch_backend('Agg')

    #fetching the age and duration values into visualize page using 'session'
    age = session.get('age', None)
    duration = session.get('duration', None)

    fig,ax=plt.subplots(figsize=(6,6))
    ax=sns.set(style="darkgrid")

    #plotting age vs duration on the given data
    gr = sns.regplot(x=np.array([age]), y=np.array([duration]), scatter=True, fit_reg=False, marker='o',
            scatter_kws={"s": 100},ax = ax,color = "darkgreen")  # the "s" key in `scatter_kws` modifies the size of the marker

    gr.set(xlabel='Age', ylabel='Duration')

    gr.set_ylim(0, 5000)
    gr.set_xlim(20, 100)

    canvas=FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)

    #sending file to {{ url_for('visualize') }} used in show_graph.html

    return send_file(img,mimetype='img/png')

if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)