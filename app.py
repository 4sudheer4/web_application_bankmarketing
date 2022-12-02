from flask import Flask, render_template, request, redirect, request, send_file, session, url_for
import pickle 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb')) #read mode
@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        #access the data from form
        ## Age
        age = int(request.form["age"])
        job = int(request.form["job"])
        marital = int(request.form["marital"])
        education = int(request.form["education"])
        default = int(request.form["default"])
        housing = int(request.form["housing"])
        loan = int(request.form["loan"])

        contact = int(request.form["contact"])
        month = int(request.form["month"])
        day = int(request.form["day"])
        duration = int(request.form["duration"])
        campaign = int(request.form["campaign"])
        poutcome = int(request.form["poutcome"])

        cons_price_idx = float(request.form["cons_price_idx"])
        cons_conf_idx = float(request.form["cons_conf_idx"])
        poutcome = float(request.form["poutcome"])


        #get prediction
        input_cols = [[age, job, marital, education, default, housing, loan,contact, month, day, duration, campaign, poutcome, cons_price_idx, cons_conf_idx]]
        
        with open('scale.pkl','rb') as file:
            myvar = pickle.load(file)

        test_record = myvar.transform(input_cols)

        prediction = model.predict(input_cols)
        output = round(prediction[0], 2)
        if output == 1:
            output = 'Yes'
        if output == 0:
            output = 'No'
            
        return render_template("index.html", prediction_text='Your predicted campaign optin status  is  {}'.format(output))
@app.route('/visualize')
def visualize():
    matplotlib.pyplot.switch_backend('Agg')

    fig,ax=plt.subplots(figsize=(6,6))
    ax=sns.set(style="darkgrid")

    x=[i for i in range(100)]
    y=[i for i in range(100)]

    sns.lineplot(x,y)
    canvas=FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img,mimetype='img/png')

if __name__ == "__main__":
    app.run(debug=True)