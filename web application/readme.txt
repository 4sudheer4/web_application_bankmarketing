Built a web application to predict a customer's probability of opting to a bank policy/campaign based on a few parameters mentioned in the dataset.

After performing a few tests, we found out that Random forest predicts more accurate results.

We have deployed RandomForest model on to the web application using Flask interface.

With the provided input values in a web form, the values are passed on to the model stored as pickle and trained to display the output on the web application.

Following is the Flask app.py setup:

#loading the model
model = pickle.load(open('model.pkl','rb')) #read mode

#calling the main html page to input the feature values
@app.route("/")
def home():


#creating a predict page to predict based on above given values in the form
@app.route("/predict", methods=['GET','POST'])

Correspondig Predict URL was proided in index.html
<form action="{{ url_for('predict')}}"method="post">

Accessed the data from the above index.html form in using
#access the data from form
        age = int(request.form["age"])
for all the features

#get data into a list for prediction
        input_cols = [[age, job, marital, education, default, housing, loan,contact, month, day, duration, campaign, poutcome, cons_price_idx, cons_conf_idx]]
        

 #loading scalar object (model, scalar, dataframe) to scale the given data 
        with open(       

#performing the scalar operation on the given input record
        test_record = scalar.transform(input_cols)

#predicting the given preprocessed data on model
        prediction = model.predict(input_cols)

#storing the prediction
        output = round(prediction[0], 2)

#helps to visualize the predicted output
@app.route('/visualize')
def visualize():

#fetching the age and duration values into visualize page using 'session'
    age = session.get('age', None)


#plotting age vs duration on the given data
    gr = sns.regplot(x=np.array([age]), y=np.array([duration]), scatter=True, fit_reg=False, marker='o',
            scatter_kws={"s": 100},ax = ax,color = "darkgreen")  # the "s" key in `scatter_kws` modifies the size of the marker

 #sending file to {{ url_for('visualize') }} used in show_graph.html

The following in the show_graph.html will display the graph.
 <img id="image" src="{{ url_for('visualize') }}"
