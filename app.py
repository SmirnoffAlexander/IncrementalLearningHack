from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        if request.form.get('action1') == 'HOME':
            return render_template("index.html")
        # Get values through input bars
        print("@@@@@@@@@@@@@@@@@@@")
        print(id)
        print(request.form.get("file")) 
        # Put inputs to dataframe
        #X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
        
        # Get prediction
        #prediction = clf.predict(X)[0]
        prediction = "AAAA"
        return render_template("generic.html", output=prediction)
    else:
        return render_template("index.html")

# Running the app
if __name__ == '__main__':
    app.run(debug = True)
