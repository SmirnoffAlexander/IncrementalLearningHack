from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Get values through input bars
        print("@@@@@@@@@@@@@@@@@@@")
        print(request.form.get("file")) 
        # Put inputs to dataframe
        #X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
        
        # Get prediction
        #prediction = clf.predict(X)[0]
        prediction = "AAAA"
    else:
        prediction = ""
        
    return render_template("index.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)
