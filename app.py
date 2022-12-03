from flask import Flask, request, render_template
import pandas as pd
from werkzeug.utils import secure_filename
import os
import shutil
from clippper import clipInference

# Declare a Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Inference'
app.config['MAX_CONTENT_PATH'] = 5000

@app.route('/', methods=['GET', 'POST'])
def main():
    # If a form is submitted
    if request.method == "POST":
        
        if request.form.get('action') == 'welcome':
            return render_template("index.html")
        elif request.form.get('action') == 'inference' or request.form.get('action') == "try it!":
            shutil.copyfile('images/pic01.jpg', 'static/images/pic01.jpg')
            return render_template("inference.html")
        # Get values through input bars
        print("@@@@@@@@@@@@@@@@@@@")
        file = request.files['file']
        fname = 'infer.jpg'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
        shutil.copyfile('Inference/infer.jpg', 'static/images/pic01.jpg')
        # Put inputs to dataframe
        #X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
        
        # Get prediction
        #prediction = clf.predict(X)[0]
        prediction = f'Predicted class is {clipInference()}.'
        return render_template("inference.html", output=prediction)
    else:
        pass 
    return render_template("index.html")

# Running the app
if __name__ == '__main__':
    app.run(debug = True)
