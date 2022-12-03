from flask import Flask, request, render_template
import os
import shutil
import zipfile
import glob
from clippper import clipInference

# Declare a Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = ['Inference', 'FineTune']
app.config['MAX_CONTENT_PATH'] = 5000

@app.route('/', methods=['GET', 'POST'])
def main():
    # If a form is submitted
    if request.method == "POST":
        
        # Left Panel - Nav Bar - html reloading
        if request.form.get('action') == 'welcome':
            return render_template("index.html")
        elif request.form.get('action') == 'architecture':
            return render_template("architecture.html")
        elif request.form.get('action') == 'fine tuning':
            return render_template("finetune.html")
        elif request.form.get('action') == 'inference' or request.form.get('action') == "try it!":
            shutil.copyfile('images/pic01.jpg', 'static/images/pic01.jpg')
            return render_template("inference.html")

        # Inference section TODO: name button
        #-------------------------------------
        if request.form.get('upload') == 'Run':
            # Get values through input bars
            print("@@@@@@@@@@@@@@@@@@@")
            file = request.files['file']
            fname = 'infer.jpg'
            file.save(os.path.join(app.config['UPLOAD_FOLDER'][0], fname))
            shutil.copyfile('Inference/infer.jpg', 'static/images/pic01.jpg')
            # Put inputs to dataframe
            #X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
            
            # Get prediction
            #prediction = clf.predict(X)[0]
            prediction = f'Predicted class is {clipInference()}.'
            return render_template("inference.html", output=prediction)

        #-------------------------------------

        # Fine Tuning section TODO: name button
        #-------------------------------------
        if request.form.get('upload') == 'Upload Dataset':
            print("#####################")
            file = request.files['newData']
            fname = 'newds.zip'
            zip_path = os.path.join(app.config['UPLOAD_FOLDER'][1], fname)
            file.save(zip_path)
            ds_path = os.path.join(app.config['UPLOAD_FOLDER'][1], 'data')

            if not os.path.exists(ds_path):
                os.makedirs(ds_path)

            with os.scandir(ds_path) as entries:
                for entry in entries:
                    if entry.is_dir() and not entry.is_symlink():
                        shutil.rmtree(entry.path)
                    else:
                        os.remove(entry.path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(ds_path)

            return render_template("finetune.html")
        #-------------------------------------
    else:
        pass 
    return render_template("index.html")

# Running the app
if __name__ == '__main__':
    for root_path in app.config['UPLOAD_FOLDER']:
        if not os.path.exists(root_path):
            os.makedirs(root_path)
    app.run(debug = True)
