from flask import Flask, request, render_template, send_file
import os
import shutil
import zipfile
import torch
import json
from clippper import clipInference, ClipModel
from scrape_images import scraper
from run import run_experiment
#from gen_sd import generate_for_class
import pandas as pd

# Declare a Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = ['Inference', 'FineTune', 'Upload']
app.config['MAX_CONTENT_PATH'] = 5000

existed_path = './classes.json'

with open(existed_path, 'r') as f:
    existed_classes = list(json.load(f))

# Clip model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model = ClipModel(class_names=existed_classes, device=DEVICE)

def update_classes(path = existed_path):
    with open(existed_path, 'r') as f:
        existed_classes = list(json.load(f))
    
    return existed_classes

@app.route('/', methods=['GET', 'POST'])
def main():
    # If a form is submitted
    if request.method == "POST":
        
        # Left Panel - Nav Bar - html reloading
        if request.form.get('action') == 'welcome':
            return render_template("index.html")
        elif request.form.get('action') == 'test':
            return render_template("test.html")
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
            
            global clip_model
            clip_model.text_prepare()
            existed_classes = update_classes()
            predictions = clipInference(main_classes=existed_classes, model=clip_model)

            shutil.copyfile('plotsInfer/plot_infer.jpg', 'static/images/plot.jpg')
            if len(predictions) > 1:
                prediction = f"Predicted classes are {', '.join(predictions)}."
            else:
                prediction = f'Predicted class is {predictions[0]}.'
            return render_template("inference.html", output=prediction, plot="../static/images/plot.jpg")

        #-------------------------------------

        # Fine Tuning section - NEW CLASS NAME ->
        #                           1) Clip train
        #                           2) LwF train
        #-------------------------------------
        if request.form.get('upload') == 'Train!':
            print("$$$$$$$$$$$$$$$$$$$$$")
            class_name = request.form['newClass']

            class_name = str(class_name).lower()
            existed_classes = update_classes()
            if class_name not in existed_classes:
                existed_classes.append(class_name)

                with open(existed_path, 'w') as f:
                    json.dump(existed_classes, f)
                print('NEW CLASSES')
                print(existed_classes)

                # 1st Clip TRAIN
                clip_model = ClipModel(class_names=existed_classes, device=DEVICE)

                # 2nd LwF TRAIN
                # TODO: TRAIN METHOD!!!!
                # New Data
                imgCount = 100
                SynthCnt = 20
                scraper(query="class_name", count=imgCount)
                #generate_for_class([class_name], cnt=SynthCnt)

                ## Train
                _config = {
                    'data_path': class_name,
                    'num_epochs': 20,
                    'T': 2,
                    'alpha': 0.01,
                    'num_new_class': 1,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'lr': 1e-2,
                    'momentum': 0.9,
                    'weight_decay': 5e-4,
                    'optimizer': 'SGD',
                    'pretrained_weights_path': 'pretrained_weights/net.pth',
                    'loss': 'CrossEntropy',
                    'batch_size': 4,
                    'num_workers': 4,
                }
                run_experiment(_config)

            return render_template("finetune.html")
        #-------------------------------------

        
        # TEST section ->
        #                           1) Clip test
        #                           2) LwF test
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

            ## TEST CLIP
            tp = 0
            fp = 0
            fn = 0
            cnt = 0
            existed_classes = update_classes()
            clip_model.text_prepare()
            class_cnt = len(os.listdir(ds_path))
            csv_anss = {'id': [], 'predict_1': [], 'predict_2': [], 'predict_3': []}
            for idx, img in enumerate(sorted(os.listdir(ds_path))):
                print(f'{idx+1} class of {class_cnt}')
                #labels = class_name.split('_')
                #class_dir = os.path.join(ds_path, class_name)
                #for img in os.listdir(class_dir):
                if img[-3:] == 'jpg':
                    cnt += 1
                    csv_anss['id'].append(img[:-4])
                    img_path = os.path.join(ds_path, img)

                    prediction = clipInference(path = img_path, main_classes=existed_classes, model=clip_model).tolist()
                    
                    while len(prediction) < 3:
                        prediction.append("")
                    
                    print(prediction)

                    cols = ['predict_1', 'predict_2', 'predict_3']
                    for col, els in zip(cols, prediction):
                        csv_anss[col].append(els)

                    '''DEPRECATED
                    cnt_tp = 0
                    for label in labels:
                        if label in prediction:
                            tp += 1
                            cnt_tp += 1
                        else:
                            fp += 1
                    
                    fn += len(prediction) - cnt_tp
                    '''
            
            ans = pd.DataFrame(data=csv_anss)
            ans.to_csv('./Test.csv', index=False)
            print("saved")

            return send_file('./Test.csv', as_attachment=True) #render_template("test.html")
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
