import json
import os

import clip
import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image


class ClipModel:
    def __init__(self, class_names, backbone_name="RN50", device="cuda"):
        self.class_names = class_names
        self.backbone_name = backbone_name
        self.device = device
        self.model, self.preprocess = clip.load(backbone_name,
                                                device=device)

    def add_class(self, new_class_name):
        self.class_names.append(new_class_name)

    def text_prepare(self):
        self.emb_text = clip.tokenize(self.class_names).to(self.device)

    def __call__(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        #text = clip.tokenize(self.class_names).to(self.device)
        text = self.emb_text
        with torch.no_grad():
            logits_per_image, _ = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy().ravel()
            res_class = np.array(self.class_names)[probs > 0.15] # TODO: I'VE CNAGED!!!!!!!!!
        return res_class, probs

def clipInference(
    path: str="Inference/infer.jpg", 
    plots_path: str = 'plotsInfer',
    main_classes = ["a tractor", "a lawnmower", "a bicycle", "a snowboard",
                    "ski", "a truck", "a minibus", "a train", "a dump truck", "a horse", 'a skateboard'],
    model = None,
):
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
        
    plot_img_name = os.path.join(plots_path, "plot_infer.jpg")
    img = Image.open(path)

    if model is None:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ClipModel(class_names=main_classes, device=DEVICE)

    res_class, probs = model(img)
    class_inds = probs.argsort()[::-1][:5]
    pred_probs = probs[class_inds]
    pred_classes = np.array(main_classes)[class_inds].tolist()
    fig = go.Figure([go.Bar(x=pred_classes, y=pred_probs)])
    fig.write_image(plot_img_name)

    return res_class



if __name__ == "__main__":
    from PIL import Image
    with open('imagenet_classes.json', 'r') as f:
        image_net_names = list(json.load(f).values())
    main_classes = ["a tractor", "a lawnmower", "a bicycle", "a snowboard",
                    "ski", "a truck", "a minibus", "a train", "a dump truck", "a horse"]
    all_classes = main_classes + image_net_names
    for cl in os.listdir('images'):
        os.makedirs(os.path.join('plots', cl), exist_ok=True)
        for img_name in os.listdir(os.path.join('images', cl)):
            plot_img_name = os.path.join('plots', cl, "plot_"+img_name)
            full_img_name = os.path.join('images', cl, img_name)
            img = Image.open(full_img_name)

            model = ClipModel(class_names=main_classes)
            res_class, probs = model(img)
            class_inds = probs.argsort()[::-1][:5]
            pred_probs = probs[class_inds]
            pred_classes = np.array(main_classes)[class_inds].tolist()
            fig = go.Figure([go.Bar(x=pred_classes, y=pred_probs)])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.write_image(plot_img_name)
