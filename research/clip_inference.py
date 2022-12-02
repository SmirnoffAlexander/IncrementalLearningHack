import json
import os

import clip
import numpy as np
import plotly.graph_objects as go
import torch


class ClipModel:
    def __init__(self, class_names, backbone_name="RN50", device="cuda"):
        self.class_names = class_names
        self.backbone_name = backbone_name
        self.device = device
        self.model, self.preprocess = clip.load(backbone_name,
                                                device=device)

    def add_class(self, new_class_name):
        self.class_names.append(new_class_name)

    def __call__(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(self.class_names).to(self.device)
        with torch.no_grad():

            logits_per_image, _ = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy().ravel()
            res_class = self.class_names[probs.argmax()]
        return res_class, probs


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
            fig.write_image(plot_img_name)
