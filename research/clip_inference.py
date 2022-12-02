import json

import clip
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
        return res_class


if __name__ == "__main__":
    from PIL import Image
    with open('imagenet_classes.json', 'r') as f:
        image_net_names = list(json.load(f).values())
    main_classes = ["a tractor", "a lawnmower", "a bicycle", "a snowboard",
                    "ski", "a truck", "a minibus", "a train", "a dump truck", "a horse"]

    img = Image.open('Classes10/lawnmower2.jpg')

    all_classes = main_classes + image_net_names
    print(len(all_classes))
    model = ClipModel(class_names=all_classes)
    print(model(img))
