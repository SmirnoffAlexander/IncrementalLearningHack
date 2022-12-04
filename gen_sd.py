import diff
import os

pipe, device = diff.load_model()

prompts = [
    "tractor",
    "lawnmower",
    "bicycle",
    "snowboard",
    "ski",
    "truck",
    "minibus",
    "train",
    "dump truck",
    "horse",
    "skateboard"
]

def generate_for_class(classes = prompts, cnt: int = 2, root_path = './training_data'):
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    for class_id in classes:
        new_dir = os.path.join(root_path, class_id)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        for iter in range(2):
            image, _ = diff.inference(pipe, device, seed=None, prompt=class_id)
            image.save(f'{new_dir}/synth_{iter}.jpg')