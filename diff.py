import torch
import numpy as np
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import os

def load_model():
    # CUDA or CPU inference
    if torch.cuda.is_available():
        print('CUDA inference started.')
        device = 'cuda'
        # RTX3070ti 8gb is low vram gpu...
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                   revision="fp16",
                                                   torch_dtype=torch.float16,
                                                   use_auth_token=True).to('cuda')
    else:
        print('CPU inference started.')
        device = 'cpu'
        # Optimization for cpu inference
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float32, low_cpu_mem_usage=True, use_auth_token=True).to('cpu')
    
    return pipe, device

def inference(pipe, device, seed: int = None, prompt: str = "A digital Illustration of the Babel tower, 4k, detailed, trending in artstation, fantasy vivid colors"):
    
    if seed is None:
        seed = int(np.random.randint(low=-2**32, high=2**32-1, size=1)[0])
    
    generator = torch.Generator(device).manual_seed(seed)
    with torch.autocast(device):
        image = pipe(prompt, guidance_scale=7.5, generator=generator)['sample'][0]
    return image, seed

def generate_for_class(classes, cnt: int = 2, root_path = './training_data'):
    pipe, device = load_model()

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    for class_id in classes:
        new_dir = os.path.join(root_path, class_id)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        for iter in range(2):
            image, _ = inference(pipe, device, seed=None, prompt=class_id)
            image.save(f'{new_dir}/synth_{iter}.jpg')