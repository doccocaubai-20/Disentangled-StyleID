#This code is from InstantID

# !pip install opencv-python transformers accelerate insightface
import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
import glob
import os
import cv2
import torch
import numpy as np
from PIL import Image
from styleprompt import styles

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

# prepare 'antelopev2' under ./models
#app = FaceAnalysis(name='buffalo_l', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

app.prepare(ctx_id=0, det_size=(448, 448))

# prepare models under ./checkpoints
face_adapter = f'./checkpoints/ip-adapter.bin'
controlnet_path = f'./checkpoints/ControlNetModel'

# load IdentityNet
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

base_model = 'wangqixun/YamerMIX_v8'  # from https://civitai.com/models/84040?modelVersionId=196039
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.cuda()

# load adapter
pipe.load_ip_adapter_instantid(face_adapter)


# load an image
generator = torch.Generator(device="cuda")


#YOUR SOURCE IMAGE PATH HERE
all_paths = glob.glob("source/*.png")



for impath in all_paths:
    print(impath)
    face_image = load_image(impath)
    #import pdb; pdb.set_trace()
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])

    for sty in styles:

        # prompt
        prompt = styles[sty]
        negative_prompt = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic"

        all_level = [val for val in range(1,8) ]

        for level in all_level:
            savedir = impath.replace('source','inst')
            save_path = savedir.replace('.png',f'_{level}.png')

            if os.path.exists(save_path):
                continue
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            scale= 0.02*1.8**(8-level)

            guidance_scale = level


            # generate image
            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                image_embeds=face_emb,
                image=face_kps,
                controlnet_conditioning_scale= 1-level/8,
                ip_adapter_scale=scale,
                num_inference_steps=25,
                guidance_scale=guidance_scale,
                generator=generator#.manual_seed(42)
            ).images

            image[0].save(save_path)
