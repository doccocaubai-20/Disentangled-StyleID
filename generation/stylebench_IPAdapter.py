#This code is from IP-Adapter

import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import torch
import glob
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
import os
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from styleprompt import styles



app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(448, 448))

v2 = False
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = "ip-adapter-faceid-plus_sd15.bin" if not v2 else "ip-adapter-faceid-plusv2_sd15.bin"
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

# load ip-adapter
ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)

negative_prompt = "lowres, worst quality, low quality"


#YOUR SOURCE IMAGE PATH HERE
all_paths = glob.glob("source/*.png")

for impath in all_paths:
    image = cv2.imread(impath)
    faces = app.get(image)

    for sty in styles:
        all_level = [val for val in range(1,8) ]

        for level in all_level:
            savedir = impath.replace('source','ipa')
            save_path = savedir.replace('.png',f'_{level}.png')

            if os.path.exists(save_path):
                continue
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            scale= 0.01*2**(8-level)
            guidance_scale = level
            
            faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
            face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224) # you can also segment the face

            prompt = styles[sty]
            images = ip_model.generate(
                 prompt=prompt, negative_prompt=negative_prompt, face_image=face_image, faceid_embeds=faceid_embeds, shortcut=v2, scale=scale, s_scale=scale, guidance_scale=guidance_scale,
                 num_samples=1, width=512, height=512, num_inference_steps=25)

            images[0].save(save_path)
