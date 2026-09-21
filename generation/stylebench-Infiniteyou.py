#This code is from InfiniteYou

import argparse
import os
import glob
import torch
from PIL import Image
from styleprompt import styles
from pipelines.pipeline_infu_flux import InfUFluxPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_image', default='./assets/examples/man.jpg', help="""input ID image""")
    parser.add_argument('--control_image', default=None, help="""control image [optional]""")
    parser.add_argument('--prompt', default='A man, portrait, cinematic')
    parser.add_argument('--base_model_path', default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--model_dir', default='ByteDance/InfiniteYou')
    parser.add_argument('--infu_flux_version', default='v1.0', help="""InfiniteYou-FLUX version: currently only v1.0""")
    parser.add_argument('--model_version', default='aes_stage2', help="""model version: aes_stage2 | sim_stage1""")
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int, help="""seed (0 for random)""")
    parser.add_argument('--guidance_scale', default=3.5, type=float)
    parser.add_argument('--num_steps', default=20, type=int)
    parser.add_argument('--infusenet_conditioning_scale', default=1.0, type=float)
    parser.add_argument('--infusenet_guidance_start', default=0.0, type=float)
    parser.add_argument('--infusenet_guidance_end', default=1.0, type=float)
    # The LoRA options below are entirely optional. Here we provide two examples to facilitate users to try, but they are NOT used in our paper.
    parser.add_argument('--enable_realism_lora', action='store_true')
    parser.add_argument('--enable_anti_blur_lora', action='store_true')
    # Memory reduction options
    parser.add_argument('--quantize_8bit', action='store_true')
    parser.add_argument('--cpu_offload', action='store_true')
    args = parser.parse_args()

    # Check arguments
    assert args.infu_flux_version == 'v1.0', 'Currently only supports InfiniteYou-FLUX v1.0'
    assert args.model_version in ['aes_stage2', 'sim_stage1'], 'Currently only supports model versions: aes_stage2 | sim_stage1'

    # Set cuda device
    torch.cuda.set_device(args.cuda_device)

    # Load pipeline
    infu_model_path = os.path.join(args.model_dir, f'infu_flux_{args.infu_flux_version}', args.model_version)
    insightface_root_path = os.path.join(args.model_dir, 'supports', 'insightface')
    pipe = InfUFluxPipeline(
        base_model_path=args.base_model_path,
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root_path,
        infu_flux_version=args.infu_flux_version,
        model_version=args.model_version,
        quantize_8bit=args.quantize_8bit,
        cpu_offload=args.cpu_offload,
    )
    # Load LoRAs (optional)
    lora_dir = os.path.join(args.model_dir, 'supports', 'optional_loras')
    if not os.path.exists(lora_dir): lora_dir = './models/InfiniteYou/supports/optional_loras'
    loras = []
    if args.enable_realism_lora:
        loras.append([os.path.join(lora_dir, 'flux_realism_lora.safetensors'), 'realism', 1.0])
    if args.enable_anti_blur_lora:
        loras.append([os.path.join(lora_dir, 'flux_anti_blur_lora.safetensors'), 'anti_blur', 1.0])
    pipe.load_loras(loras)
    
    # Perform inference
    # /source/kwan/styleid/data/p1/00140.png
    if args.seed == 0:
        args.seed = torch.seed() & 0xFFFFFFFF



    

    #YOUR SOURCE IMAGE PATH HERE
    all_paths = glob.glob("source/*.png")

    for impath in all_paths:
        simdata = Image.open(impath).convert('RGB')
        for sty in styles:
            prompt = styles[sty]
            all_level = [val for val in range(1,8) ]

            for level in all_level:
                savedir = impath.replace('source','inf')
                save_path = savedir.replace('.png',f'_{level}.png')
                if os.path.exists(save_path):
                    continue
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                image = pipe(
                    id_image=simdata,
                    prompt=prompt,
                    control_image= None,
                    guidance_scale=level+0.2,
                    num_steps=args.num_steps,
                    infusenet_conditioning_scale=1.1- 0.15*level,
                    infusenet_guidance_start=args.infusenet_guidance_start,
                    infusenet_guidance_end=args.infusenet_guidance_end,
                    cpu_offload=args.cpu_offload,
                    width=512,
                    height=512
                )

                # Save results
                image.save(save_path)


if __name__ == "__main__":
    main()
