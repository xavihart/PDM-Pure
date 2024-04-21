from deepfloyd_if.pipelines import style_transfer
from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from diffusers.utils import pt_to_pil, load_image
import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='./demo.png', type=str, help='path of image to be purified')
    parser.add_argument('--save_path', default='./', type=str, help='path to save output image')
    parser.add_argument('--prompt', default='a picture', type=str, help='a sentense to describe the image, can be vague descriptions')
    parser.add_argument('--device', default=0, type=int) # single GPU
    args = parser.parse_args()
    
    
    device = args.device
    
    # LOAD IMAGES
    image_p = args.image
    raw_pil_image = load_image(image_p)
    OUT_SHAPE = raw_pil_image.size
    raw_pil_image_mid = raw_pil_image.resize((256, 256))
    
    # LOAD DEEPFLOYD MODELS
    if_II = IFStageII('IF-II-L-v1.0', device=device)
    if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device)
    t5 = T5Embedder(device=device)
    
    

    # RUN PURIFICATION
    print(f'Begin to purify {args.image}' + '-' * 10)
    with torch.no_grad():
        result = style_transfer(
            t5=t5, if_I=None, if_II=if_II, if_III=if_III,
            support_pil_img=raw_pil_image,
            style_prompt=[
                args.prompt
            ],
            seed=0,
            if_II_kwargs={ 
                "guidance_scale": 7,
                "sample_timestep_respacing": "10,0,0,0,0,0,0,0,0,0",
                "support_noise_less_qsample_steps": 5,
                "low_res": raw_pil_image_mid
            },
            if_III_kwargs={ 
                "guidance_scale": 4.0,
                "sample_timestep_respacing": "50",
            },
            disable_watermark=True
        )
        
        raw_pil_image.save(args.save_path + 'original.png')
        result['III'][0].resize(OUT_SHAPE).save(args.save_path + 'purified.png')

if __name__ == '__main__':
    main()