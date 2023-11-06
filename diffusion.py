import numpy as np
import cv2
import matplotlib.pyplot as plt 
import requests
import datetime
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image
from diffusers import StableDiffusionUpscalePipeline
from diffusers import StableDiffusionXLImg2ImgPipeline

def get_model_path(name = None):

    models = {
        "stable": "CompVis/stable-diffusion-v1-4",
        "realistic": "SG161222/Realistic_Vision_V5.1_noVAE",
        "realistic2": "coreml-community/coreml-realisticVision-v20_cn",
        "dreamshaper": "Lykon/DreamShaper",
        "dreamshaper2": "coreml-community/coreml-DreamShaper-v5.0_cn",
        "deliberate": "peterwilli/deliberate-2",
        "photon": "digiplay/Photon_v1",
        "meinamix": "stablediffusionapi/meinamix",
        "openjourney": "prompthero/openjourney-v4",
        "dreamlike": "dreamlike-art/dreamlike-photoreal-2.0",
        "xl": "stabilityai/stable-diffusion-xl-base-1.0",
    }

    if name is None:
        print("Available models:")
        for name, path in models.items():
            print(f"{name}: {path}")
        return None
        
    return models[name]



def get_controls(control, sc):
    if isinstance(control, str):
        control = [control]

    controlnets= []
    mconds = []
    for ctrl in control:
        if "seg" in ctrl:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
            )
            mcond = Image.fromarray(sc.get_seg_map())
        elif "norm" in ctrl:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-normal", torch_dtype=torch.float16
            )
            mcond = Image.fromarray(sc.get_norm_map())
        elif "depth" in ctrl:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
            )
            mcond = Image.fromarray(sc.get_depth_map())
        else:
            throw("Unknown controlnet")
        controlnets.append(controlnet)
        mconds.append(mcond)
    return controlnets, mconds


def run_diffusion(scene , device = "cuda", prompt = None, model = "realistic", control = "seg", steps = 200):
    if prompt is None:
        prompt = "a countryside farm. Epic realistic, (hdr:1.4), (muted colors:1.4), abandoned, (intricate details), (intricate details, hyperdetailed:1.4), artstation, vignette"

    model_id_or_path = get_model_path(model)

    controlnets, mconds = get_controls(control, scene)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id_or_path, controlnet=controlnets, safety_checker=None, torch_dtype=torch.float16
    )

    pipe = pipe.to(device)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()

    image = pipe(prompt, mconds, num_inference_steps=steps).images[0]

    stamp_image = np.asarray(image)

    scene.set_image(stamp_image)

    return stamp_image, mconds


def present(src, results):
    n = len(results)
    assert(n==len(src))
    m = len(src[0]) + 1

    fig, ax= plt.subplots(n,m, figsize=(25*n, 25*m))
    
    
    if n==1:
        ax = [ax]
    
    for i in range(n):
        for j in range(m-1):
            img = src[i][j]
            ax[i][j].imshow(src[i][j])
            ax[i][j].grid(False)
            ax[i][j].axis('off')
            
        ax[i][m-1].imshow(results[i])
        ax[i][m-1].grid(False)
        ax[i][m-1].axis('off')
    plt.savefig("images/prespective_change_with_diffusion.png")
    plt.show()


def run_img2img(scene , device = "cuda", prompt = None, model = "realistic", control = "seg", steps = 200, strength = 1.0):
    if prompt is None:
        prompt = "a countryside farm. Epic realistic, (hdr:1.4), (muted colors:1.4), abandoned, (intricate details), (intricate details, hyperdetailed:1.4), artstation, vignette"

    model_id_or_path = get_model_path(model)

    controlnets, mconds = get_controls(control, scene)

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_id_or_path, controlnet=controlnets, safety_checker=None, torch_dtype=torch.float16
    )
    
    # pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    #     model_id_or_path, controlnet=controlnets, safety_checker=None, torch_dtype=torch.float16
    # )

    pipe = pipe.to(device)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()

    start_image = scene.get_image()

    image = pipe(prompt, image=start_image, control_image= mconds, 
                         strength = strength, num_inference_steps=steps,
                         height=1024, width=1024).images[0]
    
    stamp_image = np.asarray(image)

    scene.set_image(stamp_image)

    return stamp_image, mconds

def run_inpainting(scene , device = "cuda", prompt = None, model = "realistic", control = "seg", steps = 200, strength = 1.0):
    if prompt is None:
        prompt = "a countryside farm. Epic realistic, (hdr:1.4), (muted colors:1.4), abandoned, (intricate details), (intricate details, hyperdetailed:1.4), artstation, vignette"

    model_id_or_path = get_model_path(model)

    controlnets, mconds = get_controls(control, scene)

    # pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    #     model_id_or_path, controlnet=controlnets, safety_checker=None, torch_dtype=torch.float16
    # )
    
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        model_id_or_path, controlnet=controlnets, safety_checker=None, torch_dtype=torch.float16
    )

    pipe = pipe.to(device)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()

    start_image = scene.get_image()

    mask = np.where(start_image == (0,0,0), 1, 0).sum(2)==3
    mask = mask.astype(np.uint8)*255

    ##import pdb; pdb.set_trace()

    image = pipe(prompt, image=start_image, mask_image=mask, control_image= mconds, 
                         strength = strength, num_inference_steps=steps,
                         height=1024, width=1024).images[0]

    stamp_image = np.asarray(image)

    scene.set_image(stamp_image)



    return stamp_image, mconds


def run_refine(scene , device = "cuda", prompt = None, steps = 200, strength = 1.0):
    if prompt is None:
        prompt = "a countryside farm. Epic realistic, (hdr:1.4), (muted colors:1.4), abandoned, (intricate details), (intricate details, hyperdetailed:1.4), artstation, vignette"

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )

    pipe = pipe.to(device)

    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()

    start_image = scene.get_image()

    ##import pdb; pdb.set_trace()

    image = pipe(prompt, init_image=start_image, strength = strength, 
                         num_inference_steps=steps,
                         height=1024, width=1024).images[0]

    stamp_image = np.asarray(image)

    scene.set_image(stamp_image)

    return stamp_image