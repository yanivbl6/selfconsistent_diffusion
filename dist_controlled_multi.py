from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
import numpy as np
from PIL import Image

from diffusion import *

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection


from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers import StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline

import torch.distributed as dist

import matplotlib.pyplot as plt

def is_compiled_module(module):
    return hasattr(module, "_orig_mod")

def is_torch_version(operator, version):
    return eval(f"torch.__version__ {operator} '{version}'")

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views

def get_mapping2(device, scene1, scene2, scene1_rank, scene2_rank, interpolate_k = 2, reshuffle = True):

    if not dist.is_initialized():
        return get_mapping(device, scene1, scene2, scene1_rank, scene2_rank, interpolate_k = interpolate_k)

    print(f'[INFO] pairing scenes')
    if scene1_rank == scene2_rank:
        map1 = torch.tensor([[]])
        map2 = torch.tensor([[]])
        return map1, map2

    
    map1,map2 = scene1.connect(scene2, interpolate_k = interpolate_k)

    
    map1 = map1.astype(np.long)
    map2 = map2.astype(np.long)
    
    ## convert np to torch, and move to device
    ## the map is for the image space, so we need to convert it to the latent space (Is it how it is done?)
    map1 = map1 // 8
    map2 = map2 // 8

    if scene1_rank > scene2_rank:

        shape_tensor = torch.tensor([map1.shape[1]], dtype = torch.long).to(device).contiguous()
        dist.send(shape_tensor, dst=scene2_rank, tag = 0)

        map1 = torch.tensor(map1, dtype = torch.long).to(device).contiguous()
        map2 = torch.tensor(map2, dtype = torch.long).to(device).contiguous()
        dist.send(map1, dst=scene2_rank, tag = 1)
        dist.send(map2, dst=scene2_rank, tag = 2)
    else:
        shape_tensor = torch.empty([1], dtype = torch.long, device = device)
        dist.recv(shape_tensor, src=scene2_rank, tag = 0)
        shape_tensor = shape_tensor.cpu().item()
        map1_ex = torch.empty([2, shape_tensor], dtype = torch.long, device = device)
        map2_ex = torch.empty([2, shape_tensor], dtype = torch.long, device = device)

        dist.recv(map1_ex, src=scene2_rank, tag = 1)
        dist.recv(map2_ex, src=scene2_rank, tag = 2)

        map1_ex = map1_ex.cpu().numpy()
        map2_ex = map2_ex.cpu().numpy()

        map1 = np.concatenate([map1, map2_ex], axis = 1)
        map2 = np.concatenate([map2, map1_ex], axis = 1)

        ## reshuffle the map on dimension 1:
        if reshuffle:
            indices = np.arange(map1.shape[1])
            np.random.shuffle(indices)
            map1 = map1[:,indices]
            map2 = map2[:,indices]

        new_map1 = []
        new_map2 = []
        for i in range(map1.shape[1]):
            val1 = tuple(map1[:,i])
            val2 = tuple(map2[:,i])
            if not val1 in new_map1 and not val2 in new_map2:
                new_map1.append(val1)
                new_map2.append(val2)
                
        map1 = np.asarray(new_map1, dtype = np.long).transpose()
        map2 = np.asarray(new_map2, dtype = np.long).transpose()

        map1 = torch.tensor(map1, dtype = torch.long).to(device).contiguous()
        map2 = torch.tensor(map2, dtype = torch.long).to(device).contiguous()

    if scene1_rank < scene2_rank:

        shape_tensor = torch.tensor([map1.shape[1]], dtype = torch.long).to(device).contiguous()
        dist.send(shape_tensor, dst=scene2_rank, tag = 0)

        dist.send(map1, dst=scene2_rank, tag = 1)
        dist.send(map2, dst=scene2_rank, tag = 2)
        return map1, map2
    else:
        shape_tensor = torch.empty([1], dtype = torch.long, device = device)
        dist.recv(shape_tensor, src=scene2_rank, tag = 0)
        shape_tensor = shape_tensor.cpu().item()
        map1 = torch.empty([2, shape_tensor], dtype = torch.long, device = device)
        map2 = torch.empty([2, shape_tensor], dtype = torch.long, device = device)

        dist.recv(map1, src=scene2_rank, tag = 1)
        dist.recv(map2, src=scene2_rank, tag = 2)
        return map2, map1




def get_mapping(device, scene1, scene2, scene1_rank, scene2_rank, interpolate_k = 2):


    print(f'[INFO] pairing scenes')
    if scene1_rank == scene2_rank:
        map1 = torch.tensor([[]])
        map2 = torch.tensor([[]])
        return map1, map2


    if scene1_rank < scene2_rank:
        map1,map2 = scene1.connect(scene2, interpolate_k = interpolate_k)
    else:
        map1,map2 = scene2.connect(scene1, interpolate_k = interpolate_k)




    map1 = map1.astype(np.long)
    map2 = map2.astype(np.long)
    
    ## convert np to torch, and move to device
    ## the map is for the image space, so we need to convert it to the latent space (Is it how it is done?)
    map1 = map1 // 8
    map2 = map2 // 8

    new_map1 = []
    new_map2 = []
    for i in range(map1.shape[1]):
        val1 = tuple(map1[:,i])
        val2 = tuple(map2[:,i])
        if not val1 in new_map1 and not val2 in new_map2:
            new_map1.append(val1)
            new_map2.append(val2)
            
    map1 = np.asarray(new_map1, dtype = np.long).transpose()
    map2 = np.asarray(new_map2, dtype = np.long).transpose()

    map1 = torch.tensor(map1, dtype = torch.long).to(device).contiguous()
    map2 = torch.tensor(map2, dtype = torch.long).to(device).contiguous()

    if scene1_rank < scene2_rank:
        if dist.is_initialized():
            dist.send(map1, dst=scene2_rank, tag = 0)
            dist.send(map2, dst=scene2_rank, tag = 1)
        return map1, map2
    else:

        if dist.is_initialized():
            rmap1 = torch.zeros_like(map1)
            rmap2 = torch.zeros_like(map2)
            dist.recv(rmap1, src=scene2_rank, tag = 0)
            dist.recv(rmap2, src=scene2_rank, tag = 1)

            if not torch.all(map1 == rmap1):
                mismatches_count = torch.where(map2 != rmap1)[0].shape[0]

                print(f'[ERROR] rank {scene1_rank} received from rank {scene2_rank}, {mismatches_count} mismatches')

            if not torch.all(map2 == rmap2):
                mismatches_count = torch.where(map1 != rmap2)[0].shape[0]
                print(f'[ERROR] rank {scene1_rank} received from rank {scene2_rank}, {mismatches_count} mismatches')

        return map2, map1

class DistMultiDiffusion(nn.Module):
    def __init__(self, device, control, scene1, model='2.0' , hf_key=None):
        super().__init__()

        self.device = device
        self.version = model

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            model_key =  get_model_path(self.version) #For custom models or fine-tunes, allow people to use arbitrary versions
            #raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model



        controlnets, self.mconds = get_controls(control, scene1)

        if model == "xl":
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(model_key, controlnet=controlnets, safety_checker=None, torch_dtype=torch.float16)
        else:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(model_key, controlnet=controlnets, safety_checker=None, torch_dtype=torch.float16)
        
        
        
        pipe = pipe.to(self.device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        self.rank = dist.get_rank()
        # self.scheduler1 = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        # self.scheduler2 = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")


        self.pipe = pipe
        self.controlnets = controlnets
        self.world_size = 0
        self.iter = 0
        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def find_all_differences(self, stage = ""):
        print("[INFO] finding all differences, stage = ", stage)
        
        mismatches = 0
        mismatches+= self.find_differences(self.pipe.unet)
        for cnt in self.controlnets:
            mismatches+= self.find_differences(cnt)

        mismatches+= self.find_differences(self.pipe.vae)
        mismatches+= self.find_differences(self.pipe.text_encoder)
        mismatches+= self.find_differences(self.pipe.tokenizer)

        print(f'[INFO] total mismatches = {mismatches}')

    @torch.no_grad()
    def find_differences(self, module):

        if not hasattr(module, 'state_dict'):
            return 0

        st_dict = module.state_dict()

        Ks = []
        mistmatch = 0
        for k, v in st_dict.items():
            if not v is None and isinstance(v, torch.Tensor) and not torch.isnan(v).any():
                v2 = v.clone()
                dist.all_reduce(v2, op=dist.ReduceOp.AVG)
                if torch.allclose(v, v2, atol=1e-05):
                    Ks.append(k)
                else:
                    max_diff = "%.09f" % (torch.max((v - v2)**2)).item()
                    avg_diff = "%.09f" % (torch.mean((v - v2)**2)).item()
                    print(f'mismatch at {k}, excluding from allreduce, max_diff = {max_diff}, avg_diff = {avg_diff}')
                    mistmatch = mistmatch + 1
        if mistmatch > 0:
            print(f'[INFO] module {str(module)[0:20]} has {mistmatch} mismatches')
        self.Ks = Ks

        return mistmatch


    @torch.no_grad()
    def set_mappings(self, maps):
        if not isinstance(maps, list):
            maps = [maps]
        
        self.world_size = len(maps)
        self.maps = [map.clone() for map in maps]

    @torch.no_grad()
    def process_cond(self, conds, batch_size = 1, num_images_per_prompt = 1, do_classifier_free_guidance = False, guess_mode = False):
        assert(self.world_size > 0)

        controlnet = self.pipe.controlnet

        if isinstance(controlnet, ControlNetModel):
            width, height = conds.size
            image = self.prepare_image(
                image=conds,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=self.device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []
            
            for image_ in conds:
                width, height = image_.size

                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=self.device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        return image


    @torch.no_grad()
    def all2all(self, x, alpha):
        srcs = []
        dsts = []

        self.iter = self.iter + 1

        for r, map in enumerate(self.maps):

            if r == self.rank:
                srcs.append(torch.empty([], dtype = torch.float16 ,  device = self.device))
            else:  
                srcs.append(x[:,:,map[0],map[1]].clone())
            dsts.append(torch.empty_like(srcs[-1]))

        srcs_to_send = [src.clone().contiguous() for src in srcs]

        dist.all_to_all(dsts, srcs_to_send)

        ## assert correction
        check_all2all = False
        if check_all2all:
            dsts_copy = [dst.clone() for dst in dsts]
            src_again = [torch.empty_like(dst) for dst in dsts]
            dist.all_to_all(src_again, dsts_copy)
            for i in range(len(src_again)):
                if i != self.rank:
                    assert(not torch.allclose(src_again[i], dsts[i]))
                    if not torch.all(src_again[i] ==  srcs[i]):
                        print(f'[ERROR] rank {self.rank} received from rank {i}, all_to_all failed')

                        different_indices = torch.where(src_again[i] != srcs[i])
                        different_values1 = src_again[i][different_indices]
                        different_values2 = srcs[i][different_indices]

                        print(f'[ERROR] rank {self.rank} received from rank {i}, all_to_all failed, {different_indices[0].shape[0]} mismatches')
                        print(f'mistmatches: {different_values1} <-> {different_values2}')

        for r, map in enumerate(self.maps):

            if r == self.rank:
                continue



            diff = (srcs[r] - dsts[r])/2
            x[:,:,map[0],map[1]] -= diff * alpha
            
            # if self.iter % 50 == 0:
            #     print(f'[INFO] rank {self.rank} received from rank {r}: ')
            #     print(f'{srcs[r][:,:,0:3].numpy()}')
            #     print(f'{dsts[r][:,:,0:3].numpy()}')
            #     print(f'{diff[:,:,0:3].numpy()}')
            #     print("-" * 50)
        return x

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        pairing_strength: float = 0.1,
        max_pairing_steps: int = -1,
        display_every: int = -1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the controlnet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the controlnet stops applying.
            pairing_strength (`float`, *optional*, defaults to 0.1):
                The strength of the pairing between the two scenes.
            max_pairing_steps (`int`, *optional*, defaults to -1):
                The maximum number of steps for pairing. If -1, it will be set to `num_inference_steps//2`.
            display_every (`int`, *optional*, defaults to -1):
                The frequency at which the generated images will be displayed. If -1, it will be set to
                `num_inference_steps//10`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if max_pairing_steps < 0:
            max_pairing_steps = num_inference_steps//2

        controlnet = self.pipe.controlnet._orig_mod if is_compiled_module(self.pipe.controlnet) else self.pipe.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 1. Check inputs. Raise error if not correct.
        self.check_inputs(prompt, self.mconds, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds, controlnet_conditioning_scale, control_guidance_start, control_guidance_end)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipe._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare image
        self.cond_img = self.process_cond(self.mconds)

        image = self.cond_img

        height = self.cond_img[0].shape[-2]
        width = self.cond_img[0].shape[-1]

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps



        # 6. Prepare latent variables
        num_channels_latents = self.pipe.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        ##force the latents to start at the same point
        
        ##TODO : latents2[:,:,self.map2[0],self.map2[1]] = latents1[:,:,self.map1[0],self.map1[1]]

        
        ##latents = self.all2all(latents, 1.0)
            
    
        pbar = self.pipe.progress_bar(total=num_inference_steps)
    
        with pbar as progress_bar:
            for i, t in enumerate(timesteps):


                ##if latents has nan:
                if torch.isnan(latents).any():
                    import pdb; pdb.set_trace()

                ##import pdb; pdb.set_trace()

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    cond_scale = controlnet_conditioning_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )



                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]


                if pairing_strength > 0:
                    
                    if i < max_pairing_steps:
                        alpha = pairing_strength
                    else:
                        alpha = pairing_strength * (1 - (i - max_pairing_steps) / (num_inference_steps - max_pairing_steps))

                    latents = self.all2all(latents.clone(), alpha)

                    # diff = (latents1[:,:,self.map1[0],self.map1[1]] - latents2[:,:,self.map2[0],self.map2[1]])/2

                    # # update the latent
                    # latents1[:,:,self.map1[0],self.map1[1]] = latents1[:,:,self.map1[0],self.map1[1]] - diff * alpha
                    # latents2[:,:,self.map2[0],self.map2[1]] = latents2[:,:,self.map2[0],self.map2[1]] + diff * alpha

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    # if callback is not None and i % callback_steps == 0:
                    #     callback(i, t, latents1)
                    #     callback(i, t, latents2)


        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self.pipe, "final_offload_hook") and self.pipe.final_offload_hook is not None:
            self.pipe.unet.to("cpu")
            self.pipe.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            outimage = self.pipe.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            ##_, has_nsfw_concept = self.pipe.run_safety_checker(image1, device, prompt_embeds.dtype)
            ##_, has_nsfw_concept = self.pipe.run_safety_checker(image2, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            outimage = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * outimage.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        outimage = self.pipe.image_processor.postprocess(outimage, output_type=output_type, do_denormalize=do_denormalize)


        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.pipe.final_offload_hook is not None:
            self.pipe.final_offload_hook.offload()

        if not return_dict:
            return (outimage[0], has_nsfw_concept)

        return outimage[0]
    

    def enable_vae_slicing(self):
        return self.vae.enable_slicing()

    def disable_vae_slicing(self):
        return self.vae.disable_slicing()

    def enable_vae_tiling(self):
        return self.vae.enable_tiling()

    def disable_vae_tiling(self):
        return self.vae.disable_tiling()

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        return self.pipe._encode_prompt( prompt, device, num_images_per_prompt, do_classifier_free_guidance, 
                                 negative_prompt, prompt_embeds, negative_prompt_embeds, lora_scale)

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        return self.pipe.encode_prompt( prompt, device, num_images_per_prompt, do_classifier_free_guidance, 
                                 negative_prompt, prompt_embeds, negative_prompt_embeds, lora_scale, clip_skip)
        

    def encode_image(self, image, device, num_images_per_prompt):
        return self.pipe.encode_image(image, device, num_images_per_prompt)

    def run_safety_checker(self, image, device, dtype):
        return self.pipe.run_safety_checker(image, device, dtype)

    def decode_latents(self, latents):
        return self.pipe.decode_latents(latents)

    def prepare_extra_step_kwargs(self, generator, eta):
        return self.pipe.prepare_extra_step_kwargs(generator, eta)


    def check_inputs(
        self,
        prompt,
        image,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ):
        self.pipe.check_inputs( prompt, image, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds, 
                                controlnet_conditioning_scale, control_guidance_start, control_guidance_end)

    def check_image(self, image, prompt, prompt_embeds):
        return self.pipe.check_image(image, prompt, prompt_embeds)

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        return self.pipe.prepare_image( image, width, height, batch_size, num_images_per_prompt, device, dtype, 
                                 do_classifier_free_guidance, guess_mode)
        
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        return self.pipe.prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, latents)


    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        self.pipe.enable_freeu(s1, s2, b1, b2)

    def disable_freeu(self):
        self.pipe.disable_freeu()

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        return self.pipe.get_guidance_scale_embedding(w, embedding_dim, dtype)


    @torch.no_grad()
    def get_random_background(self, n_samples):
        # sample random background with a constant rgb value
        backgrounds = torch.rand(n_samples, 3, device=self.device)[:, :, None, None].repeat(1, 1, 512, 512)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    @torch.no_grad()
    def decode_latents2(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs




def preprocess_mask(mask_path, h, w, device):
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    return mask


class DistDiffusion(nn.Module):
    def __init__(self, device, control, scene, model='2.0' , hf_key=None):
        super().__init__()

        self.device = device
        self.version = model

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            model_key =  get_model_path(self.version) #For custom models or fine-tunes, allow people to use arbitrary versions
            #raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        controlnets, self.mconds = get_controls(control, scene)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(model_key, controlnet=controlnets, safety_checker=None, torch_dtype=torch.float16)
        pipe = pipe.to(self.device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        self.pipe = pipe
        self.controlnets = controlnets


        print(f'[INFO] loaded stable diffusion!')
        rank = dist.get_rank()


        self.iter = 0
        self.rank = rank

    @torch.no_grad()
    def all2all(self, x, alpha, maintain_norm = False):
        srcs = []
        dsts = []

        if maintain_norm:
            norm = (x**2).sum(dim = (1,2,3), keepdim = True).sqrt()

        self.iter = self.iter + 1
        for r, map in enumerate(self.maps):

            if r == self.rank:
                srcs.append(torch.empty([], dtype = torch.float16 ,  device = self.device))
            else:  
                srcs.append(x[:,:,map[0],map[1]].clone())
            dsts.append(torch.empty_like(srcs[-1]))

        srcs_to_send = [src.clone().contiguous() for src in srcs]

        dist.all_to_all(dsts, srcs_to_send)

        for r, map in enumerate(self.maps):

            if r == self.rank:
                continue

            diff = (srcs[r] - dsts[r])/2
            x[:,:,map[0],map[1]] -= diff * alpha
            
            # if self.iter % 50 == 0:
            #     print(f'[INFO] rank {self.rank} received from rank {r}: ')
            #     print(f'{srcs[r][:,:,0:3].cpu().numpy()}')
            #     print(f'{dsts[r][:,:,0:3].cpu().numpy()}')
            #     print(f'{diff[:,:,0:3].cpu().numpy()}')
            #     print("-" * 50)

        if maintain_norm:
            x = x * norm / (x**2).sum(dim = (1,2,3), keepdim = True).sqrt()

        return x


    @torch.no_grad()
    def process_cond(self, conds, batch_size = 1, num_images_per_prompt = 1, do_classifier_free_guidance = False, guess_mode = False):

        controlnet = self.pipe.controlnet

        if isinstance(controlnet, ControlNetModel):
            width, height = conds.size
            image = self.prepare_image(
                image=conds,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=self.device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []
            
            for image_ in conds:
                width, height = image_.size

                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=self.device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        return image

    @torch.no_grad()
    def set_mappings(self, maps):
        if not isinstance(maps, list):
            maps = [maps]
        
        self.world_size = len(maps)
        self.maps = [map.clone() for map in maps]


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        pairing_strength: float = 0.1,
        max_pairing_steps: int = -1,
        display_every: int = -1,
        maintain_norm: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the controlnet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the controlnet stops applying.
            pairing_strength (`float`, *optional*, defaults to 0.1):
                The strength of the pairing between the two scenes.
            max_pairing_steps (`int`, *optional*, defaults to -1):
                The maximum number of steps for pairing. If -1, it will be set to `num_inference_steps//2`.
            display_every (`int`, *optional*, defaults to -1):
                The frequency at which the generated images will be displayed. If -1, it will be set to
                `num_inference_steps//10`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if max_pairing_steps < 0:
            max_pairing_steps = num_inference_steps//2

        controlnet = self.pipe.controlnet._orig_mod if is_compiled_module(self.pipe.controlnet) else self.pipe.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 1. Check inputs. Raise error if not correct.
        self.check_inputs(prompt, self.mconds, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds, controlnet_conditioning_scale, control_guidance_start, control_guidance_end)


        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipe._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare image

        self.cond_img = self.process_cond(self.mconds)
        image = self.cond_img

        height = image[0].shape[-2]
        width = image[0].shape[-1]

        # 5. Prepare timesteps

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        order = self.scheduler.order

        # 6. Prepare latent variables
        num_channels_latents = self.pipe.unet.config.in_channels


        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop

        num_warmup_steps = len(timesteps) - num_inference_steps * order

        ##force the latents to start at the same point

        for i in range(self.world_size-1):
            if self.rank <= i:
                latents = self.all2all(latents, 0.0, maintain_norm = False)
            else:
                latents = self.all2all(latents, 2.0, maintain_norm = True)


        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                k = self.rank

                image = self.cond_img
                scheduler = self.scheduler


                ##if latents has nan:
                if torch.isnan(latents).any():
                    import pdb; pdb.set_trace()


                ##import pdb; pdb.set_trace()

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    cond_scale = controlnet_conditioning_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )



                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]



                if pairing_strength > 0:
                    
                    if i < max_pairing_steps:
                        alpha = pairing_strength
                    else:
                        alpha = pairing_strength * (1 - (i - max_pairing_steps) / (num_inference_steps - max_pairing_steps))

                    latents = self.all2all(latents.clone(), alpha, maintain_norm = maintain_norm)

                    # update the latent

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % order == 0):
                    progress_bar.update()
                    # if callback is not None and i % callback_steps == 0:
                    #     callback(i, t, latents1)
                    #     callback(i, t, latents2)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self.pipe, "final_offload_hook") and self.pipe.final_offload_hook is not None:
            self.pipe.unet.to("cpu")
            self.pipe.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.pipe.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            ##_, has_nsfw_concept = self.pipe.run_safety_checker(image1, device, prompt_embeds.dtype)
            ##_, has_nsfw_concept = self.pipe.run_safety_checker(image2, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.pipe.final_offload_hook is not None:
            self.pipe.final_offload_hook.offload()

        if not return_dict:
            return (image[0], has_nsfw_concept)

        return image[0]
    

    def enable_vae_slicing(self):
        return self.vae.enable_slicing()

    def disable_vae_slicing(self):
        return self.vae.disable_slicing()

    def enable_vae_tiling(self):
        return self.vae.enable_tiling()

    def disable_vae_tiling(self):
        return self.vae.disable_tiling()

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        return self.pipe._encode_prompt( prompt, device, num_images_per_prompt, do_classifier_free_guidance, 
                                 negative_prompt, prompt_embeds, negative_prompt_embeds, lora_scale)

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        return self.pipe.encode_prompt( prompt, device, num_images_per_prompt, do_classifier_free_guidance, 
                                 negative_prompt, prompt_embeds, negative_prompt_embeds, lora_scale, clip_skip)
        

    def encode_image(self, image, device, num_images_per_prompt):
        return self.pipe.encode_image(image, device, num_images_per_prompt)

    def run_safety_checker(self, image, device, dtype):
        return self.pipe.run_safety_checker(image, device, dtype)

    def decode_latents(self, latents):
        return self.pipe.decode_latents(latents)

    def prepare_extra_step_kwargs(self, generator, eta):
        return self.pipe.prepare_extra_step_kwargs(generator, eta)


    def check_inputs(
        self,
        prompt,
        image,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ):
        self.pipe.check_inputs( prompt, image, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds, 
                                controlnet_conditioning_scale, control_guidance_start, control_guidance_end)

    def check_image(self, image, prompt, prompt_embeds):
        return self.pipe.check_image(image, prompt, prompt_embeds)

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        return self.pipe.prepare_image( image, width, height, batch_size, num_images_per_prompt, device, dtype, 
                                 do_classifier_free_guidance, guess_mode)
        
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        return self.pipe.prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, latents)


    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        self.pipe.enable_freeu(s1, s2, b1, b2)

    def disable_freeu(self):
        self.pipe.disable_freeu()

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        return self.pipe.get_guidance_scale_embedding(w, embedding_dim, dtype)


    @torch.no_grad()
    def get_random_background(self, n_samples):
        # sample random background with a constant rgb value
        backgrounds = torch.rand(n_samples, 3, device=self.device)[:, :, None, None].repeat(1, 1, 512, 512)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    @torch.no_grad()
    def decode_latents2(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs


if __name__ == '__main__':
    pass