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


class MultiDiffusion(nn.Module):
    def __init__(self, device, control, scene1, scene2, model='2.0' , hf_key=None,src = None, dst = None, interpolate_k = 2):
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



        controlnets, self.mconds1 = get_controls(control, scene1)
        _, self.mconds2 = get_controls(control, scene2)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(model_key, controlnet=controlnets, safety_checker=None, torch_dtype=torch.float16)
        pipe = pipe.to(self.device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler1 = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        self.scheduler2 = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # self.scheduler1 = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        # self.scheduler2 = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")


        self.pipe = pipe
        self.controlnets = controlnets


        if src is None or dst is None:
            print(f'[INFO] pairing scenes')
            map1,map2 = scene1.connect(scene2, interpolate_k = interpolate_k)

            map1 = map1.astype(np.int)
            map2 = map2.astype(np.int)
            
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
                    
            map1 = np.asarray(new_map1).transpose()
            map2 = np.asarray(new_map2).transpose()

            self.map1 = torch.tensor(map1, dtype = torch.long).to(self.device)
            self.map2 = torch.tensor(map2, dtype = torch.long).to(self.device)
        else:
            print(f'[INFO] using provided pairing')
            self.map1 = torch.tensor(src.astype(np.int)//8, dtype = torch.long).to(self.device)
            self.map2 = torch.tensor(dst.astype(np.int)//8, dtype = torch.long).to(self.device)

        ##eliminate repeating indexes from self.map1:
        ##self.map1 = torch.unique(self.map1, dim = 1)


        # self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        # self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        # self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
        
        

        print(f'[INFO] loaded stable diffusion!')

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
        latents1: Optional[torch.FloatTensor] = None,
        latents2: Optional[torch.FloatTensor] = None,
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
        self.check_inputs(prompt, self.mconds1, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds, controlnet_conditioning_scale, control_guidance_start, control_guidance_end)
        self.check_inputs(prompt, self.mconds2, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds, controlnet_conditioning_scale, control_guidance_start, control_guidance_end)


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
        self.cond_img1 = self.process_cond(self.mconds1)
        self.cond_img2 = self.process_cond(self.mconds2)

        image = self.cond_img1

        height = self.cond_img1[0].shape[-2]
        width = self.cond_img1[0].shape[-1]

        # 5. Prepare timesteps
        self.scheduler1.set_timesteps(num_inference_steps, device=device)
        self.scheduler2.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler1.timesteps



        # 6. Prepare latent variables
        num_channels_latents = self.pipe.unet.config.in_channels
        latents1 = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents1,
        )

        latents2 = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents2,
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
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler1.order

        ##force the latents to start at the same point
        latents2[:,:,self.map2[0],self.map2[1]] = latents1[:,:,self.map1[0],self.map1[1]]

        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                k = i % 2
                for k in range(2):
                    if k == 0:
                        latents = latents1.clone()
                        image = self.cond_img1
                        scheduler = self.scheduler1
                    else:
                        latents = latents2.clone()
                        image = self.cond_img2
                        scheduler = self.scheduler2



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

                    if k == 0:
                        latents1 = latents.clone()
                    else:
                        latents2 = latents.clone()


                    del latents


                if pairing_strength > 0:
                    
                    if i < max_pairing_steps:
                        alpha = pairing_strength
                    else:
                        alpha = pairing_strength * (1 - (i - max_pairing_steps) / (num_inference_steps - max_pairing_steps))


                    diff = (latents1[:,:,self.map1[0],self.map1[1]] - latents2[:,:,self.map2[0],self.map2[1]])/2

                    # update the latent
                    latents1[:,:,self.map1[0],self.map1[1]] = latents1[:,:,self.map1[0],self.map1[1]] - diff * alpha
                    latents2[:,:,self.map2[0],self.map2[1]] = latents2[:,:,self.map2[0],self.map2[1]] + diff * alpha

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler1.order == 0):
                    progress_bar.update()
                    # if callback is not None and i % callback_steps == 0:
                    #     callback(i, t, latents1)
                    #     callback(i, t, latents2)

                if display_every > 0 and i % display_every == 0:

                    tmp_image1 = self.pipe.vae.decode(latents1 / self.vae.config.scaling_factor, return_dict=False)[0]
                    ##_, has_nsfw_concept = self.pipe.run_safety_checker(image1, device, prompt_embeds.dtype)
                    tmp_image2 = self.pipe.vae.decode(latents2 / self.vae.config.scaling_factor, return_dict=False)[0]
                    do_denormalize = [True] * tmp_image1.shape[0]
                    tmp_image1 = self.pipe.image_processor.postprocess(tmp_image1, output_type=output_type, do_denormalize=do_denormalize)
                    tmp_image2 = self.pipe.image_processor.postprocess(tmp_image2, output_type=output_type, do_denormalize=do_denormalize)

                    fig, ax= plt.subplots(1,2)
                                        
                    ax[0].imshow(tmp_image1[0])
                    ax[1].imshow(tmp_image2[0])


                    plt.show()

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self.pipe, "final_offload_hook") and self.pipe.final_offload_hook is not None:
            self.pipe.unet.to("cpu")
            self.pipe.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image1 = self.pipe.vae.decode(latents1 / self.vae.config.scaling_factor, return_dict=False)[0]
            ##_, has_nsfw_concept = self.pipe.run_safety_checker(image1, device, prompt_embeds.dtype)
            image2 = self.pipe.vae.decode(latents2 / self.vae.config.scaling_factor, return_dict=False)[0]
            ##_, has_nsfw_concept = self.pipe.run_safety_checker(image2, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            image1 = latents1
            image2 = latents2
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image1.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image1 = self.pipe.image_processor.postprocess(image1, output_type=output_type, do_denormalize=do_denormalize)
        image2 = self.pipe.image_processor.postprocess(image2, output_type=output_type, do_denormalize=do_denormalize)


        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.pipe.final_offload_hook is not None:
            self.pipe.final_offload_hook.offload()

        if not return_dict:
            return (image1[0], image2[0], has_nsfw_concept)

        return image1[0], image2[0]
    


    @torch.no_grad()
    def generate_old(self, masks, sc1, sc2, src, dst, alpha, prompts, negative_prompts='', 
                       num_inference_steps=50, guidance_scale=7.5):


        height = self.height
        width = self.width

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2 * len(prompts), 77, 768]

        # Define panorama grid and get views
        latent1 = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)
        latent2 = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)

        noise = latent1.clone().repeat(len(prompts) - 1, 1, 1, 1)

        self.scheduler.set_timesteps(num_inference_steps)

        h_start = 0
        h_end = height // 8
        w_start = 0
        w_end = width // 8

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):

                # predict the noise residual
                noise_pred = self.unet(latent1, t, encoder_hidden_states=text_embeds)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the denoising step with the reference model
                latents_view_denoised1 = self.scheduler.step(noise_pred, t, latent1)['prev_sample']
                # take the MultiDiffusion step

                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.

                # predict the noise residual
                noise_pred = self.unet(latent2, t, encoder_hidden_states=text_embeds)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the denoising step with the reference model
                latents_view_denoised2 = self.scheduler.step(noise_pred, t, latent2)['prev_sample']
                # take the MultiDiffusion step

                diff = latents_view_denoised2[dst] - latents_view_denoised1[src]

                # update the latent
                latent1[src] = latent1[src] + diff * alpha
                latent2[dst] = latent2[dst] - diff * alpha                

        # Img latents -> imgs
        img1 = self.decode_latents(latent1)  # [1, 3, 512, 512]
        img2 = self.decode_latents(latent2)  # [1, 3, 512, 512]

        img1 = T.ToPILImage()(img1[0].cpu())
        img2 = T.ToPILImage()(img2[1].cpu())
        return img1, img2

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_paths', type=list)
    # important: it is necessary that SD output high-quality images for the bg/fg prompts.
    parser.add_argument('--bg_prompt', type=str)
    parser.add_argument('--bg_negative', type=str)  # 'artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image'
    parser.add_argument('--fg_prompts', type=list)
    parser.add_argument('--fg_negative', type=list)  # 'artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image'
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'],
                        help="stable diffusion version")
    parser.add_argument('--H', type=int, default=768)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    # bootstrapping encourages high fidelity to tight masks, the value can be lowered is most cases
    parser.add_argument('--bootstrapping', type=int, default=20)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = MultiDiffusion(device, opt.sd_version)

    fg_masks = torch.cat([preprocess_mask(mask_path, opt.H // 8, opt.W // 8, device) for mask_path in opt.mask_paths])
    bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
    bg_mask[bg_mask < 0] = 0
    masks = torch.cat([bg_mask, fg_masks])

    prompts = [opt.bg_prompt] + opt.fg_prompts
    neg_prompts = [opt.bg_negative] + opt.fg_negative

    img = sd.generate(masks, prompts, neg_prompts, opt.H, opt.W, opt.steps, bootstrapping=opt.bootstrapping)

    # save image
    img.save('out.png')