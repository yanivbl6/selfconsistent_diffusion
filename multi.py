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
    def __init__(self, device, control, scene1, scene2, model='2.0' , hf_key=None):
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
        _, self.mconds2 = get_controls(control, scene1)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(model_key, controlnet=controlnets, safety_checker=None, torch_dtype=torch.float16)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe



        
        self.cond_img1 = self.process_cond(pipe, self.mconds1, controlnets, device)

        self.height, self.width = self.cond_img1[0].shapes[-2:]

        self.cond_img2 = self.process_cond(pipe, self.mconds2, controlnets, device)

        # self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        # self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        # self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
        # self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def process_cond(self, pipe, conds, controlnets, device):
        images = []
        for i in range(len(conds)):
            cond = conds[i]
            cnet = controlnets[i]
            width = cond.shape[-1]
            height = cond.shape[-2]

            image_ = pipe.prepare_image(
                image= cond,
                width=width,
                height=height,
                batch_size= 1,
                num_images_per_prompt=1,
                device=device,
                dtype=cnet.dtype,
                do_classifier_free_guidance=False,
                guess_mode=False,
            )

            images.append(image_)

        return images

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
        **kwargs,
    ):
        return self.pipe._encode_prompt( prompt, device, num_images_per_prompt, do_classifier_free_guidance, 
                                 negative_prompt, prompt_embeds, negative_prompt_embeds, lora_scale, **kwargs)

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
        callback_on_step_end_tensor_inputs=None,
    ):
        return self.pipe.check_inputs( prompt, image, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds, 
                                controlnet_conditioning_scale, control_guidance_start, control_guidance_end, 
                                callback_on_step_end_tensor_inputs)

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
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs






    @torch.no_grad()
    def generate(self, masks, sc1, sc2, src, dst, alpha, prompts, negative_prompts='', 
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