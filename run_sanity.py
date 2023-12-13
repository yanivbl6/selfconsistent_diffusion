from engine import *
from diffusion import *

from matplotlib import pyplot as plt
import argparse
import random
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from tqdm import tqdm


import controlled_multi
import dist_controlled_multi

def main():
    skretch = gen_metropolis(30)

    skretch = add_skies(skretch, 4, Z = 200)

    verbose = 0
    linewidth = 0
    pov = Point(0,0,12)
    direction = Point(1,1,-0.1)
    vision_angle = 35

    sc1 = Scene(skretch, pov, direction, resolution = (512,512), vision_angle = 35, verbose = verbose, linewidth = linewidth)

    seg_map = sc1.get_seg_map()
    print(seg_map.shape)

    #plt.imshow(seg_map)

    pov = Point(0,10,40)
    direction = Point(1,0.9,-0.1)
    sc2 = Scene(skretch, pov, direction, resolution = (512,512), vision_angle = 35, verbose = verbose, linewidth = linewidth)
    ##seg_map = sc2.get_seg_map()
    # print(seg_map.shape)
    #plt.imshow(sc2.get_seg_map())

    device = "cuda"
    model = "realistic"
    control = ['seg','normal']


    pipe = controlled_multi.MultiDiffusion(device, control, sc1, sc2, model=model , hf_key=None, interpolate_k = 2)
    src_map = pipe.map1
    dst_map = pipe.map2

    # prompt = "a cyberpunk city. Neon lighting, Dark skies. Epic realistic, (hdr:1.4), (muted colors:1.4), apocalypse, abandoned, screen space refractions, (intricate details), (intricate details, hyperdetailed:1.4), artstation, vignette"
    # nprompt = "blurry, foggy"
    # image1, image2 = pipe(prompt = prompt, negative_prompt = nprompt, 
    #                     num_inference_steps = 200, pairing_strength = 0.8, max_pairing_steps = 200, display_every= -1, guidance_scale = 9.0,
    #                     controlnet_conditioning_scale = [1.0, 0.4])


    display_map(sc1, sc2, src_map, dst_map, control, img_name_ext = "one_pipe_01")
    display_map(sc2, sc1, dst_map, src_map, control, img_name_ext = "one_pipe_10")


    src_map, dst_map = dist_controlled_multi.get_mapping(device  ,sc1,sc2, 0, 1, 2)

    dst_map2, src_map2 = dist_controlled_multi.get_mapping(device  ,sc2,sc1, 1, 0, 2)

    assert(torch.all(src_map == src_map2))
    assert(torch.all(dst_map == dst_map2))


    display_map(sc1, sc2, src_map, dst_map, control, img_name_ext = "dist_pipe_01")
    display_map(sc2, sc1, dst_map, src_map, control, img_name_ext = "dist_pipe_10")



if __name__ == '__main__':
    main()