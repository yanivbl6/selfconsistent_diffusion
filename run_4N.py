from engine import *
from diffusion import *

from matplotlib import pyplot as plt
import dist_controlled_multi
import argparse
import random
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# num_types = ["weight", "activate", "grad", "error", "momentum"]


num_types = ["weight", "activate", "error"]




parser.add_argument('--data', metavar='DIR', type=str,
                    help='path to dataset')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')


parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--resolution', default=512, type=int,
                    help='resolution')

parser.add_argument('-a', '--name', default='farm', type=str,
                    help='type of 3d enviorment', choices=["farm", "metro"])

parser.add_argument('-N', '--extra_num', default=30, type=int,
                    help='number of additional objects in 3d enviorment')

parser.add_argument('--add_skies', default=False, action='store_true',
                    help='add skies to 3d enviorment')

parser.add_argument('--model', default='realistic', type=str,
                    help='model name')
                    
parser.add_argument('--device', default='cuda:0', type=str,
                    help='device')

parser.add_argument('--num_inference_steps', default=800, type=int,
                    help='steps')

parser.add_argument('--pairing_strength', default=0.5, type=float,
                    help='pairing_strength')

parser.add_argument('--max_pairing_steps', default=-50, type=int,
                    help='max_pairing_steps')

parser.add_argument('--display_every', default=-1, type=int,
                    help='display_every')

parser.add_argument('--guidance_scale', default=9.0, type=float,
                    help='guidance_scale')

parser.add_argument('--norm_conditioning_scale', default=0.3, type=float,
                    help='norm_conditioning_scale')

parser.add_argument('--seg_conditioning_scale', default=1.0, type=float,
                    help='seg_conditioning_scale')

parser.add_argument('-k','--interpolate_k', default=2, type=int,
                    help='interpolate_k')

parser.add_argument('--show_maps', default=False, action='store_true',
                    help='output the mapped scenes')

parser.add_argument('--diff_seed', default=None, type=int,
                    help='diffusion seed')

parser.add_argument('--maintain_norm', default=False, action='store_true',
                    help='maintain norm during all to all-v')


parser.add_argument('--refine_steps', default=0, type=int,
                    help='refine_steps')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        ##cudnn.deterministic = True

    skretch, scenes = init_work(args)
    print("INFO: Scenes were initialized")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    ngpus_per_node = len(scenes)
    
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args, scenes, skretch))
    else:
        # Simply call main_worker function
        args.disributed = False
        main_worker(args.gpu, ngpus_per_node, args, scenes, skretch)


    return


def init_work(args):

    if args.name == "farm":
        skretch = gen_farm()
    elif args.name == "metro":
        skretch = gen_metropolis(args.extra_num)

    if args.add_skies:
        skretch = add_skies(skretch, 4, Z = 200)

    scenes = [0,1]

    resolution = args.resolution

    verbose = 0
    linewidth = 0
    pov = Point(0,0,12)
    direction = Point(1,1,-0.1)

    sc1 = Scene(skretch, pov, direction, resolution = (resolution,resolution), vision_angle = 35, verbose = verbose, linewidth = linewidth)


    pov = Point(0,10,40)
    direction = Point(1,0.9,-0.1)
    sc2 = Scene(skretch, pov, direction, resolution = (resolution,resolution), vision_angle = 35, verbose = verbose, linewidth = linewidth)

    scenes = [sc1, sc2]

    # sc1 = Scene(skretch, pov, direction, resolution = (resolution,resolution), vision_angle = 35, verbose = verbose, linewidth = linewidth)


    # pov = Point(0,10,40)
    # direction = Point(1,0.9,-0.1)
    # sc2 = Scene(skretch, pov, direction, resolution = (resolution,resolution), vision_angle = 35, verbose = verbose, linewidth = linewidth)

    # scenes = [sc1, sc2]

    return skretch, scenes


def setup_mapping(scenes, rank,device , show_maps, interpolate_k ):
    
    maps = []
    
    dst_maps = []
    scene = scenes[rank]
    ## copy maps:
    if rank == 0:
        pbar = tqdm(range(len(scenes)), desc="Calculating maps", total=len(scenes))
    else:
        pbar = range(len(scenes))

    for i in pbar:
        map, dst_map = dist_controlled_multi.get_mapping2(device  , scene,scenes[i], rank, i, interpolate_k)

        maps.append(map.clone())
        dst_maps.append(dst_map)

        if i != rank:
            if show_maps:
                display_map(scene, scenes[i], map, dst_map, num_types, img_name_ext = f"dist_pipe_{rank}{i}")

    return maps

def refine(image, refine_steps, prompt, nprompt, inference_steps):
    from diffusers import StableDiffusionXLImg2ImgPipeline

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    refiner = refiner.to("cuda")
    strength = refine_steps/inference_steps
    image = refiner(prompt, image=image, num_inference_steps = inference_steps, negative_prompt = nprompt, strength = strength).images[0]
    return image


def main_worker(gpu, ngpus_per_node, args, scenes, skretch):
    

    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=rank,
                                timeout=datetime.timedelta(minutes=1))

    scene = scenes[rank]


    model = args.model

    if args.norm_conditioning_scale > 0.0:
        control = ['seg','normal']
    else:
        control = ['seg']

    device = "cuda:{}".format(rank)

    print("[INFO] rank = ", rank, "device = ", device)

    if args.name == "farm":
        prompt = "a countryside farm. Bright skies. Epic realistic, (hdr:1.4), (muted colors:1.4), abandoned, (intricate details), (intricate details, hyperdetailed:1.4), artstation, vignette"
    elif args.name == "metro":
        prompt = "a cyberpunk city. Neon lighting, Dark skies. Epic realistic, (hdr:1.4), (muted colors:1.4), apocalypse, abandoned, screen space refractions, (intricate details), (intricate details, hyperdetailed:1.4), artstation, vignette"
    nprompt = "blurry, foggy"

    if args.max_pairing_steps < 0:
        args.max_pairing_steps = args.num_inference_steps + args.max_pairing_steps

    dist.barrier()

    if args.diff_seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.diff_seed)
    else:
        generator = None


    if 0:
        import controlled_multi
        import controlled_multi_sanity
        
        pipe = controlled_multi_sanity.MultiDiffusion(device, control, scenes[rank], model=model , hf_key=None)

        maps = setup_mapping(scenes, rank, device, args.show_maps, args.interpolate_k)
        pipe.set_mappings(maps)

        image = pipe(prompt = prompt, negative_prompt = nprompt, num_inference_steps = args.num_inference_steps, 
                            pairing_strength = args.pairing_strength, max_pairing_steps = args.max_pairing_steps , 
                            display_every= args.display_every, guidance_scale = args.guidance_scale,
                            controlnet_conditioning_scale = [args.seg_conditioning_scale, args.norm_conditioning_scale], generator = generator)

    else:

        ##pipe = dist_controlled_multi.DistMultiDiffusion(device, control, scene, model=model , hf_key=None)
        pipe = dist_controlled_multi.DistDiffusion(device, control, scene, model=model , hf_key=None)

        
        ##pipe.find_all_differences(stage = "init")

        maps = setup_mapping(scenes, rank, device, args.show_maps, args.interpolate_k)        
        pipe.set_mappings(maps)

        image = pipe(prompt = prompt, negative_prompt = nprompt, num_inference_steps = args.num_inference_steps, 
                            pairing_strength = args.pairing_strength, max_pairing_steps = args.max_pairing_steps , 
                            display_every= args.display_every, guidance_scale = args.guidance_scale, maintain_norm = args.maintain_norm,
                            controlnet_conditioning_scale = [args.seg_conditioning_scale, args.norm_conditioning_scale], generator = generator)

        ##pipe.find_all_differences(stage = "final")

    if args.refine_steps > 0:
        image = refine(image, args.refine_steps, prompt, nprompt, args.num_inference_steps)

    plt.close()
    image = np.asarray(image)
    plt.imshow(image)
    plt.savefig(f"images/image_{rank}.pdf")
    plt.close()

    image = torch.tensor(image).to(device)
    if rank == 0:
        results = [torch.empty_like(image) for _ in range(args.world_size)]

        dist.gather(image, results, dst=0)
        results = [res.cpu().numpy() for res in results]

        src = [[sc.get_seg_map(),sc.get_norm_map()] for sc in scenes] 

        present(src, results, img_name = f"prespective_change_with_diffusion")
        present([[sc.get_seg_map()] for sc in scenes] , results, img_name = f"prespective_change_with_diffusion", format = "pdf")

        print("[INFO] Done. Saved to images/prespective_change_with_diffusion.png")
    else:
        dist.gather(image, None, dst=0)
    
    dist.barrier()
    ##close dist
    dist.destroy_process_group()
    return
if __name__ == '__main__':
    main()