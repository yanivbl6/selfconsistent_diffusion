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

parser.add_argument('-k','--interpolate_k', default=2, type=int,
                    help='interpolate_k')

parser.add_argument('--show_maps', default=False, action='store_true',
                    help='output the mapped scenes')

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
        main_worker(args.gpu, ngpus_per_node, args, scenes, skretch)


    return


def init_work(args):

    if args.name == "farm":
        skretch = gen_farm()
    elif args.name == "metro":
        skretch = gen_metropolis(args.extra_num)

    if args.add_skies:
        skretch = add_skies(skretch, 4, Z = 200)

    resolution = args.resolution

    verbose = 0
    linewidth = 0
    pov = Point(0,0,12)
    direction = Point(1,1,-0.1)

    scenes = [0,0]

    # sc1 = Scene(skretch, pov, direction, resolution = (resolution,resolution), vision_angle = 35, verbose = verbose, linewidth = linewidth)


    # pov = Point(0,10,40)
    # direction = Point(1,0.9,-0.1)
    # sc2 = Scene(skretch, pov, direction, resolution = (resolution,resolution), vision_angle = 35, verbose = verbose, linewidth = linewidth)

    # scenes = [sc1, sc2]

    return skretch, scenes


def setup_mapping(scenes, rank, args, device):
    
    maps = []
    
    dst_maps = []
    scene = scenes[rank]
    ## copy maps:
    if rank == 0:
        pbar = tqdm(range(len(scenes)), desc="Calculating maps", total=len(scenes))
    else:
        pbar = range(len(scenes))

    results = []
    for i in pbar:
        map, dst_map = dist_controlled_multi.get_mapping(device  , scene,scenes[i], rank, i, args.interpolate_k)

        maps.append(map)
        dst_maps.append(dst_map)

        if args.show_maps:
            if map.numel() == 0:
                results.append(np.zeros_like(scenes[i].get_seg_map()))
                continue

            def expand_map(map):
                map1 = map[0].cpu().numpy()*8
                map2 = map[1].cpu().numpy()*8

                ##expand to 8x8 map
                map1 = np.repeat(map1, 64, axis=0)
                map2 = np.repeat(map2, 64, axis=0)

                map1_add = (np.arange(map1.shape[0]) // 8) % 8
                map2_add = ((np.arange(map2.shape[0]) // 1 ) % 8)

                map1 = map1 + map1_add
                map2 = map2 + map2_add

                ##break if any element is out of bound
                if np.any(map1 >= scenes[i].get_seg_map().shape[0]) or np.any(map2 >= scenes[i].get_seg_map().shape[1]):
                    print("ERROR: map out of bound: ", map1, map2)

                return map1, map2

            map1, map2 = expand_map(map)
            dmap1, dmap2 = expand_map(dst_map)

            img = scenes[i].get_seg_map().copy()
            mapped_img = np.zeros_like(img)
            mapped_img[dmap1, dmap2,:] = img[map1, map2,:]
            
            results.append(mapped_img)


    if args.show_maps:
        src = [[sc.get_seg_map(),sc.get_norm_map()] for sc in scenes] 
        present(src,  results, img_name = f"maps_rank_{rank}")
        print("Maps stored")
        dist.barrier()
        results = []

    return maps

def main_worker(gpu, ngpus_per_node, args, scenes, skretch):
    
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
    control = ['seg','normal']

    device = "cuda:{}".format(rank)

    print("rank = ", rank, "device = ", device)

    import controlled_multi
    pipe2 = controlled_multi.MultiDiffusion(device, control, sc1, sc2, model=model , hf_key=None, interpolate_k = 2)


    if rank == 0:
        maps = [   torch.tensor([],device = device)  , pipe2.map1.clone()]
    else:  
        maps = [pipe2.map2.clone() , torch.tensor([],device = device)  ]

    del pipe2
    ##maps = setup_mapping(scenes, rank, args, device)


    pipe = dist_controlled_multi.DistMultiDiffusion(device, control, scene, model=model , hf_key=None)

    pipe.set_mappings(maps)



    if args.name == "farm":
        prompt = "a countryside farm. Bright skies. Epic realistic, (hdr:1.4), (muted colors:1.4), abandoned, (intricate details), (intricate details, hyperdetailed:1.4), artstation, vignette"
    elif args.name == "metro":
        prompt = "a cyberpunk city. Neon lighting, Dark skies. Epic realistic, (hdr:1.4), (muted colors:1.4), apocalypse, abandoned, screen space refractions, (intricate details), (intricate details, hyperdetailed:1.4), artstation, vignette"
    nprompt = "blurry, foggy"

    if args.max_pairing_steps < 0:
        args.max_pairing_steps = args.num_inference_steps + args.max_pairing_steps

    dist.barrier()


    image = pipe(prompt = prompt, negative_prompt = nprompt, num_inference_steps = args.num_inference_steps, 
                        pairing_strength = args.pairing_strength, max_pairing_steps = args.max_pairing_steps , 
                        display_every= args.display_every, guidance_scale = args.guidance_scale,
                        controlnet_conditioning_scale = [1.0, args.norm_conditioning_scale])


    gather_vec = None
    image = np.asarray(image)
    image = torch.tensor(image).to(device)
    if rank == 0:
        results = [torch.empty_like(image) for _ in range(args.world_size)]

        dist.gather(image, results, dst=0)
        results = [res.cpu().numpy() for res in results]

        src = [[sc.get_seg_map(),sc.get_norm_map()] for sc in scenes] 


        present(src, results, img_name = f"prespective_change_with_diffusion")
        print("Done")
    else:
        dist.gather(image, None, dst=0)
    
    dist.barrier()
    ##close dist
    dist.destroy_process_group()
    return
if __name__ == '__main__':
    main()