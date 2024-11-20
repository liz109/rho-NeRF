import sys
ROOT = '/home/lzhou/SAX-NeRF-rho'
sys.path.append(ROOT)

import os
import os.path as osp
import argparse
import torch
import pickle
import time
import numpy as np
import tigre
import tigre.algorithms as algs

from src.config.configloading import load_config
# from src.dataset import TIGREDataset_Traditional as Dataset_Traditional
from src.dataset.tigre_traditional import ConeGeometry
from src.utils import get_psnr_3d, get_ssim_3d, cast_to_image

torch.cuda.empty_cache()
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

## ============= Target: add "image_init" to ../data/ =============

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/FDK/chest_50.yaml",
                        help="configs file path")
    # parser.add_argument("--category", default="chest",
    #                     help="the category of the tested scene")
    # parser.add_argument("--output_path", default=f"output", 
    #                     help="path to the output folder")
    # parser.add_argument("--gpu_id", default="0", help="gpu to use")
    return parser


def initial_image(projs, geo, angles, algorithm="fdk"):
    # change the voxel setting in geometry to align with our data
    geo.nVoxel = np.flip(geo.nVoxel)
    geo.sVoxel = np.flip(geo.sVoxel)
    geo.dVoxel = np.flip(geo.dVoxel)

    lmbda = 1
    lambdared = 0.999
    initmode = None
    verbose = True
    qualmeas = ["RMSE", "SSD"]
    blcks = 20
    order = "random"


    if algorithm == "fdk":      # Filtered backprojection (FBP,FDK) and variations
        image_init = algs.fdk(projs, 
                            geo, 
                            angles, 
                            filter="ram_lak",
                            )  
    # iterative algorithms
    elif algorithm == "sart":   # Gradient-based algorithms -- SART family 
        image_init = algs.sart(projs,
                            geo, 
                            angles,
                            niter=5,    
                            lmbda=lmbda,
                            lmbda_red=lambdared,
                            init=initmode,
                            verbose=verbose)
    elif algorithm == "sirt":   # Gradient-based algorithms -- SART family 
        image_init = algs.sart(projs,
                            geo, 
                            angles,
                            niter=5,    
                            lmbda=lmbda,
                            lmbda_red=lambdared,
                            init=initmode,
                            verbose=verbose)  
    elif algorithm == "asd_pocs":  # Gradient-based algorithms -- total Variation minimization algorithms
        image_init = algs.asd_pocs(projs, 
                            geo, 
                            angles,
                            niter=5,    
                            lmbda=lmbda,
                            lmbda_red=lambdared,
                            init=initmode,
                            verbose=verbose)
    elif  algorithm == "ossart": 
        image_init, _ = algs.ossart(
                            projs,
                            geo,
                            angles,
                            5,
                            lmbda=lmbda,
                            lmbda_red=lambdared,
                            verbose=verbose,
                            Quameasopts=qualmeas,
                            blocksize=blcks,
                            OrderStrategy=order,
                            computel2=True,
                        )
    elif algorithm == "cgls":   # krylov subspace family
        image_init, normL2CGLS = algs.cgls(projs, 
                            geo,      
                            angles,
                            5, computel2=True)
    else:
        raise NotImplementedError
    
    image_init = np.flip(image_init.transpose(2,1,0), axis=2)
    return image_init


class Image_Init():
    def __init__(self, cfg):
        super().__init__()
        print(f"\n\n[Start] exp: {cfg['exp']['expname']}")

        self.datadir = os.path.abspath(osp.join(ROOT, cfg["exp"]["datadir"]))
        with open(self.datadir, "rb") as handle:
            self.data = pickle.load(handle)


    def add_init(self, algorithms):
        # train_dset = Dataset_Traditional(self.datadir, cfg["train"]["n_rays"], "train", self.device) # 由dataset去构造数据集
        # geo = ConeGeometry(self.data)
        # image = self.data['image']
        # projs = self.data['train']['projections']
        # angles = self.data['train']['angles']

        if "image_init" not in self.data:
            self.data["image_init"] = dict()

        for algo in algorithms:
            print(f"*[Algo]: {algo}")
            start_time = time.time()

            geo = ConeGeometry(self.data)
            image = self.data['image']
            projs = self.data['train']['projections']
            angles = self.data['train']['angles']

            img_init = initial_image(projs, geo, angles, algorithm=algo)
            self.data["image_init"][algo] = img_init.copy()

            end_time = time.time()
            CT_reconstruct_time = end_time - start_time
        
            loss = {
                "psnr_3d": get_psnr_3d(image, img_init),
                "ssim_3d": get_ssim_3d(image, img_init),
                "CT_reconstruct_time": CT_reconstruct_time
                }
            print(loss)


        with open(self.datadir, "wb") as handle:
            pickle.dump(self.data, handle)
        print(f"Update {cfg['exp']['expname']} successfully!!!")



if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)




    ROOT = '/home/lzhou/SAX-NeRF-rho'
    # algorithms = ['fdk', 'sart', 'sirt', 'asd_pocs', 'cgls']
    algorithms = ['asd_pocs']

    start = time.time()
    add_image_init = Image_Init(cfg)
    add_image_init.add_init(algorithms)
    end = time.time()
    minutes, seconds = divmod(end-start, 60)
    print(f"Elapsed time: {int(minutes)}m {seconds:.2f}s")
