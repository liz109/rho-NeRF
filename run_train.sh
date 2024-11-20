#!/bin/bash

# ============= Run  =============
python train_mlg.py --config ./config/Lineformer/jaw_50.yaml --gpu_id 1 
# python train_mlg.py --config ./config/Lineformer/head_50.yaml --gpu_id 1 



# python train.py --config ./config/naf/chest_50.yaml
# python train.py --config ./config/naf/aneurism_50.yaml 
# python train.py --config ./config/naf/leg_50.yaml # re-train
# python train.py --config ./config/naf/jaw_50.yaml
# python train.py --config ./config/naf/abdomen_50.yaml
# python train.py --config ./config/naf/backpack_50.yaml
# python train.py --config ./config/naf/bonsai_50.yaml
# python train.py --config ./config/naf/box_50.yaml

# python train.py --config ./config/naf/carp_50.yaml
# python train.py --config ./config/naf/engine_50.yaml  # re-train
# python train.py --config ./config/naf/foot_50.yaml
# python train.py --config ./config/naf/head_50.yaml
# # python train.py --config ./config/naf/pancreas_50.yaml
# python train.py --config ./config/naf/teapot_50.yaml 
# python train.py --config ./config/naf/pelvis_50.yaml 
# python train.py --config ./config/naf/bonsai_50.yaml 
# python train.py --config ./config/naf/box_50.yaml --gpu_id 0



