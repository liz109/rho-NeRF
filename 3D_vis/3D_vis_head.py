# export LD_LIBRARY_PATH=/home/ycai51/anaconda3/envs/naf/lib
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import pickle

# from src.dataset import TIGREDataset as Dataset

from pdb import set_trace as stx
import argparse

start = time.time()

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="0", help="gpu to use")
    return parser

parser = config_parser()
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


category = 'head'
save_dir = category + '/'

path = f'../data/{category}_50.pickle'

with open(path, "rb") as handle:
            data = pickle.load(handle)

'''
    采用 np.rot90 和 np.flip 实现旋转和翻转

    B0 = np.rot90(A, 1, (0,1)) #绕 z 轴旋转90度
    B1 = np.rot90(A,-1, (0,1)) #绕 z 轴旋转270度
    B2 = np.rot90(A, 1, (1,2)) #绕 x 轴旋转90度
    B3 = np.rot90(A,-1, (1,2)) #绕 x 轴旋转270度
    B4 = np.rot90(A, 1, (0,2)) #绕 y 轴旋转90度
    B5 = np.rot90(A,-1, (0,2)) #绕 y 轴旋转270度

    B6 = np.flip(A, 2) #z轴翻折
    B7 = np.flip(A, 1) #y轴翻折
    B8 = np.flip(A, 0) #x轴翻折
'''

CT_image = data["image"]
'''
    沿着z轴反转
    沿着y轴顺时针90度
    沿着z轴顺时针90度
'''
print(CT_image.shape)
# CT_image = CT_image[...,::-1]
# CT_image = np.flip(CT_image,2)
CT_image = np.rot90(CT_image, -1, (0,2))
# CT_image = np.rot90(CT_image, -0.1, (1,2))
# CT_image = np.rot90(CT_image[::-1,:,::-1], -1, (0,1))
# CT_image = CT_image[:,::-1,::-1]
# CT_image = CT_image[...,::-1]
# CT_image = CT_image[::-1,...]
# print(CT_image.shape)

# stx()
min_value = CT_image.min()
max_value = CT_image.max()
sigma = 0.6
threshold = sigma * min_value + (1 - sigma) * max_value
verts, faces, _, _ = measure.marching_cubes(CT_image, threshold)

# stx()

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
alpha = 0.30
mesh = Poly3DCollection(verts[faces], alpha=alpha)
face_color = [0.5, 0.5, 0.5]
mesh.set_facecolor(face_color)
ax.add_collection3d(mesh)

ax.set_xlim(0, CT_image.shape[0])
ax.set_ylim(0, CT_image.shape[1])
ax.set_zlim(0, CT_image.shape[2])

# ax.set_xlim(0, 256)
# ax.set_ylim(0, 256)
# ax.set_zlim(0, 256)

alpha_axis = 0.01
ax.set_alpha(alpha_axis)

proj_num = 4
angle_interval = 360 / proj_num
elevation = 10

series_save_dir = os.path.join(save_dir,f"elevation_{elevation}_sigma_{sigma}_alpha_{alpha}/")
# series_save_dir = os.path.join(save_dir,f"elevation_{elevation}_sigma_{sigma}_alpha_{alpha}_axisoff/")
os.makedirs(series_save_dir, exist_ok=True)

img_files = []

for i in tqdm(range(proj_num)):
    angle = angle_interval * i
    ax.view_init(elev=elevation, azim=angle)
    # ax.axis("off")
    # plt.savefig(f'{series_save_dir}angle_{angle}_axisoff.png')
    plt.savefig(f'{series_save_dir}angle_{angle}.png')
    img_files.append(f'{series_save_dir}angle_{angle}.png')

print(f"Rendering used time: {time.time()-start} s")