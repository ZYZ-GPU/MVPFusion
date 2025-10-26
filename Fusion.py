import os
import sys
import glob
import time

import cv2
import torch
import numpy as np

from tqdm import tqdm
from torch import einsum
from Nets.MVPFusion import MVPFusion
from Utilities import Consistency
import Utilities.DataLoaderFM as DLr
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from Utilities.CUDA_Check import GPUorCPU
from Utilities.GuidedFiltering import guided_filter
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode




class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class Fusion:
    def __init__(self,
                 # modelpath='debug_model.ckpt',
                 modelpath='RunTimeData/loss_91_8000_112/model17.ckpt',
                 dataroot='./Datasets/Eval',
                 dataset_name='Lytro',
                 threshold=0.01,
                 resize_size=112
                 ):
        self.DEVICE = GPUorCPU().DEVICE
        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAME = dataset_name
        self.THRESHOLD = threshold
        self.RESIZE = resize_size

    def __call__(self, *args, **kwargs):
        if self.DATASET_NAME != None:
            self.SAVEPATH = '/' + self.DATASET_NAME
            self.DATAPATH = self.DATAROOT + '/' + self.DATASET_NAME
            MODEL = self.LoadWeights(self.MODELPATH)
            EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
            self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD)
        else:
            print("Test Dataset required!")
            pass

    def LoadWeights(self, modelpath):
        model = MVPFusion().to(self.DEVICE)
        model.load_state_dict(torch.load(modelpath))
        model.eval()
        # num_params = 0
        # for p in model.parameters():
        #     num_params += p.numel()
        # # print(model)
        # print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
        # from thop import profile, clever_format
        # flops, params = profile(model, inputs=(torch.rand(1, 3, 520, 520).cuda(), torch.rand(1, 3, 520, 520).cuda()))
        # flops, params = clever_format([flops, params], "%.5f")
        # print('flops: {}, params: {}\n'.format(flops, params))
        return model

    def PrepareData(self, datapath):
        eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
        eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
        return eval_list_A, eval_list_B

    def ConsisVerif(self, img_tensor, threshold):
        Verified_img_tensor = Consistency.Binarization(img_tensor)
        if threshold != 0:
            Verified_img_tensor = Consistency.RemoveSmallArea(img_tensor=Verified_img_tensor, threshold=threshold)
        return Verified_img_tensor

    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
        if not os.path.exists('./Results' + savepath):
            os.mkdir('./Results' + savepath)
        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data,
                                 batch_size=1,
                                 shuffle=False, )
        eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)
        cnt = 1
        running_time = []
        with torch.no_grad():
            for A, B in eval_loader_tqdm:
                _, _, H, W = A.shape
                A = F.interpolate(A.to(self.DEVICE), size=(self.RESIZE, self.RESIZE), mode='bilinear',align_corners=False)
                B = F.interpolate(B.to(self.DEVICE), size=(self.RESIZE, self.RESIZE), mode='bilinear',align_corners=False)
                start_time = time.time()

                D, _, _, _, _ = model(A, B)
                D = F.interpolate(D, size=(H, W), mode='bilinear', align_corners=False)
                D_raw = torch.where(D > 0.5, 1., 0.)
                D = self.ConsisVerif(D_raw, threshold)

                d_verified_save_path = './Results' + savepath + '/D_verified'
                os.makedirs(d_verified_save_path, exist_ok=True)

                # 保存 D_verified（二值图像）
                D_verified_np = (D[0][0].clone().detach().cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(d_verified_save_path, f'{self.DATASET_NAME}-{str(cnt).zfill(2)}.png'),
                            D_verified_np)  # <<< 新增

                D = einsum('c w h -> w h c', D[0]).clone().detach().cpu().numpy()
                A = cv2.imread(eval_list_A[cnt - 1])
                B = cv2.imread(eval_list_B[cnt - 1])
                IniF = A * D + B * (1 - D)
                D_GF = guided_filter(IniF, D, 4, 0.1)
                Final_fused = A * D_GF + B * (1 - D_GF)
                cv2.imwrite('./Results' + savepath + '/' + self.DATASET_NAME + '-' + str(cnt).zfill(2) + '.png', Final_fused)
                cnt += 1
                # print("process_time: {} s".format(time.time() - start_time))
                running_time.append(time.time() - start_time)
        running_time_total = 0
        for i in range(len(running_time)):
            print("process_time: {} s".format(running_time[i]))
            if i != 0:
                running_time_total += running_time[i]
        print("\navg_process_time: {} s".format(running_time_total / (len(running_time) - 1)))
        print("\nResults are saved in: " + "./Results" + savepath)


if __name__ == '__main__':
    f = Fusion()
    f()
