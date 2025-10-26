import os
import glob
import sys
import datetime

import torch
from torch import nn
from tqdm import tqdm
from Nets.MVPFusion import MVPFusion
import Utilities.DataLoaderFM as DLr
from torch.utils.data import DataLoader
from Utilities.CUDA_Check import GPUorCPU
from Utilities.Logging_SaveModel import Logging_SaveModel
import torch.nn.functional as F

class NetTrain:
    def __init__(self,
                 data_path='./Datasets/Train&Valid/DUTS_medium',
                 set_size=8000,
                 batchsize=1,
                 epochs=200,
                 lr=0.0002,
                 gamma=0.88,
                 scheduler_step=1,
                 lmd=0.,
                 patience=6):
        # Hyper parameters
        self.DEVICE = GPUorCPU().DEVICE
        self.DATAPATH = data_path
        self.SETSIEZE = set_size
        self.BATCHSIZE = batchsize
        self.EPOCHS = epochs
        self.LR = lr
        self.GAMMA = gamma
        self.SCHEDULER_STEP = scheduler_step
        self.LMD = lmd
        self.PATIENCE = patience
        # Form parameter dictionary
        self.hyperparas = {'set_size': self.SETSIEZE, 'batchsize': self.BATCHSIZE,
                           'epochs': self.EPOCHS, 'lr': self.LR, 'gamma': self.GAMMA,
                           'scheduler_step': self.SCHEDULER_STEP, 'lmd': self.LMD,
                           'patience': self.PATIENCE}

    def __call__(self, *args, **kwargs):
        TRAIN_LOADER, VALID_LOADER = self.PrepareDataLoader(self.DATAPATH, self.SETSIEZE, self.BATCHSIZE)
        MODEL, OPTIMIZER, SCHEDULER = self.BuildModel(self.DEVICE, self.LR, self.SCHEDULER_STEP, self.GAMMA)
        self.TrainingProcess(MODEL, OPTIMIZER, SCHEDULER, TRAIN_LOADER, VALID_LOADER, self.EPOCHS, self.LMD)

    def PrepareDataLoader(self, datapath, setsize, batchsize):
        train_list_A = sorted(glob.glob(os.path.join(datapath, 'train/sourceA', '*.*')))[:setsize]
        train_list_B = sorted(glob.glob(os.path.join(datapath, 'train/sourceB', '*.*')))[:setsize]
        train_list_DM = sorted(glob.glob(os.path.join(datapath, 'train/decisionmap', '*.*')))[:setsize]
        valid_list_A = sorted(glob.glob(os.path.join(datapath, 'validate/sourceA', '*.*')))[:setsize // 9]
        valid_list_B = sorted(glob.glob(os.path.join(datapath, 'validate/sourceB', '*.*')))[:setsize // 9]
        valid_list_DM = sorted(glob.glob(os.path.join(datapath, 'validate/decisionmap', '*.*')))[:setsize // 9]
        tqdm.write(f"Train Data A: {len(train_list_A)}")
        tqdm.write(f"Train Data B: {len(train_list_B)}")
        tqdm.write(f"Valid Data A: {len(valid_list_A)}")
        tqdm.write(f"Valid Data B: {len(valid_list_B)}")
        train_data = DLr.DataLoader_Train(train_list_A, train_list_B, train_list_DM)
        valid_data = DLr.DataLoader_Train(valid_list_A, valid_list_B, valid_list_DM)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=False)
        valid_loader = DataLoader(dataset=valid_data,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=False)
        tqdm.write(f"Train Data Size:{len(train_data)} , Train Loader Amount: {len(train_data)}/{batchsize} = {len(train_loader)}")
        tqdm.write(f"Valid Data Size:{len(valid_data)} , Valid Loader Amount: {len(valid_data)}/{batchsize} = {len(valid_loader)}\n")
        return train_loader, valid_loader

    def BuildModel(self, device, lr, scheduler_step, gamma):
        model = MVPFusion().to(device)
        # torch.compile(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step, gamma=gamma)
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
        return model, optimizer, scheduler


    def LossFunc(self, mask, pred):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()



    def TrainingProcess(self, model, optimizer, scheduler, train_loader, valid_loader, epochs, lmd):
        scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True
        LS = Logging_SaveModel(savepath='RunTimeData', hyperparas=self.hyperparas)

        tqdm.write('Training start...\n')

        for epoch in range(epochs):
            ######################################### Train #########################################
            epoch_loss = 0
            epoch_accuracy = 0
            epoch_loss1 = 0
            epoch_loss2 = 0
            epoch_loss3 = 0
            train_loader_tqdm = tqdm(train_loader, colour='green', leave=False, file=sys.stdout)
            for A, B, DM in train_loader_tqdm:
                optimizer.zero_grad()
                # Automatic mixed precision training.
                with torch.autocast(device_type=self.DEVICE, dtype=torch.float16):
                    final_output,sideout2,glb2,sideout1,glb1 = model(A, B)
                    b, c, h, w = DM.size()
                    target_1 = F.upsample(DM, size=h // 2, mode='nearest').cuda()
                    loss1 = self.LossFunc(DM, final_output)
                    loss2 = self.LossFunc(DM, sideout1)+self.LossFunc(DM, sideout2)
                    loss3 = self.LossFunc(target_1, glb1)+self.LossFunc(target_1, glb2)
                    loss = 0.9 * loss1 + 0.1 * (loss2 + loss3)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ''''''''''''''''''''''''''' Epoch loss calculation & progress visualization '''''''''''''''''''''''''''
                epoch_loss += loss / len(train_loader)
                epoch_accuracy += (1 - loss) / len(train_loader)
                epoch_loss1 += loss1 / len(train_loader)
                epoch_loss2 += loss2 / len(train_loader)
                epoch_loss3 += loss3 / len(train_loader)
                train_loader_tqdm.set_description("[%s] Epoch %s" % (str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')), str(epoch + 1)))
                train_loader_tqdm.set_postfix(loss1=float(loss1), loss2=float(loss2), loss3=float(loss3))
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            #########################################################################################
            ######################################### Valid #########################################
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                epoch_val_loss1 = 0
                epoch_val_loss2 = 0
                epoch_val_loss3 = 0
                valid_loader_tqdm = tqdm(valid_loader, colour='yellow', leave=False, file=sys.stdout)
                for A, B, DM in valid_loader_tqdm:
                    final_output, sideout2, glb2, sideout1, glb1 = model(A, B)
                    b, c, h, w = DM.size()
                    target_1 = F.upsample(DM, size=h // 2, mode='nearest').cuda()
                    loss1 = self.LossFunc(DM, final_output)
                    loss2 = self.LossFunc(DM, sideout1) + self.LossFunc(DM, sideout2)
                    loss3 = self.LossFunc(target_1, glb1) + self.LossFunc(target_1, glb2)
                    loss = 0.9 * loss1 + 0.1 * (loss2 + loss3)
                    ''''''''''''''''''''' Epoch loss calculation & progress visualization '''''''''''''''''''''
                    epoch_val_loss += loss / len(valid_loader)
                    epoch_val_accuracy += (1 - loss) / len(valid_loader)
                    epoch_val_loss1 += loss1 / len(train_loader)
                    epoch_val_loss2 += loss2 / len(train_loader)
                    epoch_val_loss3 += loss3 / len(train_loader)
                    valid_loader_tqdm.set_description("[Validating...] Epoch %s" % str(epoch + 1))
                    valid_loader_tqdm.set_postfix(loss1=float(loss1), loss2=float(loss2), loss3=float(loss3))
                    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            #########################################################################################
            # Print epoch loss and accuracy.
            tqdm.write(f"[{str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S'))}] Epoch {epoch + 1} - l : {epoch_loss:.4f} - a: {epoch_accuracy:.4f} - vl : {epoch_val_loss:.4f} - va: {epoch_val_accuracy:.4f} - l1: {epoch_loss1:.4f} - l2: {epoch_loss2:.4f} - l3: {epoch_loss3:.4f} - vl1: {epoch_val_loss1:.4f} - vl2: {epoch_val_loss2:.4f} - vl3: {epoch_val_loss3:.4f}")
            # Dynamic learning rate.
            scheduler.step()
            # Logging and Save model weights.
            log_contents = f"Epoch {epoch + 1} - loss: {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss: {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} - loss_1: {epoch_loss1:.4f} - loss_2: {epoch_loss2:.4f} - loss_3: {epoch_loss3:.4f} - vloss_1: {epoch_val_loss1:.4f} - vloss_2: {epoch_val_loss2:.4f} - vloss_3: {epoch_val_loss3:.4f}\n"
            LS(model, epoch + 1, log_contents, epoch_val_loss, save_every_model=True)
            if LS.ENDTRAIN:
                # Early stopping mechanism has been triggered.
                print("Early stopping!!!")
                # End training.
                break
        # Epoch loop ends.



if __name__ == '__main__':
    t = NetTrain()
    t()
