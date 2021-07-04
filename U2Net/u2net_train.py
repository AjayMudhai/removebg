import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model.u2net import U2NET
# from azure.storage.blob import BlockBlobService

from u2net_test import normPRED, save_output 
import zipfile


# from upload_saved_model import AZStorage
# from model.u2net_24million import U2NETP
# from dotenv import load_dotenv

# import boto3
import os
import random

torch.cuda.empty_cache()
print(torch.cuda.is_available())
# az=AZStorage()  
# s3 = boto3.resource('s3', aws_access_key_id='AKIAZDYZULCK6G7ECRRR',
#                aws_secret_access_key='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')

# bucketname = 'zapbg-training-datasets'
# print(bucketname)


# ------- 1. define loss function --------

#PyTorch

#PyTorch
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=10):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

iou_loss = IoULoss(size_average=True)

def muti_iou_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = iou_loss(d0,labels_v)
	loss1 = iou_loss(d1,labels_v)
	loss2 = iou_loss(d2,labels_v)
	loss3 = iou_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = iou_loss(d5,labels_v)
	loss6 = iou_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data,loss1.data,loss2.data,loss3.data,loss4.data,loss5.data,loss6.data))

	return loss0, loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

dice_bce_loss = DiceBCELoss(size_average=True)

def muti_dice_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = dice_bce_loss(d0,labels_v)
	loss1 = dice_bce_loss(d1,labels_v)
	loss2 = dice_bce_loss(d2,labels_v)
	loss3 = dice_bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = dice_bce_loss(d5,labels_v)
	loss6 = dice_bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data,loss1.data,loss2.data,loss3.data,loss4.data,loss5.data,loss6.data))

	return loss0, loss

bce_loss = nn.BCELoss(size_average=True)

# def upload_file_to_azure(root_path, file_name, save_model=True):

#     account_name = 'apimodel'
#     account_key = 'Y+gW3VKqi6DdjFyNYxiv2hI6ZmMe5lngmVlTOte2MP70brdKcVN0b4qx8vk/3xtPoGlgP1ei0TOEhmewAba1Gg=='
#     container_name = 'model-shadow-weights-512'

#     block_blob_service = BlockBlobService(
#         account_name=account_name,
#         account_key=account_key
#     )

#     # for file_name in file_names:
#     if(save_model):
#         blob_name = f"piyush-cars-removebg/{file_name}"
#     else:
#         blob_name = f"piyush-cars-removebg/test_dataset_output/{file_name}"
#     file_path = f"{root_path}/{file_name}"
#     block_blob_service.create_blob_from_path(container_name, blob_name, file_path)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data,loss1.data,loss2.data,loss3.data,loss4.data,loss5.data,loss6.data))

	return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'

data_dir = "./data/final_aug_dataset_cars/"
tra_image_dir = 'images/'
tra_label_dir = 'masks/' 

image_ext = '.jpg'
label_ext = '.jpg'

# model_dir = './models/'
saved_model_dir = '/mnt/saved_models/u2net.pth'
saved_model_dir_cur = '/mnt/curr_output_models/u2net.pth'
output_model_dir = "/mnt/curr_output_models/"
if not os.path.exists(output_model_dir):
    os.makedirs(output_model_dir)
# if not os.path.exists(saved_model_dir):
    # s3.Bucket('peoplesegmentation').download_file('/u2_net/base_model/u2net.pth', saved_model_dir)
# if not os.path.exists(model_dir):
# 	os.makedirs(model_dir)


if(model_name=='u2net'): 
    net = U2NET(3, 1)
# elif(model_name=='u2netp'):
#     net = U2NETP(3,1)
# net.load_state_dict(torch.load(saved_model_dir))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading pretrained model from - ", output_model_dir)
# net = nn.DataParallel(net)
if(os.path.exists(saved_model_dir_cur)):
    print("Loading model ...")
    net.load_state_dict(torch.load(saved_model_dir_cur))     
else:
    print("Training model from scratch")   

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    net = nn.DataParallel(net)
    net.to(device)

else:
    if torch.cuda.is_available():
        print("CUDA AVAILable")
        # print(net)
        net.cuda()
    else:
        print("cuda unavailable")

# if not os.path.exists(saved_model_dir):
# 	os.makedirs(saved_model_dir)

epoch_num = 200
batch_size_train = 6
batch_size_test=1
batch_size_val = 1
train_num = 0
val_num = 0 

tra_lbl_name_list = glob.glob(data_dir + tra_label_dir + '*' + label_ext)
# for img_name in tra_lbl_name_list:
#     if('grocery_2346' in img_name):
#         tra_lbl_name_list.remove(img_name)

random.shuffle(tra_lbl_name_list)

print(data_dir + tra_label_dir + '*' + label_ext)

tra_img_name_list = []

no_images = len(os.listdir(data_dir + tra_label_dir))
print(no_images)

print(tra_img_name_list)
for img_path in tra_lbl_name_list:
	img_name = img_path.split("/")[-1]

	aaa = ".".join(img_name.split(".")[:-1])
	imidx = aaa ;
	tra_img_name_list.append(data_dir + tra_image_dir + imidx + image_ext)

print(tra_img_name_list[:10])
print(tra_lbl_name_list[:10])
print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(800),
        RandomCrop(720),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2, drop_last=True)

# val_folder = "/mnt/data/val_dataset"
# test_img_list = [os.path.join(val_folder, i) for i in os.listdir(val_folder)]

# test_salobj_dataset = SalObjDataset(img_name_list = test_img_list,
#                                     lbl_name_list = [],
#                                     transform=transforms.Compose([RescaleT(720),
#                                                                     ToTensorLab(flag=0)])
#                                     )
# test_salobj_dataloader = DataLoader(test_salobj_dataset,
#                                     batch_size=1,
#                                     shuffle=False,
#                                     num_workers=1)

# ------- 3. define model --------
# define the net


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def test_on_val_set(model, model_name):
    dir_name = "/mnt/val_results/"+model_name
    os.makedirs(dir_name)
    for i_test, data_test in enumerate(test_salobj_dataloader):

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= model(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(test_img_list[i_test],pred,dir_name)

        del d1,d2,d3,d4,d5,d6,d7
    
    zipf = zipfile.ZipFile(model_name+'.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir(dir_name, zipf)
    zipf.close()
    # upload_file_to_azure("./", model_name+'.zip',save_model=False)

    os.remove(model_name+'.zip')
 

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.01)
# scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0000001, max_lr=0.000001 , step_size_up=800, step_size_down=1342, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1,verbose=True)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000# save the model every 2000 iterations
for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']  

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize 
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        # loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
        # loss2, loss = muti_iou_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
        loss2, loss = muti_dice_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()
        # scheduler.step()  


        # # print statistics
        running_loss += loss.data
        running_tar_loss += loss2.data

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        # if ite_num == 1:
        #     filepath = output_model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val)

        #     torch.save(net.state_dict(), filepath)
        #     filename = model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val)
        #     # az.upload_file(filename, filepath)            
        #     # os.remove(filepath)
        #     net.train()  # resume train
        
        if ite_num % save_frq == 1:
            filepath = output_model_dir + model_name+"_size_720_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val)
            
            # torch.save(net.state_dict(), filepath)
            torch.save(net.state_dict(), saved_model_dir)
            filename = model_name+"_size_720_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val)
            torch.save(net.state_dict(), output_model_dir+filename)
            # upload_file_to_azure(output_model_dir, filename,save_model=True)
            # az.upload_file(filename, filepath)
            # print("file_uploaded")
            # s3.Bucket(bucketname).upload_file(filepath, os.path.join("24_million",filename))
            # os.remove(filepath)
            running_loss = 0.0
            running_tar_loss = 0.0
            ite_num4val = 0
        
            # test_on_val_set(net, filename.replace(".pth", ""))
            net.train()  # resume train

if __name__ == "__main__":
    main()
