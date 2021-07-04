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

# from model.u2net_24million import U2NET
# from upload_saved_model import AZStorage

import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
from PIL import Image

import random
# import boto3

# s3 = boto3.resource('s3', aws_access_key_id='AKIAZDYZULCK6G7ECRRR',
#             aws_secret_access_key='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')
# bucketname = 'zapbg-training-datasets'

# s3.Bucket(bucketname).download_file("u2net.pth", "./test_models/u2net.pth")
# s3.Bucket(bucketname).download_file("FBA_matting/FBA.pth", './FBA_Matting/saved_models/FBA.pth')
# s3.Bucket(bucketname).download_file("testing_inputs.zip", "/home/azureuser/dataset/test_data/inputs.zip")

# bucketname = 'zapbg-training-datasets'

# torch.cuda.empty_cache()
# print(torch.cuda.is_available())
# az=AZStorage() 

# model_name = 'u2net' #'u2netp'

# data_dir = "../../dataset/test_data/content/content/inputs/"
# tra_image_dir = 'images/'
# tra_label_dir = 'masks/' 

# image_ext = '.jpg'
# label_ext = '.png'

# model_dir = './models/'
# saved_model_dir = './test_models/u2net.pth'

# if(model_name=='u2net'): 
#     net = U2NET(3, 1)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = nn.DataParallel(net)
# if(os.path.exists(saved_model_dir)):
#     net.load_state_dict(torch.load(saved_model_dir))        

# if torch.cuda.device_count() > 1:
#     print("Using", torch.cuda.device_count(), "GPUs")
#     net.to(device)
#     net.eval()

# else:
#     if torch.cuda.is_available():
#         print("CUDA AVAILable")
#         net.cuda()
#     else:
#         print("cuda unavailable")

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    mask = Image.fromarray(predict_np*255).convert('L')
    img_name = image_name.split(os.sep)[-1]
    # image = io.imread(image_name)
    image = Image.open(image_name)
    mask = mask.resize(image.size,resample=Image.BILINEAR)

    empty = Image.new("RGBA", image.size, 0)
    op = Image.composite(image, empty, mask)

    name = ".".join(image_name.split("/")[-1].split(".")[:-1])
    # pb_np = np.array(imo)

    # aaa = img_name.split(".")
    # bbb = aaa[0:-1]
    # imidx = bbb[0]
    # for i in range(1,len(bbb)):
    #     imidx = imidx + "." + bbb[i]

    op.save(os.path.join(d_dir,name+'.png'), quality=50)


def get_output(img_name_list):
  # --------- 1. get image path and name ---------

  # img_name_list = glob.glob(image_dir + '*')
  # print(img_name_list)

  # --------- 2. dataloader ---------
  #1. dataloader
  test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                      lbl_name_list = [],
                                      transform=transforms.Compose([RescaleT(288),
                                                                    ToTensorLab(flag=0)])
                                      )
  test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=0)
  print("Starting inference")
  # --------- 4. inference for each image ---------
  for i_test, data_test in enumerate(test_salobj_dataloader):
      # try:
    print("inferencing:",img_name_list[i_test].split("/")[-1])

    inputs_test = data_test['image']
    inputs_test = inputs_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d0_, d0_1, d0, d1, d2, d3, d4, d4_2, d5, d6 = net(inputs_test)
    # d0,d1,d2,d3,d4,d5,d6 = net(inputs_test)
    # normalization
    pred = d0_[:,0,:,:]
    pred = normPRED(pred)

    # save results to test_results folder
    save_output(img_name_list[i_test],pred,"../../dataset/test_data/u2net_outputs/")

    # del d0,d1,d2,d3,d4,d5,d6
    del d0_, d0_1, d0, d1, d2, d3, d4, d4_2, d5, d6
      # except:
      #   print("Failed for : ", img_name_list[i_test])
      #   pass

# image_name_list = None
# image_name_list = []
# for image in os.listdir(data_dir):
#   image_name_list.append(data_dir+image)
# print(len(image_name_list))

# get_output(img_name_list=image_name_list)

# for image_name in image_name_list:
#   image_name = ".".join(image_name.split("/")[-1].split(".")[:-1])
#   img = Image.open(data_dir+image_name+".jpg")
#   mask = Image.open("../../dataset/test_data/u2net_outputs/"+image_name+'.png').resize(img.size).convert("L")
#   empty = Image.new("RGBA", img.size, 0)
#   op=Image.composite(img, empty, mask)
#   op.save("../../dataset/test_data/u2net_final/"+image_name+".png", 'PNG', quality=25)

# Our libs
# from FBA_Matting.networks.transforms import trimap_transform, groupnorm_normalise_image
# from FBA_Matting.networks.models import build_model
# from FBA_Matting.dataloader import read_image, read_trimap, image_resize

# System libs
import os

# External libssdbcjdbjcb sjcjddbv
import cv2
from PIL import Image
import numpy as np
import torch 
import uuid

# encoder = 'resnet50_GN_WS'
# decoder = 'fba_decoder'
# weights = './FBA_Matting/saved_models/FBA.pth'
# image_dir = './FBA_Matting/images/input_images/'
# trimap_dir = './FBA_Matting/images/trimap_images/'
# fba_output_dir = './FBA_Matting/images/output_images/'
# pred_dir = "./FBA_Matting/images/final_images/"

# model = build_model(encoder, decoder, weights)
# model.eval()

# def np_to_torch(x):
#     return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()

# def scale_input(x: np.ndarray, scale: float, scale_type) -> np.ndarray:
#     ''' Scales inputs to multiple of 8. '''
#     h, w = x.shape[:2]
#     h1 = int(np.ceil(scale * h / 8) * 8)
#     w1 = int(np.ceil(scale * w / 8) * 8)
#     x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
#     return x_scale

# def pred(image_np: np.ndarray, trimap_np: np.ndarray, model) -> np.ndarray:
#     ''' Predict alpha, foreground and background.
#         Parameters:
#         image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
#         trimap_np -- two channel trimap, first background then foreground. Dimensions: (h, w, 2)
#         Returns:
#         fg: foreground image in rgb format between 0 and 1. Dimensions: (h, w, 3)
#         bg: background image in rgb format between 0 and 1. Dimensions: (h, w, 3)
#         alpha: alpha matte image between 0 and 1. Dimensions: (h, w)
#     '''
#     h, w = trimap_np.shape[:2]

#     image_scale_np = scale_input(image_np, 1.0, cv2.INTER_LANCZOS4)
#     trimap_scale_np = scale_input(trimap_np, 1.0, cv2.INTER_LANCZOS4)

#     with torch.no_grad():

#         image_torch = np_to_torch(image_scale_np)
#         trimap_torch = np_to_torch(trimap_scale_np)

#         trimap_transformed_torch = np_to_torch(trimap_transform(trimap_scale_np))
#         image_transformed_torch = groupnorm_normalise_image(image_torch.clone(), format='nchw')

#         output = model(image_torch, trimap_torch, image_transformed_torch, trimap_transformed_torch)

#         output = cv2.resize(output[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)
#     alpha = output[:, :, 0]
#     fg = output[:, :, 1:4]
#     bg = output[:, :, 4:7]

#     alpha[trimap_np[:, :, 0] == 1] = 0
#     alpha[trimap_np[:, :, 1] == 1] = 1
#     fg[alpha == 1] = image_np[alpha == 1]
#     bg[alpha == 0] = image_np[alpha == 0]
#     return fg, bg, alpha

# def get_alpha_image(image, fg, alpha, trimap_path):
#     # Read Trimap, Foreground and Alpha Mask Image. Resize Foreground, Original Image and Alpha Mask
#     trimap = np.array(Image.open(trimap_path).convert('L')).astype('uint8')
#     size_ = (trimap.shape[1], trimap.shape[0])
#     # print("Trimap Size is: ", size_)
#     original = Image.fromarray(image).resize(size_, resample=Image.LANCZOS).convert('RGBA')
#     img = fg.resize(size_, resample=Image.LANCZOS).convert('RGBA')
#     mask = alpha.resize(size_, resample=Image.LANCZOS)
#     #create empty image for output of Composite images
#     empty = Image.new("RGBA", size_, 0)

#     mask_array, fg_array = np.array(mask), np.array(img)

#     # print("Mask shape:", mask_array.shape, "Foreground shape: ", fg_array.shape, "Trimap Shape: ", trimap.shape, "Original Image: ", np.array(original).shape)
#     #To create inner alpha mask not for whole image but only for edges, make all values inside the border unmasked(255).
#     #Do same with foreground Image, just replace it with original Image corresponding values.
#     mask_array[trimap>200] = 255
#     fg_array[trimap>200] = np.array(original)[trimap>200]

#     mask = Image.fromarray(mask_array).convert("L")
#     img = Image.fromarray(fg_array)
#     #Run the Composite function and get final image
#     op = Image.composite(img, empty, mask)
#     image = np.array(op).astype('uint8')
#     image[:,:,3][image[:,:,3] < 180] = 0
#     op = Image.fromarray(image)
    
#     #Remove all stored Images
#     # os.remove(fg_path)
#     # os.remove(alpha_path)
#     # os.remove(trimap_path)

#     return op, mask

# def get_trimap_masks(mask, image_name):
#     mask_ = mask.resize((1080, 1080))
#     trimap = np.array(mask_).astype('uint8')
#     ret, thresh = cv2.threshold(trimap, 127, 255, cv2.THRESH_BINARY)
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(thresh, contours, -1, (128,128,128), 50)
#     trimap = Image.fromarray(thresh).resize(mask.size)
#     # trimap = Image.fromarray(trimap).resize(mask.size)
#     trimap.save(trimap_dir+image_name)
#     # trimap.save("same_trimap.png")
#     return trimap

# for mask_name in os.listdir('../../dataset/test_data/u2net_outputs/'):
#   try:
#     mask = Image.open('../../dataset/test_data/u2net_outputs/'+mask_name).convert('L')
#     get_trimap_masks(mask, mask_name)
#   except:
#     print(mask_name)

# save_dir = './FBA_Matting/images/final_images/'
# for name in os.listdir(data_dir):
#   try:
#     #Resize Orignal Image to smaller size for better processing and hardware constraints
#     image = np.array(Image.open(data_dir+name)).astype('uint8')
#     trimap_path = trimap_dir+".".join(name.split(".")[:-1])+'.png'
#     print(trimap_path)
#     image_re = read_image(image)
#     #Read trimap in gray scale mode
#     trimap = read_trimap(trimap_path)
#     item_dict = {'image': image_re, 'trimap': trimap, 'name': str(uuid.uuid1())}

#     image_np = item_dict['image']
#     trimap_np = item_dict['trimap']

#     #Predict alpha mask from resized image, trimap mask and our FBA_Matting model 
#     fg, bg, alpha = pred(image_np, trimap_np, model)
#     size_ = (image.shape[1], image.shape[0])

#     #Save foreground, background Image and Alpha Mask
#     fg_path, alpha_path = os.path.join(save_dir,  item_dict['name']+ '_fg.png'), os.path.join(save_dir, item_dict['name']+ '_alpha.png')
#     fg = Image.fromarray((fg[:, :, ::-1] * 255).astype('uint8'))#.save(fg_path)
#     alpha = alpha*255
#     alpha[alpha<40] = 0
#     alpha = Image.fromarray((alpha).astype('uint8'))#.save(alpha_path)
#     alpha.save(fba_output_dir+".".join(name.split(".")[:-1])+'.png')
#     #Get final foreground Image from alpha mask
#     foreground, mask = get_alpha_image(image, fg, alpha, trimap_path)
#     foreground.save(save_dir+".".join(name.split(".")[:-1])+'.png', 'PNG', quality=25)
#   except:
#     pass
