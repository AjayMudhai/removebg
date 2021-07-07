import cv2
from PIL import Image
import os 
import shutil


def namer(isrc_pth,idst_pth,msrc_pth,mdst_pth):
    c=4000
    for root,dirs,files in os.walk(isrc_pth):
        for file in files:
            iop=os.path.join(root,file)
            mop=os.path.join(msrc_pth,file)
            if os.path.exists(mop):
                name='{}.jpg'.format(c)
                print(name)
                c+=1
                innp=os.path.join(idst_pth,name)
                mnnp=os.path.join(mdst_pth,name)
                shutil.move(iop,innp)
                shutil.move(mop,mnnp)

        
# isrc_pth='/datadrive/RemoveBG/data/data2/image2'
# idst_pth='/datadrive/RemoveBG/data/im2'
# msrc_pth='/datadrive/RemoveBG/data/data2/masks2'
# mdst_pth='/datadrive/RemoveBG/data/msk2'

# ###
# namer(isrc_pth,idst_pth,msrc_pth,mdst_pth)


def get_masks(isrc_pth,idst_pth,msrc_pth,mdst_pth):
    c=1
    for root,dirs,files in os.walk(msrc_pth):
        for file in files:
            print('{}/{}'.format(c,len(files)))
            mop=os.path.join(root,file)
            iop=os.path.join(isrc_pth,file)
            if os.path.exists(iop):
                name='{}.jpg'.format(c)
                c+=1
                innp=os.path.join(idst_pth,name)
                mnnp=os.path.join(mdst_pth,name)
                shutil.move(mop,mnnp)
                shutil.move(iop,innp)

isrc_pth='/datadrive/RemoveBG/removebg/U2Net/data/olx/car_inner_final_dataset/images'
idst_pth='/datadrive/RemoveBG/removebg/U2Net/data/olx/Images'
msrc_pth='/datadrive/RemoveBG/removebg/U2Net/data/olx/ola_mask22'
mdst_pth='/datadrive/RemoveBG/removebg/U2Net/data/olx/Masks'
get_masks(isrc_pth,idst_pth,msrc_pth,mdst_pth)



        



                
