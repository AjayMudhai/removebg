import cv2
from PIL import Image
import os 
import shutil


def namer(isrc_pth,idst_pth,msrc_pth,mdst_pth):
    c=0
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

        
isrc_pth='/datadrive/RemoveBG/data/Images'
idst_pth='/datadrive/RemoveBG/data/im2'
msrc_pth='/datadrive/RemoveBG/data/mask'
mdst_pth='/datadrive/RemoveBG/data/msk2'

###
namer(isrc_pth,idst_pth,msrc_pth,mdst_pth)





                
