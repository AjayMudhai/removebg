import os
from os import listdir
from PIL import Image
from skimage import io, transform, color

def file_tester(src_pth,msk_pth):
    a=[]
    c=0
    for root,dirs,files in os.walk(src_pth):
        for file in files:
            print('{}/{}'.format(c,len(files)))
            c+=1
            op=os.path.join(root,file)
            mop=os.path.join(msk_pth,file)
            try:
                img = io.imread(op)
                img = io.imread(mop) # open the image file
                 # verify that it is, in fact an image
              
            except (IOError, SyntaxError) as e:
                print('Bad file:', file)
                os.remove(op)
                os.remove(mop)
                a.append(file)
                print(a)


src_pth='/datadrive/RemoveBG/removebg/U2Net/data/images'
msk_pth='/datadrive/RemoveBG/removebg/U2Net/data/masks'
file_tester(src_pth,msk_pth)

    
                
            #os.remove(base_dir+"\\"+filename) (Maybe)