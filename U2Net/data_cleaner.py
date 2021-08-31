import os 
import shutil 



def clean_data(src_pth,td_pth,dst_pth):
    c=1
    for root,dirs,files in os.walk(src_pth):
        for file in files:
            print('{}/{}'.format(c,len(files)))
            op=os.path.join(td_pth,file)
            if os.path.exists(op):
                nnp=os.path.join(dst_pth,file)
                shutil.move(op,nnp)





src_pth='/datadrive/RemoveBG/removebg/U2Net/data/test/images'
td_pth='/datadrive/RemoveBG/removebg/U2Net/data/images'
dst_pth='/datadrive/RemoveBG/removebg/U2Net/data/test/test2/i2'
clean_data(src_pth,td_pth,dst_pth)

src_pth='/datadrive/RemoveBG/removebg/U2Net/data/test/masks'
td_pth='/datadrive/RemoveBG/removebg/U2Net/data/masks'
dst_pth='/datadrive/RemoveBG/removebg/U2Net/data/test/test2/m2'
clean_data(src_pth,td_pth,dst_pth)
