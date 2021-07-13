import os,json,subprocess,time
import sys,requests
import cv2
from PIL import Image
import numpy as np
import datetime
from google.cloud import storage
import pathlib
import time
import pdb



class BulkTesting:
    def __init__(self):
        self.input_dir='/datadrive/RemoveBG/removebg/U2Net/data/images'
        # self.model_route="http://52.177.21.16/models/removebg720/"
        self.model_route="http://34.95.80.165/models/removebg720/"   
        # self.model_route2="http://20.96.11.194/models/replacecarbg/"
        self.model_route2="http://34.95.80.165/models/replacecarbg/"
        # self.model_route2="http://52.177.21.16/automobile/reflection_bg/"
        parent_dir=pathlib.Path().absolute()
        self.edited_dir=os.path.join(parent_dir,'edited')
        self.combined_dir=os.path.join(parent_dir,'combined')
        self.temp_dir=os.path.join(parent_dir,'temp')
        if not os.path.exists(self.edited_dir):
            os.makedirs(self.edited_dir)
        # if not os.path.exists(self.combined_dir):
        #     os.makedirs(self.combined_dir)
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.failed_list=[]
        self.allowed_ext=["jpg","png","jpeg","JPEG","PNG","webp"]   
        self.get_output()

    def save_to_gcp(self,img,imagename,extension,path):

        
        url = "https://storage.googleapis.com/spyne/AI/edited/"+imagename+'.'+extension 
         

    

        storage_client = storage.Client.from_service_account_json("spyne-ai-projects-4540f2fd8b95.json")
        bucket = storage_client.get_bucket("spyne")
        name = imagename+'.'+extension
     
        blob = bucket.blob('AI/edited/' + name)
        # new_img = open(image_save_path,'rb')
        blob.upload_from_filename(path)
        # os.remove(os.path.join(image_save_path))

        return url



    def get_output(self):
        pcount=0
        fcount=0
        s_time=time.time()
        Total_time=0
        for root,dirs,files in os.walk(self.input_dir):
            for file in files:
                
               
                op=os.path.join(root,file)
                # pdb.set_trace()
                img=cv2.imread(op)
                ext=file.split('.')[-1] 
                if ext in self.allowed_ext:
                    el=len(ext)+1
                    name=file[:-el]
                
                    url=self.save_to_gcp(img,name,ext,op)
    ###############################################################
                    payload = {"image_url":url,"roi":"outer"}
                    # print(url)
                    # n=input()
    ################################################################
                    try:
                        response2 = requests.request("POST",self.model_route,data=payload)
                        transparent_url = response2.json()['url']
                        
                        
                        
                        
                        r=requests.get(transparent_url)
                        
                    
                        temp_pth=os.path.join(self.temp_dir,name)
                        open(temp_pth,'wb').write(r.content)
                        # open(os.path.join(self.edited_dir,name+"_output.png"),'wb').write(r.content)
                        output=cv2.imread(temp_pth,cv2.IMREAD_UNCHANGED)
                        w,h,_=output.shape
                        output=Image.fromarray(output.astype("uint8"))
                        
                        wbg_img=Image.new("RGB",(h,w),(255,255,255))
                        wbg_img.paste(output,(0,0),output)
                        output=np.array(wbg_img, dtype=np.uint8)
                    
                        self.save_outputs(img,output,name)
                        os.remove(temp_pth)
                        pcount+=1
                    except:
                        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        print('Failed for :{}.'.format(file))
                        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        self.failed_list.append(file)
                        fcount+=1
                        n=input()

                    e_time=time.time()
                    processing_time=e_time-s_time
                    Total_time=Total_time+processing_time
                    s_time=e_time
                    print('Processed : {}/{}    {}'.format(pcount,len(files),processing_time))
                    print('Failed    : {}/{}'.format(fcount,len(files)))
                    print('Failed : {}'.format(self.failed_list))
                    
    



    def save_outputs(self,raw_img,edited_img,name):
        oname='{}.jpg'.format(name)
        co_name='{}_combined.png'.format(name)
       
        edited_pth=os.path.join(self.edited_dir,oname)
        # cv2.imwrite(edited_pth,edited_img)
        # white_value=255
        # transparent_image = Image.fromarray(edited_img)
        # new = Image.new("RGB", transparent_image.size, (white_value, white_value, white_value))
        # new.paste(transparent_image, (0,0), transparent_image)
        # new = np.array(new)
        # ip_op=cv2.hconcat([raw_img,new])
        # combined_pth=os.path.join(self.edited_dir,co_name)
      
        cv2.imwrite(edited_pth,edited_img) 



         # ip_op=cv2.hconcat([raw_img,edited_img]) 
        # combined_pth=os.path.join(self.edited_dir,co_name)
      
        # cv2.imwrite(combined_pth,ip_op)

        










bt=BulkTesting()
        
 