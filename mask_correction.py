import cv2
from PIL import Image 
import numpy as np
import os


def mc(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c_max = []  
    max_area = 0  
    max_cnt = 0  
    for i in range(len(contours)):  
        cnt = contours[i]  
        area = cv2.contourArea(cnt)  
        # find max countour  
        if (area>max_area):  
            if(max_area!=0):  
                c_min = []  
                c_min.append(max_cnt)  
                # cv2.drawContours(image, c_min, -1, (0,0,0), cv2.FILLED)  
            max_area = area  
            max_cnt = cnt  
        else:  
            c_min = []  
            c_min.append(cnt)  
            # cv2.drawContours(image, c_min, -1, (0,0,0), cv2.FILLED)  
    
    c_max.append(max_cnt)  
    
    
    cv2.drawContours(image, c_max, -1, (255, 255, 255), thickness=-1)   
    image=cv2.bitwise_not(image)
    return image
    

def clean_mask(img):
    
    img_bw = 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')

    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

    mask = np.dstack([mask, mask, mask]) / 255
    out = img * mask
    return out

def denoising(img):
    kernel = np.ones((5,5), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    return img_erosion


def main(src_pth,dst_pth):
    for root,dirs,files in os.walk(src_pth):
        for file in files:
            try:
                op=os.path.join(root,file)
                img=cv2.imread(op)
                img=mc(img)
                # img=clean_mask(img)
                img=denoising(img)
                # img = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
                nnp=os.path.join(dst_pth,file)
                cv2.imwrite(nnp,img)
            except:
                pass


src_pth='/home/ajay/Documents/Spyne/RemoveBG/data_prep/masks'
dst_pth='/home/ajay/Documents/Spyne/RemoveBG/data_prep/mask4'
main(src_pth,dst_pth)

        



