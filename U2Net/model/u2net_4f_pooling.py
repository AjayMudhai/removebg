import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

### RSU-2F ###
class RSU2F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU2F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2,hx1),1))

        return hx1d + hxin

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bicubic', align_corners=False)
    src = src.clamp(min=0, max=255)
    return src


### RSU-9 ###
class RSU9(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU9,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool6 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool7 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv8 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv9 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv8d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv7d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)
        hx = self.pool6(hx5)

        hx7 = self.rebnconv7(hx)
        hx = self.pool7(hx5)

        hx8 = self.rebnconv6(hx)

        hx9 = self.rebnconv7(hx6)

        hx8d =  self.rebnconv8d(torch.cat((hx9,hx8),1))
        hx8dup = _upsample_like(hx8d,hx7)

        hx7d =  self.rebnconv7d(torch.cat((hx8,hx7),1))
        hx7dup = _upsample_like(hx7d,hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-8 ###
class RSU8(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU8,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool6 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv8 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv7d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)
        hx = self.pool6(hx5)

        hx7 = self.rebnconv7(hx)

        hx8 = self.rebnconv6(hx)

        hx7d =  self.rebnconv7d(torch.cat((hx8,hx7),1))
        hx7dup = _upsample_like(hx7d,hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin

##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NET,self).__init__()

        self.stage0 = RSU9(in_ch,32,64)
        self.pool01 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage0_1 = RSU8(64,32,128)
        self.pool01_1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage1 = RSU7(128,64,256)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(256,64,256)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        # self.stage2_2 = RSU6(256,64,256)
        # self.pool23_2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(256,128,512)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        # self.stage4 = RSU5(512,256,512)
        # self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        # self.stage4_2 = RSU4(512,256,512)
        # self.pool45_2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,1024)
        self.pool67 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage7 = RSU4F(1024,512,2048)

        # decoder
        self.stage6d = RSU4F(3072,512,1280)
        self.stage5d = RSU4(1792,256,640)
        # self.stage4d = RSU4(1152,256,640)
        # self.stage4_2d = RSU5(1152,128,768)
        self.stage3d = RSU5(1152,64,640)
        # self.stage2_2d = RSU6(640,64,192)
        self.stage2d = RSU6(896,64,320)
        self.stage1d = RSU7(576,64,160)
        self.stage0_1d = RSU8(288,64,80)
        self.stage0d = RSU9(144,32,40)

        self.side0 = nn.Conv2d(40,out_ch,3,padding=1)
        self.side0_1 = nn.Conv2d(80,out_ch,3,padding=1)
        self.side1 = nn.Conv2d(160,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(320,out_ch,3,padding=1)
        # self.side2_2 = nn.Conv2d(192,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(640,out_ch,3,padding=1)
        # self.side4_2 = nn.Conv2d(768,out_ch,3,padding=1)
        # self.side4 = nn.Conv2d(640,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(640,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(1280,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(7,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 0
        hx0 = self.stage0(hx)
        # print('hx0 size is: ', hx0.size())
        hx = self.pool01(hx0)
        # print('hx size is: ', hx.size())

        #stage 0_1
        hx0_1 = self.stage0_1(hx)
        # print('hx0 size is: ', hx0_1.size())
        hx = self.pool01_1(hx0_1)
        # print('hx size is: ', hx.size())

        #stage 1
        hx1 = self.stage1(hx)
        # print('hx1 size is: ', hx1.size())
        hx = self.pool12(hx1)
        # print('hx size is: ', hx.size())

        #stage 2
        hx2 = self.stage2(hx)
        # print('hx2 size is: ', hx2.size())
        hx = self.pool23(hx2)
        # print('hx size is: ', hx.size())

        #stage 2_2
        # hx2_2 = self.stage2_2(hx)
        # print('hx2_2 size is: ', hx2_2.size())
        # hx = self.pool23_2(hx2_2)
        # print('hx size is: ', hx.size())

        #stage 3
        hx3 = self.stage3(hx)
        # print('hx3 size is: ', hx3.size())
        hx = self.pool34(hx3)
        # print('hx size is: ', hx.size())

        #stage 4
        # hx4 = self.stage4(hx)
        # print('hx4 size is: ', hx4.size())
        # hx = self.pool45(hx4)
        # print('hx size is: ', hx.size())

        #stage 4_2
        # hx4_2 = self.stage4_2(hx)
        # print('hx4_2 size is: ', hx4_2.size())
        # hx = self.pool45_2(hx4_2)
        # print('hx size is: ', hx.size())

        #stage 5
        hx5 = self.stage5(hx)
        # print('hx5 size is: ', hx5.size())
        hx = self.pool56(hx5)
        # print('hx size is: ', hx.size())

        #stage 6
        hx6 = self.stage6(hx)
        # print('hx6 size is: ', hx6.size())
        hx = self.pool67(hx6)
        # print('hx size is: ', hx.size())
        
        # hx = _upsample_like(hx6,hx5)
        # # print('hx6up size is: ', hx.size())

        #stage 7
        hx7 = self.stage7(hx)
        # print('hx7 size is: ', hx7.size())
        hx7up = _upsample_like(hx7,hx6)
        # print('hx7up size is: ', hx7up.size())

        #-------------------- decoder --------------------
        hx6d = self.stage6d(torch.cat((hx7up,hx6),1))
        # print('hx6d size is: ', hx6d.size())
        hx6dup = _upsample_like(hx6d,hx5)
        # print('hx6dup size is: ', hx6dup.size())

        hx5d = self.stage5d(torch.cat((hx6dup,hx5),1))
        # print('hx5d size is: ', hx5d.size())
        hx5dup = _upsample_like(hx5d,hx3)
        # print('hx5dup size is: ', hx5dup.size())

        # hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        # print('hx4d size is: ', hx4d.size())
        # hx4dup = _upsample_like(hx4d,hx4_2)
        # print('hx4dup size is: ', hx4dup.size())

        # hx4_2d = self.stage4_2d(torch.cat((hx4dup,hx4_2),1))
        # print('hx4_2d size is: ', hx4_2d.size())
        # hx4_2dup = _upsample_like(hx4_2d,hx3)
        # print('hx4_2dup size is: ', hx4_2dup.size())

        hx3d = self.stage3d(torch.cat((hx5dup,hx3),1))
        # print('hx3d size is: ', hx3d.size())
        hx3dup = _upsample_like(hx3d,hx2)
        # print('hx3dup size is: ', hx3dup.size())

        # hx2_2d = self.stage2_2d(torch.cat((hx3dup,hx2_2),1))
        # print('hx2_2d size is: ', hx2_2d.size())
        # hx2_2dup = _upsample_like(hx2_2d,hx2)
        # print('hx2_2dup size is: ', hx2_2dup.size())

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        # print('hx2d size is: ', hx2d.size())
        hx2dup = _upsample_like(hx2d,hx1)
        # print('hx2dup size is: ', hx2dup.size())

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        # print('hx1d size is: ', hx1d.size())
        hx1dup = _upsample_like(hx1d,hx0_1)
        # print('hx1dup size is: ', hx1dup.size())

        hx0_1d = self.stage0_1d(torch.cat((hx1dup,hx0_1),1))
        # print('hx0_1d size is: ', hx0_1d.size())
        hx0_1dup = _upsample_like(hx0_1d,hx0)
        # print('hx0_1dup size is: ', hx0_1dup.size())

        hx0d = self.stage0d(torch.cat((hx0_1dup,hx0),1))
        # print('hx0d size is: ', hx0d.size())


        #side output
        d0 = self.side0(hx0d)
        # print('d0 size is: ', d0.size())

        d0_1 = self.side0_1(hx0_1d)
        # print('d0_1 size is: ', d0_1.size())
        d0_1 = _upsample_like(d0_1,d0)
        # print('d0_1 size is: ', d0_1.size())

        d1 = self.side1(hx1d)
        # print('d1 size is: ', d1.size())
        d1 = _upsample_like(d1,d0_1)
        # print('d1 size is: ', d1.size())

        d2 = self.side2(hx2d)
        # print('d2 size is: ', d2.size())
        d2 = _upsample_like(d2,d1)
        # print('d2 size is: ', d2.size())

        # d2_2 = self.side2(hx2_2d)
        # print('d2_2 size is: ', d2_2.size())
        # d2_2 = _upsample_like(d2_2,d2)
        # print('d2_2 size is: ', d2_2.size())

        d3 = self.side3(hx3d)
        # print('d3 size is: ', d3.size())
        d3 = _upsample_like(d3,d2)
        # print('d3 size is: ', d3.size())

        # d4_2 = self.side4_2(hx4_2d)
        # print('d4_2 size is: ', d4_2.size())
        # d4_2 = _upsample_like(d4_2,d3)
        # print('d4_2 size is: ', d4_2.size())

        # d4 = self.side4(hx4d)
        # print('d4 size is: ', d4.size())
        # d4 = _upsample_like(d4,d4_2)
        # print('d4 size is: ', d4.size())

        d5 = self.side5(hx5d)
        # print('d5 size is: ', d5.size())
        d5 = _upsample_like(d5,d1)
        # print('d5 size is: ', d5.size())

        d6 = self.side6(hx6d)
        # print('d6 size is: ', d6.size())
        d6 = _upsample_like(d6,d1)
        # print('d6 size is: ', d6.size())

        d0_ = self.outconv(torch.cat((d0,d0_1,d1,d2,d3,d5,d6),1))
        # print('d0 size is: ', d0_.size())

        return F.sigmoid(d0_), F.sigmoid(d0), F.sigmoid(d0_1), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3),F.sigmoid(d5), F.sigmoid(d6)

### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NETP,self).__init__()

        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(64,16,64)

        # decoder
        self.stage5d = RSU4F(128,16,64)
        self.stage4d = RSU4(128,16,64)
        self.stage3d = RSU5(128,16,64)
        self.stage2d = RSU6(128,16,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(64,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
