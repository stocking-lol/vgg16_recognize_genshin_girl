import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x
def cvtColor(img):
    if len(np.shape(img)) == 3 and np.shape(img)[-2] == 3:
        return img
    else:
        img = img.convert('RGB')
        return img

class DataGenerator(data.Dataset):
    def __init__(self, annotation_lines, input_shape, random=True):
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.random = random

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        annotation_path = self.annotation_lines[index].split(';')[1].split('\n')[0]
        img = Image.open(annotation_path)
        img = self.get_random_data(img,self.input_shape,random=self.random)
        img = np.transpose(preprocess_input(np.array(img).astype(np.float32)), [2, 0, 1])
        y = int(self.annotation_lines[index].split(';')[0])
        return img, y

    def rand(self,a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self,img,input_shape,jitter=.3,hue=.1,sat=0.5,val=0.5,random=True):

        img = cvtColor(img)
        iw, ih = img.size
        _,h,w = input_shape
        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = int((w-nw)/2)
            dy = int((h-nh)/2)

            img = img.resize((nw,nh), Image.BICUBIC)
            new_img = Image.new('RGB', (w,h),(128,128,128))

            new_img.paste(img,(dx,dy))
            image_data = np.array(new_img,np.float32)
            return image_data
        new_ar = w/h*self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
        scale=self.rand(.75,1.25)
        if new_ar < 1:
            nh=int(scale*h)
            nw=int(nh*new_ar)
        else:
            nw=int(scale*w)
            nh=int(nw/new_ar)
        img = img.resize((nw,nh), Image.BICUBIC)
        #多余的部分贴灰条
        dx=int(self.rand(0,w-nw))
        dy=int(self.rand(0,h-nh))
        new_img = Image.new('RGB', (w,h), (128,128,128))
        new_img.paste(img,(dx,dy))
        img = new_img
        #图片翻转
        flit=self.rand()<.5
        if flit: img = img.transpose(Image.FLIP_LEFT_RIGHT)
        rotate=self.rand()<.5
        if rotate:
            angle = np.random.randint(-15,15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            img = cv2.warpAffine(np.array(img),M,(w,h),borderValue=(128,128,128))
        #色域扭曲
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(img,np.float32)/225, cv2.COLOR_RGB2HSV)
        x[...,1] *= sat
        x[...,2] *= val
        x[x[:,:,0] > 360 , 0] = 360
        x[:,:,1:][x[:,:,1:]>1] = 1
        x[x<0]  = 0
        img_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 225
        return img_data