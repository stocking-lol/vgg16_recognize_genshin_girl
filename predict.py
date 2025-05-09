import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from net import vgg16
from torchvision import transforms

#加载网络
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = vgg16()
model = torch.load('./animatecharacter.6.pth',map_location=device)
net.load_state_dict(model)
net.eval()

#类名
class_names = ['keqing', 'ganyu']

#获取输入图片的路径
img_path = './test1.keqing.jpg'
img = Image.open(img_path)

#将图片进行处理
if img.mode == "RGBA":
    img = img.convert("RGB")
image = img
transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
img = transforms(img)
img = torch.reshape(img,(1,3,224,224))
#进行预测
'''
prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction, axis=1)[0]
predicted_class = class_names[predicted_class_index]
predicted_class_probability = prediction[0][predicted_class_index]
'''
with torch.no_grad():
    output = net(img)
output = F.softmax(output,dim=1)
output = output.data.cpu().numpy()
#输出预测
print(output)
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title("Predicted:ganyu{:.1%},keqing{:.1%}".format(output[0,0],output[0,1]))
plt.axis('off')
plt.show()
