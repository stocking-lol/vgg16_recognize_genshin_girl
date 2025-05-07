import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import DataGenerator
from net import vgg16

annotation_path = 'cls_train.txt'
with open(annotation_path, 'r') as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*0.1)
num_train = len(lines)-num_val
#输入图片大小
input_shape = (3, 224, 224)
train_data = DataGenerator(lines[:num_train],input_shape=input_shape,random=True)
val_data = DataGenerator(lines[num_train:],input_shape=input_shape,random=False)
val_len = len(val_data)
"""加载数据"""
get_train = DataLoader(train_data, batch_size=32, shuffle=True)
get_val = DataLoader(val_data, batch_size=32, shuffle=False)
'''构建网络'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = vgg16(pretrained=True, progress=True, num_classes=6)
net.to(device)
'''选择优化器和学习率的调整方法'''
lr = 0.0001
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
lr_sculer =torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
torch.save(net.state_dict(), 'vgg16.pth')
'''训练'''
epochs = 50
for epoch in range(epochs):
    total_train = 0
    for data in get_train:
        img, label = data
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
        optimizer.zero_grad()
        output = net(img)
        train_loss = nn.CrossEntropyLoss()(output, label).to(device)
        train_loss.backward()
        optimizer.step()
        total_train += train_loss
    total_test = 0
    total_accuracy = 0
    for data in get_val:
        img, label = data
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = net(img)
            test_loss = nn.CrossEntropyLoss()(output, label).to(device)
            total_test += test_loss
            total_accuracy += (torch.argmax(output, dim=1) == label).sum().item()
    print('训练集上的损失，{} '.format(total_train))
    print("测试集上的损失，{}".format(total_test))
    print("测试集上的精度,{:.1%}".format(total_accuracy / val_len))
    torch.save(net.state_dict(), 'animatecharacter.{}.pth'.format(epoch+1))
    print("保存模型")