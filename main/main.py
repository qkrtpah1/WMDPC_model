# 이미지 로드 함수

import cv2
import os

def load_images(path):
    defects = {'A':0, 'B':1 , 'C':2}
    images = []
    labels = []
    image_dir = image_dir
    image_list = os.listdir(image_dir)


    

    for defect in defects:
        print("loading:", path, defect, defects[defect])
        path_dir = path +"/"+ defect
        file_list = os.listdir(path_dir)
        for file_name in file_list: 
            img_path = path_dir +'/'+ file_name
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,(224,224))
            images.append(image)
            labels.append(defects[defect])

        print("image count : ", len(image_list))
        return (images, labels)


## 딥러닝 모델 구축

import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

np.random.seed(0) #여기까지

loader = DataLoader(dataset,batch_size=32,shuffle=True)


class CNNModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.fc = nn.Sequential(

            nn.Linear(32*56*56,128),
            nn.ReLU(),
            nn.Linear(128,2)

        )

    def forward(self,x):

        x = self.conv(x)

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        return x


model = CNNModel()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


for epoch in range(5):

    for images in loader:

        outputs = model(images)

        labels = torch.zeros(images.size(0),dtype=torch.long)

        loss = criterion(outputs,labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print("epoch:",epoch,"loss:",loss.item())