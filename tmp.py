import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms # 이미지 변환(전처리) 기능을 제공
from torch.autograd import Variable
from torch import optim # 경사하강법을 이용하여 가중치를 구하기 위한 옵티마이저
import os # 파일 경로에 대한 함수들을 제공
import cv2
from PIL import Image
from tqdm import tqdm_notebook as tqdm # 진행 상황 표현
import random
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ['dog', 'elephant', 'giraffe','guitar','horse','house','person']

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),  ## 1
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase):  ## 2
        return self.data_transform[phase](img)

path = './images/train/'
test_path = './images/test/0/'

cls_list = os.listdir(path)
data_list = []
for i in cls_list:
    tmp_path = path + i+'/'

    tmp_images_path = sorted([os.path.join(tmp_path, f)
                               for f in os.listdir(tmp_path)])
    for j in tmp_images_path:
        data_list.append(j)

random.seed(42) ## 4
random.shuffle(data_list)

train_images_filepaths = data_list[:1200]
val_images_filepaths = data_list[1200:-10]

test_list = os.listdir(test_path)
test_images_filepaths = []
for i in test_list:
    test_images_filepaths.append(test_path+i)

# test_images_filepaths = data_list[-10:]


def display_image_grid(images_filepaths, predicted_labels=(), cols=5):
    rows = len(images_filepaths) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(12, 6))
    ax = ax.ravel()

    for i, image_filepath in enumerate(images_filepaths):  # 경로를 정규화 함
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  ## 1

        true_label = os.path.normpath(image_filepath).split(os.sep)[-2]  ## 2
        predicted_label = predicted_labels[i] if predicted_labels else true_label  ## 3
        color = "green" if true_label == predicted_label else "red"

        ax[i].imshow(image)
        ax[i].set_title(predicted_label, color=color)
        ax[i].axis("off")

    plt.tight_layout()
    plt.show()


class DogvsCatDataset(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform  # DogvsCatDataset 클래스를 호출할 때 transform에 대한 매개변수를 받아 옵니다.
        self.phase = phase  # train 적용

    def __len__(self):  # images_filepaths 데이터셋의 전체 길이 반환
        return len(self.file_list)

    def __getitem__(self, idx):  # 데이터셋에서 데이터를 가져오는 부분으로 결과는 텐서 형태
        img_path = self.file_list[idx]
        img = Image.open(img_path)  # img_path 위치에서 이미지 데이터들을 가져옴
        img_transformed = self.transform(img, self.phase)  # 이미지에 'train' 전처리 적용
        label = img_path.split('/')[-2]

        if label == 'dog':
            label = 0
        elif label == 'elephant':
            label = 1
        elif label == 'giraffe':
            label = 2
        elif label == 'guitar':
            label = 3
        elif label == 'horse':
            label = 4
        elif label == 'house':
            label = 5
        elif label == 'person':
            label = 6

        return img_transformed, label

size = 227
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 32

train_dataset = DogvsCatDataset(train_images_filepaths,
                                transform=ImageTransform(size,mean, std),
                                phase='train') # train 이미지에 train_transforms를 적용
val_dataset = DogvsCatDataset(val_images_filepaths,
                              transform=ImageTransform(size,mean, std),
                              phase='val') # val 이미지에 val_transforms를 적용
test_dataset = DogvsCatDataset(test_images_filepaths,
                              transform=ImageTransform(size,mean, std),
                              phase='val') # val 이미지에 val_transforms를 적용

index = 0
print(train_dataset.__getitem__(index)[0].size()) # 훈련 데이터 train_dataset.__getitem__[0][0]의 크기(size) 출력
print(train_dataset.__getitem__(index)[1]) # 훈련 데이터의 레이블 출력

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) ## 1
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# train_dataloader와 val_dataloader를 합쳐서 표현
dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

batch_iterator = iter(train_dataloader)
inputs, label = next(batch_iterator)
print(inputs.size())
print(label)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(46656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(50):   # 데이터셋을 수차례 반복합니다.

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        running_loss += loss.item()
        # print('1')
        if i % 10 == 9:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.7f}')
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
dataiter = iter(test_dataloader)
images, labels = next(dataiter)

# 이미지를 출력합니다.
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


# inference

# net = Net()
# net.load_state_dict(torch.load(PATH))
#
# outputs = net(images)
#
# _, predicted = torch.max(outputs, 1)
#
# print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
#                               for j in range(len(predicted))))

# with torch.no_grad():
#     for data in test_dataloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
#                                       for j in range(len(predicted))))

# with torch.no_grad():
#     for i, data in enumerate(test_dataloader):
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#
#         # 파일 위치 출력
#         print('File Path:', test_dataloader.dataset.file_list[i])
#
#         # 결과 출력
#         print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
#                                       for j in range(len(predicted))))

import csv

# Initialize an empty list to store predictions
predictions = []

with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        # Store the prediction value (predicted class index) in the list
        predictions.append(predicted.item())

# Define the CSV file path
csv_file_path = 'predictions.csv'

# Write the predictions to the CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['answer', 'value'])  # Write the header
    for i, prediction in enumerate(predictions):
        writer.writerow([i, prediction])