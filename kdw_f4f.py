"""
error index만 F4F에 넣어서 offset을 뽑아서 돌려보기
실행하지말자 -> path 덮어씌워질듯
"""

import os
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision import models
import torchvision.models as models
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

pretrained_model = models.vgg16_bn(pretrained=True)
#summary(pretrained_model, (3, 224, 224)) # (channels, width, height)
if torch.cuda.device_count() > 1: # batch size가 알아서 여러개의 gpu로 분배
    pretrained_model = nn.DataParallel(pretrained_model)
pretrained_model.cuda() # model에 GPU 할당
print(pretrained_model)

new_model=models.vgg16_bn(pretrained=True)
if torch.cuda.device_count() > 1: # batch size가 알아서 여러개의 gpu로 분배
    new_model = nn.DataParallel(new_model)
new_model.cuda() # model에 GPU 할당

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transforms_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # numpy를 tensor(torch)로 바꾸어 넣는다. 이미지의 경우 픽셀 값 하나는 0 ~ 255 값을 갖는다. 하지만 ToTensor()로 타입 변경시 0 ~ 1 사이의 값으로 바뀜.
    normalize,
])

transforms_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), # 바로 resize 하지 말고 여기서 input format으로 맞춤.
    transforms.ToTensor(),
    normalize,
])

file = open('Filter_for_Filter_result.txt', 'w')    # hello.txt 파일을 쓰기 모드(w)로 열기. 파일 객체 반환
num_epochs = 80
batchsize = 32
lr = 0.001
class_num=1000 # class 개수
channel_per_packet=2 # channel당 packet 수 (pooling5에서 자르면 8이 된다.)
packet_loss_per_feature=64 # feature의 총 256개 packet 중에서 packet loss 개수 (즉, feature당 channel loss는 512개 중에서 2 * 64로 128개가 loss 된다.)

TRAIN_DATA_PATH = "/media/2/Network/Imagenet_dup/train"
TEST_DATA_PATH="/media/2/Network/Imagenet_dup/val"

### data loader ###

trainset = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transforms_train) # numpy를 Tensor로 바꾸어 넣는다.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True,num_workers=4)
testset = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=transforms_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=True, num_workers=4)
# num_workers : how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)


loss_start_index=0
# hook 설정!
def preprocessing(name): # pre_hook 부분
    def hook(model, input): # pre hook은 해당 layer의 pre processing 부분이기에 output이 없어서 output을 parameter로 놓지 않는다.
        input[0][:,loss_start_index:loss_start_index+128] = 0 # channel loss 발생! (input이 2D이기에 input[0]으로 해줘야 우리가 생각하는 feature이다.)
    return hook

class F4F_only_error_index(nn.Module): # error index만 넣어서 offset(3 x 3 x 512) 뽑는다. offset은 512개 각 filter에 동일하게 적용된다. 
    def __init__(self):
        super(F4F_only_error_index, self).__init__()
        self.fc1=nn.Linear(512, 3 * 3 * 512* 512) # input : error bit vector(512), output : 3 x 3 x 512 x 512 (filter set offset)
    def forward(self,x): # x는 data를 나타낸다.
        x=self.fc1(x) # data가 fc를 지나간다.
        #output=F.tanh(x) # x를 tanh activation function에 대입한다.
        output=torch.tanh(x)
        return output

activation1 = {}
def get_activation1(name): # 기존 모델에서 conv5_1의 output feature 뽑기
    def hook(model, input, output):
        activation1[name] = output.detach()
    return hook
        
activation2 = {}
def get_activation2(name): # 기존 모델에서 conv5_1의 output feature 뽑기
    def hook(model, input, output):
        activation2[name] = output
    return hook
    
F4F=F4F_only_error_index() # model 선언
F4F.cuda()
criterion = nn.MSELoss().cuda() # cost function
optimizer = torch.optim.SGD(F4F.parameters(), lr=lr, weight_decay=1e-4) # optimizer
F4F.train()
before_accuracy=0.0
before_lr=lr    

##### 새로운 모델에서 pre hook ##### (error input 넣기)
for name, module in new_model.named_modules():
    if name=="features.34": # conv 5-1 pre hook 걸기 (input feature에 error 넣기)
        module.register_forward_pre_hook(preprocessing(name)) # pre hook (forward)
        break

#### 기존 모델에서 forward hook 통해서 conv5_1의 output 저장 ###
for name, module in pretrained_model.named_modules():
    if name=="features.34": # conv 5-1 위치
        pretrained_model.features[34].register_forward_hook(get_activation1(name))
        break
            
#### 새로운 모델에서 conv5_1의 결과 뽑기 ####
for name, module in new_model.named_modules():
    if name=="features.34": # 
        new_model.features[34].register_forward_hook(get_activation2(name))
        break

for epoch in range(num_epochs): # epoch 80번
    error_index=[] # error 위치는 1, 아닌것은 0 (channel 위치)
    loss_start_index=128*(epoch%4) # 첫번째 epoch : 0~127 channel 깨짐, 두번째 epoch : 128~255 channel 깨짐, 세번째 epoch : 256~383번째 channel 깨짐, 네번째 epoch : 384~511번째 channel 깨짐. 이렇게 4 주기로 진행됨
    for index in range(512):
        if loss_start_index<=index and index < loss_start_index+128: # error 위치
            error_index.append(1)
        else:
            error_index.append(0)
    error_index=torch.Tensor(error_index)
    error_index=error_index.cuda() # 512 bit vector (error location)
    
    result = F4F(error_index) # result : 4608의 Tensor 형태 (512 x 3 x 3) => filter의 offset으로 사용될 예정. 더하고 빼는 경우 2가지 다 해보자.
    result=torch.reshape(result,[512,512,3,3])  # filter에 넣을 형태로 변경
        
    #### filter를 변경한 새로운 모델 ####
    for name, parameter in new_model.named_parameters():
        if name == 'features.34.weight': # 바꿀 filter (multi gpu를 쓰면 module.features.34.weight 이렇게 이름 바뀜!!!!!!!!!!!!!!!!!!!!!!!!!)
            new_model.features[34].weight.data = parameter[:]+result # 새로운 filter
            break       
        
    ####### train #######
    for idx, (images, labels) in enumerate(tqdm(trainloader,desc=f'EPOCH {epoch} ')):
        images = images.cuda()
        out1=pretrained_model(images)
        out2=new_model(images)
            
        #### 학습 #####
        optimizer.zero_grad() #  autograd에서 gradient가 축적 되기 때문에, gradient를 통해 가중치들을 업데이트할 때마다, 다음으로 넘어가기 전에 이 zero_() 메소드를 통해 gradient를 0로 만들어 줘야한다. 
        loss = criterion(activation2['features.34'],activation1['features.34']) # MSE cost function 적용
        loss.backward() # autograd 를 사용하여 역전파 단계를 계산합니다. 이는 requires_grad=True를 갖는 모든 텐서들에 대한 손실의 변화도를 계산합니다.
        optimizer.step()
        if idx>1600:
            break
        
    ###### test (새로운 filter가 들어간 vgg16) ######             
    print("Test start!!!")
    new_model.eval() # Change model to 'eval' mode (BN uses moving mean/var)

    correct_top1 = 0
    total = 0
    with torch.no_grad(): # 이 컨텍스트 내부에서 새로 생성된 텐서들은 requires_grad=False 상태가 되어, 메모리 사용량을 아껴준다. (훈련 안함)
        for idx, (images, labels) in enumerate(testloader):
            images = images.cuda()
            labels = labels.cuda()
            outputs = new_model(images)
            _, predicted = torch.max(outputs, 1) # top 1 기준
            total += labels.size(0) # labels.size : batch size가 64이니 64이다. (맨 끝에만 16개)
            correct_top1 += (predicted == labels).sum().item()
            print("step : {} / {}".format(idx + 1, len(testset)/int(labels.size(0))))
            print("top-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
    file.write("error channel {0}~{1}, epoch : [{2}/{3}]\n".format(loss_start_index,loss_start_index+128-1, epoch+1, num_epochs))        
    file.write("top-1 percentage :  {0:0.2f}%\n".format(correct_top1 / total * 100))
    if (correct_top1 / total * 100) < before_accuracy:
        optimizer = torch.optim.SGD(F4F.parameters(), lr=before_lr*0.5, weight_decay=1e-4) # optimizer (아까 뽑아낸 훈련시킬 것만 설정)  => 안쓰는 것들은 자동으로 freeze
        before_lr=before_lr*0.5
    before_accuracy=(correct_top1 / total * 100)
        
    ###### 새로운 filter 저장 #######
    for name, parameter in new_model.named_parameters():
        if name == 'features.34.weight': # 훈련시킬 것만 뽑는다.
            epoch_num=str(epoch+1).zfill(2)
            torch.save(parameter, f"/media/3/Network/filter/pooling4/only_error_input_F4F/conv5_1_train_epoch_{epoch_num}.pt") # 예를들어 128~255이면 128~255번째 channel이 깨진 것이다. 그리고 뒤의 숫자는 학습 횟수이다. 7이면 7번 epoch 돌린것
            break
        
    ###### F4F 의 weight 저장 ######
    for name, parameter in F4F.named_parameters():
        if name=='fc1.weight':
            epoch_num=str(epoch+1).zfill(2)
            torch.save(parameter, f"/media/3/Network/filter/pooling4/F4F_weight/F4F_weight_train_epoch_{epoch_num}.pt") # 예를들어 128~255이면 128~255번째 channel이 깨진 것이다. 그리고 뒤의 숫자는 학습 횟수이다. 7이면 7번 epoch 돌린것
            continue
        if name=='fc1.bias':
            epoch_num=str(epoch+1).zfill(2)
            torch.save(parameter, f"/media/3/Network/filter/pooling4/F4F_weight/F4F_bias_train_epoch_{epoch_num}.pt") # 예를들어 128~255이>면 128~255번째 channel이 깨진 것이다. 그리고 뒤의 숫자는 학습 횟수이다. 7이면 7번 epoch 돌린것
            break

file.close() # 파일 객체 닫기



