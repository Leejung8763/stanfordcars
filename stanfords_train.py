import copy, time, re
import torch
from torch import optim
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

##############
# 데이터 저장 #
##############
data_train = datasets.StanfordCars(
    root="/data1/lj/dataset/",
    split='train',
    download=True,
    transform=transforms.Compose([transforms.Resize((224,224)), ToTensor()])
)

data_test = datasets.StanfordCars(
    root="/data1/lj/dataset/",
    split='test',
    download=True,
    transform=transforms.Compose([transforms.Resize((224,224)), ToTensor()])
)

################
# 데이터 표준화 #
################
# To normalize the dataset, calculate the mean and std
meanRGB_train = [np.mean(x.numpy(), axis=(1,2)) for x, _ in data_train]
stdRGB_train = [np.std(x.numpy(), axis=(1,2)) for x, _ in data_test]
meanR_train = np.mean([m[0] for m in meanRGB_train])
meanG_train = np.mean([m[1] for m in meanRGB_train])
meanB_train = np.mean([m[2] for m in meanRGB_train])
stdR_train = np.mean([s[0] for s in stdRGB_train])
stdG_train = np.mean([s[1] for s in stdRGB_train])
stdB_train = np.mean([s[2] for s in stdRGB_train])

# To normalize the dataset, calculate the mean and std
meanRGB_test = [np.mean(x.numpy(), axis=(1,2)) for x, _ in data_test]
stdRGB_test = [np.std(x.numpy(), axis=(1,2)) for x, _ in data_test]
meanR_test = np.mean([m[0] for m in meanRGB_test])
meanG_test = np.mean([m[1] for m in meanRGB_test])
meanB_test = np.mean([m[2] for m in meanRGB_test])
stdR_test = np.mean([s[0] for s in stdRGB_test])
stdG_test = np.mean([s[1] for s in stdRGB_test])
stdB_test = np.mean([s[2] for s in stdRGB_test])

# Transformation Setup
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize([meanR_train, meanG_train, meanB_train]
                         ,[stdR_train, stdG_train, stdB_train]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize([meanR_test, meanG_test, meanB_test]
                         ,[stdR_test, stdG_test, stdB_test]),
])

# apply transforamtion
data_train.transform = transform_train
data_test.transform = transform_test

# create DataLoader
train_loader = DataLoader(data_train, batch_size=32, shuffle=True)
test_loader = DataLoader(data_test, batch_size=32, shuffle=True)


class model_setup(nn.Module):
    
    def __init__(self, model_name):
        super(model_setup, self).__init__()

        self.model = eval(f"models.{model_name}(pretrained=True)")
        
        # 1000개 클래스 196개로 변환
        if "resnet" in model_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, 196)
        elif "googlenet" in model_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, 196)
        elif "densenet" in model_name:
            self.model.classifier = nn.Linear(self.model.classifier.in_features, 196)
        elif "mobilenet_v2" in model_name:
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 196)
        elif "mobilenet_v3" in model_name:
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 196)
        
    def forward(self, x):
        return self.model(x)

resnet = model_setup("resnet50")
googlenet = model_setup("googlenet")
densenet = model_setup("densenet121")
mobilenetv2 = model_setup("mobilenet_v2")
mobilenetv3 = model_setup("mobilenet_v3_small")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_rs = resnet.to(device)
model_gg = googlenet.to(device)
model_ds = densenet.to(device)
model_m2 = mobilenetv2.to(device)
model_m3 = mobilenetv3.to(device)

loss_func = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model_m3.parameters(), lr=0.001)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

# function to get current lr
def get_lr(opt):
    
    for param_group in opt.param_groups:
        return param_group['lr']
    
# function to calculate metric per mini-batch
def metric_batch(output, target):
    
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

# function to calculate loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    
    loss = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

# function to calculate loss and metric per epoch
def loss_epoch(model, loss_func, data_dataloader, sanity_check=False, opt=None):
    
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(data_dataloader.dataset)

    for x_b, y_b in data_dataloader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)
        output = model(x_b)

        loss_b, metric_b = loss_batch(loss_func, output, y_b, opt)

        running_loss += loss_b
        
        if metric_b is not None:
            running_metric += metric_b
        
        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric

def train(model, params):
    
    num_epochs = params['num_epochs']
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_loader"]
    test_dl = params["test_loader"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    # # GPU out of memoty error
    # best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = float('inf')

    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print(f'Epoch {epoch}/{num_epochs-1}, current lr={current_lr}')

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, test_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), '/data1/lj/cnn_model/mobilenetv3/mobilev3.pt')
            print('Copied best model weights!')
            print('Get best val_loss')

        lr_scheduler.step(val_loss)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
        print('-'*10)

    # model.load_state_dict(best_model_wts)

    return model, loss_history, metric_history

# define the training parameters
params_train = {
    'num_epochs':20,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_loader':train_loader,
    'test_loader':test_loader,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
}

model, loss_hist, metric_hist = train(model_m3, params_train)