import copy, datetime, time, re, warnings, logging, argparse, os
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

warnings.filterwarnings(action="ignore")
wkdir = "/data1/lj"
parser = argparse.ArgumentParser(description="stanfordcars classification", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-m", "--model", required=False, default="mobilenet_v3_small", type=str, help="Training Model 선택")
parser.add_argument("-e", "--epoch", required=False, default=500, type=int, help="Training Epoch 설정")
parser.add_argument("-p", "--path", required=False, default=f"{wkdir}/cnn_model", type=str, help="Best Trained Model 저장경로")
args = parser.parse_args()

def create_logger(path, formatter):
    
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(formatter)

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(path)

    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    
    logger.setLevel(logging.INFO)

    return logger

logger = create_logger(path=f"{wkdir}/cnn_model/log/{args.model}_{datetime.datetime.now().strftime('%y%m%d')}", formatter="[%(asctime)s][%(levelname)s] >> %(message)s")

if __name__ == "__main__":
    try:
        if len([name for name in os.listdir(f'{wkdir}/dataset/stanford_cars/') if '.dl' in name]) < 1:
            
            tic = time.time()
            logger.info(f"[START] 데이터 세팅")
            
            ##############
            # 데이터 저장 #
            ##############
            data_train = datasets.StanfordCars(
                root=f"{wkdir}/dataset/",
                split='train',
                download=True,
                transform=transforms.Compose([transforms.Resize((224,224)), ToTensor()])
            )

            data_test = datasets.StanfordCars(
                root=f"{wkdir}/dataset/",
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
            torch.save(train_loader, f'{wkdir}/dataset/stanford_cars/cars_train.dl')
            torch.save(test_loader, f'{wkdir}/dataset/stanford_cars/cars_test.dl')
            
            toc = time.time() - tic
            logger.info(f"[ END ] 데이터 세팅 | 소요시간: {int(toc//3600):02d}:{int(toc//3600%60):02d}:{int(toc%60):02d}")

        else:
            tic = time.time()
            logger.info(f"[START] 데이터 불러오기")
            
            train_loader = torch.load(f'{wkdir}/dataset/stanford_cars/cars_train.dl')
            test_loader = torch.load(f'{wkdir}/dataset/stanford_cars/cars_test.dl')
            
            toc = time.time() - tic
            logger.info(f"[ END ] 데이터 불러오기 | 소요시간: {int(toc//3600):02d}:{int(toc//3600%60):02d}:{int(toc%60):02d}")

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

        tic = time.time()
        logger.info(f"[START] 학습 모델 셋업")
        
        model_cls = model_setup(args.model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_cls = model_cls.to(device)
        
        toc = time.time() - tic
        logger.info(f"[ END ] 학습 모델 셋업 | 소요시간: {int(toc//3600):02d}:{int(toc//3600%60):02d}:{int(toc%60):02d}")
            
        loss_func = nn.CrossEntropyLoss(reduction='sum')
        opt = optim.Adam(model_cls.parameters(), lr=0.001)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)
        
        tic = time.time()
        logger.info(f"[START] Metric, Optimizer 셋업")
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
        
        toc = time.time() - tic
        logger.info(f"[ END ] Metric, Optimizer 셋업 | 소요시간: {int(toc//3600):02d}:{int(toc//3600%60):02d}:{int(toc%60):02d}")

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

            tic = time.time()
            logger.info(f"[START] 학습 시작")
            for epoch in range(num_epochs):
                
                tic_epoch = time.time()
                current_lr = get_lr(opt)
                logger.info(f'\t ㄴEpoch {epoch}/{num_epochs-1}, current lr={current_lr}')

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

                    torch.save(model.state_dict(), f'{args.path}/{args.model}.pt')
                    logger.info('\t   ㄴCopied best model weights!')
                    logger.info('\t   ㄴGet best val_loss')

                lr_scheduler.step(val_loss)
                
                toc = time.time() - tic_epoch
                logger.info(f'\t   ㄴtrain loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy: {100*val_metric:.2f}, time: {int(toc//3600):02d}:{int(toc//3600%60):02d}:{int(toc%60):02d}')
            
            toc = time.time() - tic
            logger.info(f"[ END ] Metric, Optimizer 셋업 | 소요시간: {int(toc//3600):02d}:{int(toc//3600%60):02d}:{int(toc%60):02d}")
            # model.load_state_dict(best_model_wts)

            return model, loss_history, metric_history

        # define the training parameters
        params_train = {
            'num_epochs':args.epoch,
            'optimizer':opt,
            'loss_func':loss_func,
            'train_loader':train_loader,
            'test_loader':test_loader,
            'sanity_check':False,
            'lr_scheduler':lr_scheduler,
        }

        model, loss_hist, metric_hist = train(model_cls, params_train)
        
    except Exception as e:
        logger.error(f"{e}")