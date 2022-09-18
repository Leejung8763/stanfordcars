import copy, datetime, time, re, warnings, logging, argparse, os, random
import torch
from torch import optim
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings(action="ignore")
wkdir = "/data1/lj"

# random seed 고정
random_seed=1701010
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

#################
# argparse 설정 #
#################
parser = argparse.ArgumentParser(description="stanfordcars classification", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-m", "--model", required=False, default="mobilenet_v3_small", type=str, help="Training Model 선택")
parser.add_argument("-a", "--augmentation", required=False, default=False, type=int, help="Augmentation 수행 여부`")
parser.add_argument("-f", "--freeze", required=False, default=0, type=float, help="Layer Freeze 비율`")
parser.add_argument("-b", "--batch", required=False, default=64, type=int, help="Training Batch Size 설정")
parser.add_argument("-e", "--epoch", required=False, default=100, type=int, help="Training Epoch 설정")
parser.add_argument("-p", "--path", required=False, default=f"{wkdir}/cnn_model", type=str, help="Best Trained Model 저장경로")
parser.add_argument("-o", "--output", required=False, type=str, help="Best Trained Model 저장명")
parser.add_argument("-c", "--cuda", required=False, default=f"0", type=str, help="Graphic Card number")

args = parser.parse_args()

#################
# log 생성 코드 #
#################
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

logger = create_logger(path=f"{wkdir}/cnn_model/log/{args.output}_{datetime.datetime.now().strftime('%y%m%d')}.log", formatter="[%(asctime)s][%(levelname)s] >> %(message)s")

if __name__ == "__main__":
    try:
        tic = time.time()
        logger.info(f"[START] 데이터 세팅")

        #############################
        # 데이터 불러오기 및 핸들링 #
        #############################
        if args.augmentation == False:
            data_train = datasets.StanfordCars(
                root=f"{wkdir}/dataset/",
                split='train',
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((224,224)),
                    ToTensor()
                ])
            )
            data_test = datasets.StanfordCars(
                root=f"{wkdir}/dataset/",
                split='test',
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((224,224)), 
                    ToTensor()
                ])
            )
        else:
            data_train_add = datasets.StanfordCars(
                root=f"{wkdir}/dataset/",
                split='train',
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((224,224)),
                    ToTensor(),
                    transforms.RandomChoice([
                        transforms.CenterCrop((random.randint(100,200), random.randint(100,200))),
                        transforms.RandomHorizontalFlip(p=1),
                        transforms.RandomVerticalFlip(p=1),
                        transforms.RandomRotation(random.randint(1,359))
                    ]),
                    transforms.Resize((224,224))
                ])
            )
            data_test_add = datasets.StanfordCars(
                root=f"{wkdir}/dataset/",
                split='test',
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((224,224)),
                    ToTensor(),
                    transforms.RandomChoice([
                        transforms.CenterCrop((random.randint(100,200), random.randint(100,200))),
                        transforms.RandomHorizontalFlip(p=1),
                        transforms.RandomVerticalFlip(p=1),
                        transforms.RandomRotation(random.randint(1,359))
                    ]),
                    transforms.Resize((224,224))
                ])
            )
            # Concatenate dataset
            data_train = ConcatDataset([data_train, data_train_add])
            data_test = ConcatDataset([data_test, data_test_add])
        
        # Create Dataloader
        train_loader = DataLoader(data_train, batch_size=64, shuffle=True)
        test_loader = DataLoader(data_test, batch_size=64, shuffle=True)
    #     torch.save(train_loader, f'{wkdir}/dataset/stanford_cars/cars_train.dl')
    #     torch.save(test_loader, f'{wkdir}/dataset/stanford_cars/cars_test.dl')

        toc = time.time() - tic
        logger.info(f"[ END ] 데이터 세팅 | Augmentation: {False if args.augmentation==0 else True} | 소요시간: {int(toc//3600):02d}:{int(toc%3600//60):02d}:{int(toc%60):02d}")
        
        ########################
        # train model 불러오기 #
        ########################
        class model_setup(nn.Module):

            def __init__(self, model_name):
                super(model_setup, self).__init__()

                self.model = eval(f"models.{model_name}(pretrained=True)")
                    
                # classifier 1000개 클래스 196개로 변환
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
        device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
        model_cls = model_cls.to(device)
        
        ######################
        # model freeze setup #
        ######################
        block_cnt = 0
        block_cnt_tot = len(np.unique([re.split('weight|bias|fc', name)[0] for name, _ in model_cls.named_parameters()]))
        block_name = None

        for name, param in model_cls.named_parameters():
            # block count
            if block_name != re.split('weight|bias|fc', name)[0]:
                block_name = re.split('weight|bias|fc', name)[0]
                block_cnt += 1
            block_rate = np.round(block_cnt/block_cnt_tot, 2)
            # freeze level set
            if block_rate <= args.freeze:
                param.requires_grad=False
                
        toc = time.time() - tic
        logger.info(f"[ END ] 학습 모델 셋업 | Model: {args.model} & Freeze: {args.freeze}| 소요시간: {int(toc//3600):02d}:{int(toc%3600//60):02d}:{int(toc%60):02d}")
            
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
        logger.info(f"[ END ] Metric, Optimizer 셋업 | 소요시간: {int(toc//3600):02d}:{int(toc%3600//60):02d}:{int(toc%60):02d}")

        def train(model, params):

            num_epochs = params['num_epochs']
            loss_func = params["loss_func"]
            opt = params["optimizer"]
            train_dl = params["train_loader"]
            test_dl = params["test_loader"]
            sanity_check = params["sanity_check"]
            lr_scheduler = params["lr_scheduler"]

            loss_history = {'train_loss': [], 'val_loss': []}
            metric_history = {'train_acc': [], 'val_acc': []}
            time_history = {'time': []}

            # # GPU out of memoty error
            # best_model_wts = copy.deepcopy(model.state_dict())

            best_loss = float('inf')

            tic = time.time()
            logger.info(f"[START] {args.model} 학습")
            for epoch in range(num_epochs):
                
                tic_epoch = time.time()
                current_lr = get_lr(opt)
                logger.info(f'\t ㄴEpoch {epoch}/{num_epochs-1}, current lr={current_lr}')

                model.train()
                train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
                loss_history['train_loss'].append(train_loss)
                metric_history['train_acc'].append(train_metric)

                model.eval()
                with torch.no_grad():
                    val_loss, val_metric = loss_epoch(model, loss_func, test_dl, sanity_check)
                loss_history['val_loss'].append(val_loss)
                metric_history['val_acc'].append(val_metric)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                    torch.save(model.state_dict(), f'{args.path}/{args.output}.pt')
                    logger.info('\t   ㄴCopied best model weights!')
                    logger.info('\t   ㄴGet best val_loss')

                lr_scheduler.step(val_loss)
                
                toc_epoch = time.time() - tic_epoch
                time_history['time'].append(toc_epoch)
                logger.info(f'\t   ㄴtrain loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy: {100*val_metric:.2f}, time: {int(toc_epoch//3600):02d}:{int(toc_epoch%3600//60):02d}:{int(toc_epoch%60):02d}')
            
            train_hist = pd.concat([pd.DataFrame(loss_history), pd.DataFrame(metric_history), pd.DataFrame(time_history)], axis=1)
            
            toc = time.time() - tic
            logger.info(f"[ END ] {args.model} 학습 | 소요시간: {int(toc//3600):02d}:{int(toc%3600//60):02d}:{int(toc%60):02d}")
            # model.load_state_dict(best_model_wts)

            return model, train_hist

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

        model, train_hist = train(model_cls, params_train)
        train_hist.to_csv(f"{args.path}/{args.output}_train_hist.csv", index=False)
        
    except Exception as e:
        logger.error(f"{e}")