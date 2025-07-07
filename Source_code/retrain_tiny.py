import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pickle

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import (get_model_params, BlockDecoder)
from torch.utils.data import DataLoader, Subset
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'src')
sys.path.append(srcFolder)

from src.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from src import regressor

from PIL import Image
import numpy as np
import torch.nn.functional as F
import random
import pickle

state = {
    "lr": 1e-3,
    "bsz": 64,
    "val_bsz": 1536,
    "schedule": [10, 20],
    "gamma": 0.1,
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
#处理TinyImageNet图像
def load_allimages(dir):
    images = []
    if not os.path.isdir(dir):
        sys.exit(-1)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            #if datasets.folder.is_image_file(fname):
            if datasets.folder.has_file_allowed_extension(fname,('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images

def adjust_learning_rate_two(optimizer, optimizer1, epoch):
    global state
    if epoch in state['schedule']:
        state['lr'] *= state['gamma']
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
        for param_group in optimizer1.param_groups:
            param_group['lr'] = state['lr']

def parseClasses(file):
    classes = []
    filenames = []
    with open(file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for x in range(0,len(lines)):
        tokens = lines[x].split()
        classes.append(tokens[1])
        filenames.append(tokens[0])
    return filenames,classes

class TImgNetDataset(data.Dataset):
    """Dataset wrapping images and ground truths."""
    
    def __init__(self, img_path, gt_path, class_to_idx , transform=None):
        self.img_path = img_path
        self.transform = transform
        self.gt_path = gt_path
        self.class_to_idx = class_to_idx
        self.classidx = []
        self.imgs, self.classnames = parseClasses(gt_path)
        for classname in self.classnames:
            self.classidx.append(self.class_to_idx[classname])

    def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, y) where y is the label of the image.
            """
            img = None
            with open(os.path.join(self.img_path, self.imgs[index]), 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
            y = self.classidx[index]
            return img, y

    def __len__(self):
        return len(self.imgs)


# Use CUDA
use_cuda = torch.cuda.is_available()

if __name__ == '__main__':
    import sys
    data_path = sys.argv[1]
    output_path = sys.argv[2]

    print("load data")
    path = str(data_path)+'/tiny-imagenet-200'
    subset_file_path = {
        "SETS": str(data_path)+"/SETS/tinyimagenet_resnet101_{size}.pkl",
        "DeepGD": str(data_path)+"/DeepGD/tinyimagenet_resnet101_{size}.txt"
    }
    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #training
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.ImageFolder(traindir, transform)
    train_set = dataset

    print("training set load finish")

    #testing
    valdir = os.path.join(path, 'val', 'images')
    valgtfile = os.path.join(path, 'val', 'val_annotations.txt')
    val_dataset_forall = TImgNetDataset(valdir, valgtfile, class_to_idx=dataset.class_to_idx.copy(),
            transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
            ]))

    elements = list(range(10000))
    f_name = str(data_path)+"/tinyimagenet_resnet101.pkl"
    with open(f_name, 'rb') as f: #get the testing set T
        test_indices = pickle.load(f)
    val_indices = list(set(elements) - set(test_indices))
    val_dataset = Subset(val_dataset_forall, val_indices)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=state['val_bsz'], shuffle=False,
        num_workers=32, pin_memory=True)


    num_epochs = 10


    ori_ac = 0.8270
    ori_prec1 = 84.6939
    # set_seed(42)
    # arch = 'resnet101'
    # pretrained_model = '/home/nfs03/laizj/wjl/ETS/model_best.pth.tar'
    # model = models.__dict__[arch]()
    # model,classifier = decomposeModel(model, 200, keep_pre_pooling=True)
    # model = torch.nn.DataParallel(model).cuda()
    # classifier = classifier.cuda()
    # optimizer = torch.optim.SGD(model.parameters(), state['lr'])
    # optimizer_reg = optim.SGD(classifier.parameters(), state['lr'])
    # reg_net = regressor.Net(classifier, optimizer_reg,
    #                         ref_size=1,
    #                         backendtype=arch,
    #                         dcl_offset=0,
    #                         dcl_window=1,
    #                         QP_margin=0.5)
    # checkpoint = torch.load(pretrained_model)
    # #print(checkpoint.keys())
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # reg_net.load_state_dict(checkpoint['classifier_state_dict'])
    # print("加载成功")
    # criterion = nn.CrossEntropyLoss().cuda()
    # model.eval()
    # correct = 0
    # total = 0
    # for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, total=len(val_loader))):
    #     if use_cuda:
    #         inputs, targets = inputs.cuda(), targets.cuda()
    #     with torch.no_grad():
    #         inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
    #         features = model(inputs)
    #         features = features.squeeze(-1).squeeze(-1)
    #         outputs = classifier(features)
    #         prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
    #         _, predicted = torch.max(outputs, 1)
    #         total += targets.size(0)
    #         correct += (predicted == targets).sum().item()
    # acc = correct / total

    results = {"DeepGD": {}, "SETS": {}}


    for size in [500]:

        best_acc_re_list = []
        best_acc_imp_list = []

        #这边两个文件不一样了，不同放在一起处理
        for model_name in ["DeepGD", "SETS"]:
            if model_name == "DeepGD":
                file_name = subset_file_path[model_name].format(size=size)
                with open(file_name, "r") as f:
                    lines = f.readlines()
                subset_list = []
                in_subset_list = False
                subset_data = ""
                for line in lines:
                    if "Subset List:" in line:
                        in_subset_list = True
                        continue
                    if in_subset_list:
                        subset_data += line.strip() + " "
                        if "]" in line:
                            subset_data = subset_data.replace("[", "").replace("]", "").strip()
                            subset_data_cleaned = subset_data.replace(",", "")
                            subset_list.append(list(map(int, subset_data_cleaned.split())))
                            subset_data = ""
            else:
                file_name = subset_file_path[model_name].format(size=size)
                with open(file_name, 'rb') as f:
                    subset_list = pickle.load(f)
                subset_list = [subset_list] * 1


            for idx, subset in enumerate(subset_list):
                print('\n\n')
                print("=" * 30)
                print(f"loading test data from size {size}, subset top_{idx}")
                extra_train_set = Subset(val_dataset_forall, subset)
                train_loader = DataLoader(
                    train_set + extra_train_set,
                    batch_size=state['bsz'],
                    shuffle=True,
                    num_workers=32,
                    pin_memory=True
                )


                set_seed(42)
                arch = 'resnet101'
                pretrained_model = str(data_path)+'/model_best.pth.tar'
                model = models.__dict__[arch]()
                model,classifier = decomposeModel(model, 200, keep_pre_pooling=True)
                model = torch.nn.DataParallel(model).cuda()
                classifier = classifier.cuda()

                optimizer = torch.optim.SGD(model.parameters(), state['lr'])
                optimizer_reg = optim.SGD(classifier.parameters(), state['lr'])

                reg_net = regressor.Net(classifier, optimizer_reg,
                                        ref_size=1,
                                        backendtype=arch,
                                        dcl_offset=0,
                                        dcl_window=1,
                                        QP_margin=0.5)
                checkpoint = torch.load(pretrained_model)
                #print(checkpoint.keys())
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                reg_net.load_state_dict(checkpoint['classifier_state_dict'])

                print("load successfully")

                criterion = nn.CrossEntropyLoss().cuda()

                acc_list = []

                for epoch in range(num_epochs):
                    model.train()

                    adjust_learning_rate_two(optimizer, optimizer_reg, epoch)
                    running_loss = 0.0
                    epoch_loss = []

                    for batch_idx, (inputs, targets) in enumerate(train_loader):
                        inputs, targets = inputs.cuda(), targets.cuda()

                        optimizer.zero_grad()
                        optimizer_reg.zero_grad()

                        outputs = model(inputs)
                        features = outputs.squeeze(-1).squeeze(-1)
                        outputs = classifier(features)

                        targets = targets.long()
                        loss = criterion(outputs, targets)

                        loss.backward()
                        optimizer.step()
                        optimizer_reg.step()

                        running_loss += loss.item()
                        epoch_loss.append(loss.item())


                        if (batch_idx + 1) % 50 == 0:
                            print(f"Subset {idx}, Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")


                    print(f"Subset {idx}, Epoch [{epoch+1}/{num_epochs}],Loss: {sum(epoch_loss) / len(epoch_loss):.4f}")

                    model.eval()
                    correct = 0
                    total = 0
                    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, total=len(val_loader))):
                        if use_cuda:
                            inputs, targets = inputs.cuda(), targets.cuda()
                        with torch.no_grad():
                            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                            features = model(inputs)
                            features = features.squeeze(-1).squeeze(-1)
                            outputs = classifier(features)
                            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                            _, predicted = torch.max(outputs, 1)
                            total += targets.size(0)
                            correct += (predicted == targets).sum().item()

                    acc = correct / total
                    acc_list.append((acc-ori_ac, prec1 - ori_prec1))
                    print(f"Validation Accuracy for Epoch {epoch}: {acc:.4f}")
                    print(f"Validation Accuracy for Epoch {epoch}: {prec1:.4f}")


                acc = correct / total
                best_acc_re_list.append(max(acc_list)[0])
                best_acc_imp_list.append(max(acc_list)[1])


            results[model_name]["acc_re_list"] = best_acc_re_list
            results[model_name]["acc_imp_list"] = best_acc_imp_list


        file_path = str(output_path)+"/results/tinyimagenet_resnet101_{size}.txt"
        with open(file_path, "w") as f:
            f.write(f"Original Model Accuracy: 0.8311\n")

            f.write("SETS Retraining\n")
            f.write("Acc Re List:\n")
            f.write("\n".join([f"{x:.4f}" for x in results["SETS"]["acc_re_list"]]) + "\n")
            f.write("Acc Imp List:\n")
            f.write("\n".join([f"{x:.4f}" for x in results["SETS"]["acc_imp_list"]]) + "\n")

            f.write("DeepGD Retraining\n")
            f.write("Acc Re List:\n")
            f.write("\n".join([f"{x:.4f}" for x in results["DeepGD"]["acc_re_list"]]) + "\n")

            f.write("Acc Imp List:\n")
            f.write("\n".join([f"{x:.4f}" for x in results["DeepGD"]["acc_imp_list"]]) + "\n")

        print("save successfully!\n")
    

    


