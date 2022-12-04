import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# set seeds


def get_models(_config):
    pretrained_weights_path = _config['pretrained_weights_path']
    try:
        state_dict = torch.load(pretrained_weights_path)
    except:
        imnet_model = torchvision.models.resnet50(pretrained=True)
        state_dict = imnet_model.state_dict()
        torch.save(state_dict, pretrained_weights_path)
    out_features, in_features = state_dict['fc.weight'].shape
    model = torchvision.models.resnet50(pretrained=False)
    old_model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(in_features, out_features)
    old_model.fc = torch.nn.Linear(in_features, out_features)
    model.load_state_dict(state_dict)
    old_model.load_state_dict(state_dict)
    old_model.eval()

    return model, old_model


def get_loss(_config):
    if _config['loss'] == "CrossEntropy":
        loss = nn.CrossEntropyLoss()
    return loss


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')


class CustomDataset(Dataset):
    def __init__(self, ds_type='train', data_path=None, imgs_list=None):
        self.ds_type = ds_type
        if ds_type == 'train':
            self.transforms = transforms.Compose([transforms.Resize(size=(224, 224)),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
                                                  transforms.RandomRotation(degrees=30),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.transforms = transforms.Compose([transforms.Resize(size=(224, 224)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.data_path = data_path  # path to dir with images
        self.class_name = self.data_path.split('/')[-1]
        self.images_names = imgs_list

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, index):
        img_name = self.images_names[index]
        img_path = os.path.join(self.data_path, img_name)
        label = torch.tensor(0)
        img = Image.open(img_path)
        img = self.transforms(img)
        return img, label


def get_dataloader(train_imgs_list, test_imgs_list, _config):
    train_dataset = CustomDataset(ds_type='train', data_path=_config['data_path'], imgs_list=train_imgs_list)
    test_dataset = CustomDataset(ds_type='test', data_path=_config['data_path'], imgs_list=test_imgs_list)
    train_loader = DataLoader(
        train_dataset, batch_size=_config['batch_size'], num_workers=_config['num_workers'], shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=_config['batch_size'],
                            num_workers=_config['num_workers'], shuffle=False)
    return train_loader, val_loader


def prepare_model(_config):
    num_new_class = _config['num_new_class']
    device = _config['device']
    model, old_model = get_models(_config)
    in_features = model.fc.in_features
    out_features = model.fc.out_features
    weight = model.fc.weight.data
    bias = model.fc.bias.data
    new_out_features = out_features + num_new_class
    new_fc = nn.Linear(in_features, new_out_features)
    kaiming_normal_init(new_fc)
    new_fc.weight.data[:out_features] = weight
    new_fc.bias.data[:out_features] = bias
    model.fc = new_fc

    # freeze backbone
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    for param in old_model.parameters():
        param.requires_grad = False

    model = model.to(device)
    old_model = old_model.to(device)
    cudnn.benchmark = True
    return model, old_model


def get_optimizer(model, _config):
    if _config['optimizer'] == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=_config['lr'],
                              momentum=_config['momentum'], weight_decay=_config['weight_decay'])
    if _config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=_config['lr'], weight_decay=_config['weight_decay'])

    return optimizer


def train_one_epoch(epoch, model, old_model, train_loader, optimizer, criterion, _config):
    model.eval()
    train_loss = 0.0
    correct = 0
    total = 0
    device = _config['device']
    T = _config['T']
    alpha = _config['alpha']
    out_features = old_model.fc.out_features
    pbar = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        targets += out_features
        optimizer.zero_grad()
        outputs = model(inputs)
        soft_target = old_model(inputs)
        loss1 = criterion(outputs, targets)

        outputs_S = F.softmax(outputs[:, :out_features]/T, dim=1)
        outputs_T = F.softmax(soft_target[:, :out_features]/T, dim=1)

        loss2 = outputs_T.mul(-1*torch.log(outputs_S))
        loss2 = loss2.sum(1)
        loss2 = loss2.mean()*T*T
        loss = loss1*alpha+loss2*(1-alpha)
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        lr = optimizer.param_groups[0]['lr']
        pbar.set_description(
            f"Epoch: {epoch} | Loss: {train_loss / (batch_idx+1)} | Acc: {100. * correct /total} | LR: {lr}")
    if epoch == 10:
        optimizer.param_groups[0]['lr'] *= 0.1

    return train_loss / (batch_idx+1)


def eval_one_epoch(epoch, model, old_model, test_dataloader, criterion, _config, best_acc):
    device = _config['device']
    T = _config['T']
    alpha = _config['alpha']
    model.eval()
    out_features = old_model.fc.out_features
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_dataloader)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            targets += out_features
            outputs = model(inputs)
            soft_target = old_model(inputs)
            loss1 = criterion(outputs, targets)
            loss = loss1
            outputs_S = F.softmax(outputs[:, :out_features]/T, dim=1)
            outputs_T = F.softmax(soft_target[:, :out_features]/T, dim=1)
            loss2 = outputs_T.mul(-1*torch.log(outputs_S))
            loss2 = loss2.sum(1)
            loss2 = loss2.mean()*T*T
            loss = loss1*alpha+loss2*(1-alpha)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_description(
                f"Epoch: {epoch} | Test Loss: {test_loss / (batch_idx+1)} | Cur test Acc: {100. * correct /total}")

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        torch.save(model.state_dict(), 'pretrained_weights/net.pth')
        best_acc = acc
    return best_acc, acc


def create_markup(_config):
    imgs_list = os.listdir(_config['data_path'])
    n = len(imgs_list)
    train_len = int(0.8*n)
    random.shuffle(imgs_list)
    train_imgs_list = imgs_list[:train_len]
    test_imgs_list = imgs_list[train_len:]
    return train_imgs_list, test_imgs_list


def run_experiment(_config):
    best_acc = 0.0
    model, old_model = prepare_model(_config)
    train_imgs_list, test_imgs_list = create_markup(_config)
    train_dataloader, val_dataloader = get_dataloader(train_imgs_list, test_imgs_list, _config)
    optimizer = get_optimizer(model, _config)

    loss = get_loss(_config)
    for epoch in range(_config['num_epochs']):
        train_one_epoch(epoch, model, old_model, train_dataloader, optimizer, loss, _config)
        best_acc, res = eval_one_epoch(epoch, model, old_model, val_dataloader, loss, _config, best_acc)

        print("Test ACC: ", res)


if __name__ == "__main__":

    run_experiment(_config)
