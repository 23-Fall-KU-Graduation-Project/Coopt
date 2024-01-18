import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import os
import copy
import argparse
from datetime import datetime

from models import *
from utility import *

class TradesAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = criterion_kl(F.log_softmax(self.model(x), dim=1),
                                       F.softmax(self.model(x_natural), dim=1))
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv

learning_rate = 0.05
epsilon = 8 / 255
k = 7
alpha = 0.003
file_name = 'trades_adversarial_training'

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

best_clean_val = 0
best_adv_val = 0

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

criterion_kl = nn.KLDivLoss(size_average=False)


parser = argparse.ArgumentParser()
parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
parser.add_argument("--threads", default=8, type=int, help="Number of CPU threads for dataloaders.")
parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
parser.add_argument("--bilevel",action='store_true', help="use bilevel optimization to do SAM+AT. default is false")
parser.add_argument("--trades",action="store_true",help="use trades")
parser.add_argument("--sgd", action='store_true', help="use sgd.")
parser.add_argument("--adam",action='store_true',help="use adam")
args = parser.parse_args()
initialize(args, seed=42)

def train(epoch):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        adv = adversary.perturb(inputs, targets)

        optimizer.zero_grad()

        adv_outputs = net(adv)

        loss_st = criterion(net(inputs), targets)

        loss_adv = criterion_kl(F.log_softmax(net(adv), dim=1), F.softmax(net(inputs), dim=1))

        

        loss = loss_st + loss_adv

        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = adv_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial train loss:', loss.item())

    print('\nTotal adversarial train accuarcy:', 100. * correct / total)
    print('Total adversarial train loss:', train_loss)

    writer.add_scalar('Adv_Acc/train', 100. * correct / total, epoch)
    writer.add_scalar('Adv_Loss/train', train_loss, epoch)

def test(epoch):
    global best_clean_val, best_adv_val
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            benign_loss += loss.item()

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print('\nCurrent batch:', str(batch_idx))
                print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current benign test loss:', loss.item())

            adv = adversary.perturb(inputs, targets)
            adv_outputs = net(adv)
            loss = criterion(adv_outputs, targets)
            adv_loss += loss.item()

            _, predicted = adv_outputs.max(1)
            adv_correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current adversarial test loss:', loss.item())

    writer.add_scalar('Clean_Acc/test', 100. * benign_correct / total, epoch)
    writer.add_scalar('Clean_Loss/test', benign_loss, epoch)

    writer.add_scalar('Adv_Acc/test', 100. * adv_correct / total, epoch)
    writer.add_scalar('Adv_Loss/test', adv_loss, epoch)

    state = {
        'net': net.state_dict()
    }

    if 100. * benign_correct / total > best_clean_val:
        torch.save(state, './checkpoint/trades_at/' + file_name + "_clean_best")
        print('Clean Best Model Saved!')
        best_clean_val = 100. * benign_correct / total

    if 100. * adv_correct / total > best_adv_val:
        torch.save(state, './checkpoint/trades_at/' + file_name + "_adv_best")
        print('Adv Best Model Saved!')
        best_adv_val = 100. * adv_correct / total


    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    if not os.path.isdir('checkpoint/trades_at'):
        os.mkdir('checkpoint/trades_at')

    

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

    time = datetime.now().strftime('%m-%d_%H%M%S')

    net = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    writer = SummaryWriter(log_dir = f"logs/trades_at/{time}") # directory for tensorboard logs
    log_dir = "./trades_at/best_checkpoint" # directory for model checkpoints

    net = net.to(device)
    # net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True

    adversary = TradesAttack(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

    for epoch in range(0, 200):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)
