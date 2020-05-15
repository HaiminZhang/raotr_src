from __future__ import print_function

import argparse
import csv
import os
import os.path
import shutil
import time
import math

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable
from tensorboardX import SummaryWriter

from PIL import Image

best_prec1 = 0




parser = argparse.ArgumentParser(description='Multi-level RCNN for Emotion classification')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true', help='evaluate model on test set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # Avgplloing layer
        self.avgpool1 = nn.AvgPool2d( 7,stride=1)
        self.avgpool2 = nn.AvgPool2d(14,stride=1)
        self.avgpool3 = nn.AvgPool2d(28,stride=1)
        self.avgpool4 = nn.AvgPool2d(56,stride=1)

        #fully connected layer
        self.classifier = nn.Linear(256, 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        # Avgpool
        p5 = self.avgpool1(p5)
        p4 = self.avgpool2(p4)
        p3 = self.avgpool3(p3)
        p2 = self.avgpool4(p2)
        #concate
       # p  = torch.cat((p2,p3,p4,p5),1)
        p  = p2.view(x.size(0),-1)
        #fully connect
        p  = self.classifier(p)
        return p
        #return p2, p3, p4, p5


def FPN101(pretrained=False):
    model = FPN(Bottleneck, [3,4,23,3])
    if pretrained:
        model_path = "/home/trao/Desktop/FPN/resnet101_caffe.pth"
        print("Loading pretrained weights from %s" % (model_path))
        pretrained_dict = torch.load(model_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        #print(model_dict)
        model.load_state_dict(model_dict)


        #model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})
        #model.load_state_dict({k: v for k, v in state_dict.items()})
    return model



'''
def test():
    net = FPN101(pretrained=True)
    #for param in net.parameters():
    #    print(param.size())
    fms = net(Variable(torch.randn(1,3,224,224)))
    print(fms.size())
    #for fm in fms:
    #    print(fm.size())


    #res101 = models.resnet101()
    #print(res101)
    #print('*' * 50)

    #params = net.state_dict()
    #for k, v in params.items():
    #    print(k)  # 打印网络中的变量名


   # params = list(net.parameters())
   # k = 0
   # for i in params:
   #    l = 1
   #    print("该层的结构：" + str(list(i.size())))
   #    for j in i.size():
   #         l *= j
   #     print("该层参数和：" + str(l))
   #     k = k + l
#print("总参数数量和：" + str(k))

    #for x, y in enumerate(FPN101.named_parameters()):
     #   print('layer num:', x, ('-' * 5), 'layer name:', y[0], ('-' * 5), 'layer size:', y[1].size())

test()
'''


def main():

    global args, best_prec1
    args = parser.parse_args()

    writer = SummaryWriter('./keep/adam_final')

    model = FPN101(pretrained=True)
    model = torch.nn.DataParallel(model).cuda()

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    testdir = os.path.join(args.data, 'test')


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = data.DataLoader(
        datasets.ImageFolder(traindir,
                             transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    val_loader = data.DataLoader(
        datasets.ImageFolder(valdir,
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    test_loader = data.DataLoader(
        TestImageFolder(testdir,
                        transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ])),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False)

    if args.test:
        print("Testing the model and generating a output csv for submission")
        test(test_loader, model)
        return
    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    #optimizer = optim.Adam(model.module.fc.parameters(), args.lr, weight_decay=args.weight_decay) #resnet
    optimizer = optim.Adam([
        {'params': model.module.conv1.parameters()},
        {'params': model.module.layer1.parameters()},
        {'params': model.module.layer2.parameters()},
        {'params': model.module.layer3.parameters()},
        {'params': model.module.layer4.parameters()},
        {'params': model.module.toplayer.parameters()},
        {'params': model.module.smooth1.parameters()},
        {'params': model.module.smooth2.parameters()},
        {'params': model.module.smooth3.parameters()},
        {'params': model.module.latlayer1.parameters()},
        {'params': model.module.latlayer2.parameters()},
        {'params': model.module.latlayer3.parameters()},
        #{'params': model.module.avgpool1.parameters()},
        #{'params': model.module.avgpool2.parameters()},
       # {'params': model.module.avgpool3.parameters()},
        {'params': model.module.avgpool4.parameters()},
        #{'params': model.module.classifier.parameters()}
        {'params': model.module.classifier.parameters(), 'lr': args.lr * 10}
                               ], args.lr, weight_decay=args.weight_decay)

    #optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay) #SGD

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, prec1 = validate(val_loader, model, criterion)

        writer.add_scalar('train_loss', train_loss,  epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', prec1, epoch)

        # remember best Accuracy and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)

    writer.close()

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        image_var = torch.autograd.Variable(images)
        label_var = torch.autograd.Variable(target)

        # compute y_pred
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # measure accuracy and record loss
        prec1, prec1 = accuracy(y_pred.data, target, topk=(1, 1))
        losses.update(loss.data[0], images.size(0))
        acc.update(prec1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=acc))

    return losses.avg, acc.avg

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, labels) in enumerate(val_loader):
        labels = labels.cuda(async=True)
        image_var = torch.autograd.Variable(images, volatile=True)
        label_var = torch.autograd.Variable(labels, volatile=True)

        # compute y_pred
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # measure accuracy and record loss
        prec1, temp_var = accuracy(y_pred.data, labels, topk=(1, 1))
        losses.update(loss.data[0], images.size(0))
        acc.update(prec1[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('TrainVal: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))

    print(' * Accuracy {acc.avg:.3f}'.format(acc=acc))

    return losses.avg, acc.avg


def test(test_loader, model):
    csv_map = {}
    # switch to evaluate mode
    model.eval()
    for i, (images, filepath) in enumerate(test_loader):
        # pop extension, treat as id to map
        filepath = os.path.splitext(os.path.basename(filepath[0]))[0]
        filepath = int(filepath)

        image_var = torch.autograd.Variable(images, volatile=True)
        y_pred = model(image_var)
        # get the index of the max log-probability
        smax = nn.Softmax()
        smax_out = smax(y_pred)[0]
        cat_prob = smax_out.data[0]
        dog_prob = smax_out.data[1]
        prob = dog_prob
        if cat_prob > dog_prob:
            prob = 1 - cat_prob
        prob = np.around(prob, decimals=4)
        prob = np.clip(prob, .0001, .999)
        csv_map[filepath] = prob
        # print("{},{}".format(filepath, prob))

    with open(os.path.join(args.data, 'entry.csv'), 'wb') as csvfile:
        fieldnames = ['id', 'label']
        csv_w = csv.writer(csvfile)
        csv_w.writerow(('id', 'label'))
        for row in sorted(csv_map.items()):
            csv_w.writerow(row)

    return


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 10))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class TestImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        images = []
        for filename in os.listdir(root):
            if filename.endswith('jpg'):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    main()