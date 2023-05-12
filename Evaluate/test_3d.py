import os
import sys
import time
import argparse
import re
from tqdm import tqdm

sys.path.append('../Utils')
sys.path.append('../Backbone')
from Evaluate.dataset_3d_lc import UCF101_3d, HMDB51_3d
from Evaluate.model_3d_lc import *
from Utils.augmentation import *
from Utils.utils import AverageMeter, ConfusionMeter, save_checkpoint, write_log, calc_topk_accuracy, denorm, \
    calc_accuracy, neq_load_customized

import torch
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
from torchvision import datasets, models, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='uniformer_small', type=str)
parser.add_argument('--model', default='CVC', type=str)
parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument('--split', default=9, type=int)
parser.add_argument('--num_class', default=101, type=int)
parser.add_argument('--num_seq', default=8, type=int, help='number of video frames')
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model to initiate test training')
parser.add_argument('--test', default='', type=str, help='path of model for final classification test')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=1, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--print_freq', default=50, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='last', type=str)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--representation_dim', default=512, type=int, help='dimension of final representation vector')


def main():
    global args;
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda;
    cuda = torch.device('cuda')

    if args.dataset == 'k400':
        args.num_class = 400
    elif args.dataset == 'ucf101':
        args.num_class = 101
    elif args.dataset == 'hmdb51':
        args.num_class = 51

    ### classifier model ###
    if args.model == 'CVC':
        model = CVC_lc(sample_size=args.img_dim,
                       network=args.net,
                       num_class=args.num_class,
                       train_what=args.train_what)
    else:
        raise ValueError('wrong model!')

    model = model.to(cuda)

    global criterion;
    criterion = nn.CrossEntropyLoss()

    ### optimizer ### 
    params = None
    # ONLY USE THIS CODE ONCE
    if args.train_what == 'ft':
        print('=> finetune backbone with smaller lr')
        params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                params.append({'params': param, 'lr': args.lr / 10})
            else:
                params.append({'params': param})
    # Freeze pretrained network, only train with linear probe
    elif args.train_what == 'last':
        print('=> linear probe - freezing the network and only train a linear classifier')
        params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            else:
                params.append({'params': param})

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))

    if params is None: params = model.parameters()

    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wd, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)

    args.old_lr = None
    best_acc = 0
    global iteration;
    iteration = 0

    ### restart training ###
    if args.test:
        if os.path.isfile(args.test):
            print("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                print('=> [Warning]: weight structure is not equal to test model; Use non-equal load ==')
                model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded testing checkpoint '{}' (epoch {})".format(args.test, checkpoint['epoch']))
            global num_epoch;
            num_epoch = checkpoint['epoch']
        elif args.test == 'random':
            print("=> [Warning] loaded random weights")
        else:
            raise ValueError()

        transform = transforms.Compose([
            RandomSizedCrop(consistent=True, size=224, p=0.0),
            GroupScale(size=(args.img_dim, args.img_dim)),
            ToTensor(),
            Normalize()
        ])
        test_loader = get_data(transform, 'test')
        test_loss, test_acc = test(test_loader, model)
        sys.exit()
    else:  # not test
        torch.backends.cudnn.benchmark = True

    if args.resume:
        if os.path.isfile(args.resume):
            args.old_lr = float(re.search('_lr(.+?)_', args.resume).group(1))
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.reset_lr:  # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('==== Change lr from %f to %f ====' % (args.old_lr, args.lr))
            iteration = checkpoint['iteration']
            print(
                "=> loaded resumed checkpoint '{}' (epoch {}) with best_acc {}".format(args.resume, checkpoint['epoch'],
                                                                                       best_acc))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if (not args.resume) and args.pretrain:
        if args.pretrain == 'random':
            print('=> using random weights')
        elif os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            model = neq_load_customized(model, checkpoint['state_dict'])
            # model = test_load(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    ### load data ###
    transform = transforms.Compose([
        RandomSizedCrop(consistent=True, size=224, p=1.0),
        GroupScale(size=(args.img_dim, args.img_dim)),
        RandomHorizontalFlip(consistent=True),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.3, consistent=True),
        ToTensor(),
        Normalize()
    ])
    val_transform = transforms.Compose([
        RandomSizedCrop(consistent=True, size=224, p=0.3),
        GroupScale(size=(args.img_dim, args.img_dim)),
        RandomHorizontalFlip(consistent=True),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3, consistent=True),
        ToTensor(),
        Normalize()
    ])

    train_loader = get_data(transform, 'train')
    val_loader = get_data(val_transform, 'val')

    # setup tools
    global img_path;
    img_path, model_path = set_path(args)

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):

        train_loss, train_acc = train(train_loader, model, optimizer, epoch)

        scheduler.step()
        print('\t Epoch: ', epoch, 'with lr: ', scheduler.get_last_lr())

        if epoch % 5 == 0:
            val_loss, val_acc = validate(val_loader, model)
            # save check_point
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'net': args.net,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'iteration': iteration
            }, is_best, gap=5, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch + 1)), keep_all=False)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))


def train(data_loader, model, optimizer, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    global iteration
    model.train()

    for idx, (input_seq, target) in enumerate(data_loader):
        tic = time.time()
        input_seq = input_seq.to(cuda, non_blocking=True)
        target = target.to(cuda, non_blocking=True)
        B = input_seq.size(0)
        output = model(input_seq)

        del input_seq

        target = target.view(-1)

        loss = criterion(output, target)
        acc = calc_accuracy(output, target)

        del target

        losses.update(loss.item(), B)
        accuracy.update(acc.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.local_avg:.4f})\t'
                  'Acc: {acc.val:.4f} ({acc.local_avg:.4f}) T:{3:.2f}\t'.format(
                epoch, idx, len(data_loader), time.time() - tic,
                loss=losses, acc=accuracy))

            iteration += 1

    return losses.local_avg, accuracy.local_avg


def validate(data_loader, model):
    losses = AverageMeter()
    accuracy = AverageMeter()
    model.eval()
    with torch.no_grad():
        for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda, non_blocking=True)
            target = target.to(cuda, non_blocking=True)
            B = input_seq.size(0)
            output = model(input_seq)

            target = target.view(-1)

            loss = criterion(output, target)
            acc = calc_accuracy(output, target)

            losses.update(loss.item(), B)
            accuracy.update(acc.item(), B)

    print('Loss {loss.avg:.4f}\t'
          'Acc: {acc.avg:.4f} \t'.format(loss=losses, acc=accuracy))
    return losses.avg, accuracy.avg


def test(data_loader, model):
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top3 = AverageMeter()
    confusion_mat = ConfusionMeter(args.num_class)
    model.eval()
    count = 0
    with torch.no_grad():
        for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda)
            target = target.to(cuda)
            B = input_seq.size(0)
            # squeeze the '1' batch dim, because in 'test' mode batchSize is set to 1
            input_seq = input_seq.squeeze(0)

            output = model(input_seq)  # output = (B, 1, Class_num)

            if target.squeeze(0) == torch.argmax(
                        torch.mean(nn.functional.softmax(output, 1), 0)):
                count += 1

            del input_seq

            top1, top3 = calc_topk_accuracy(torch.mean(nn.functional.softmax(output, 1), 0, keepdim=True), target, (1, 3))
            acc_top1.update(top1.item(), B)
            acc_top3.update(top3.item(), B)
            del top1, top3

            output = torch.mean(output, 0, keepdim=True)
            loss = criterion(output, target.squeeze(-1))

            losses.update(loss.item(), B)
            del loss

            _, pred = torch.max(output, 1)
            confusion_mat.update(pred, target.view(-1).byte())

    print("************count == ", count)
    print('Loss {loss.avg:.4f}\t Acc top1: {top1.avg:.4f} Acc top3: {top3.avg:.4f} \t'.format(loss=losses, top1=acc_top1, top3=acc_top3))
    confusion_mat.plot_mat(args.test + '.svg')
    write_log(content='Loss {loss.avg:.4f}\t Acc top1: {top1.avg:.4f} Acc top3: {top3.avg:.4f} \t'.format(loss=losses,
                                                                                                          top1=acc_top1,
                                                                                                          top3=acc_top3,
                                                                                                          args=args),
              epoch=num_epoch,
              filename=os.path.join(os.path.dirname(args.test), 'test_log.md'))
    import ipdb;
    ipdb.set_trace()
    return losses.avg, [acc_top1.avg, acc_top3.avg]


def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)
    if args.dataset == 'k400':
        pass
    elif args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                            transform=transform,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            which_split=args.split)
    elif args.dataset == 'hmdb51':
        dataset = HMDB51_3d(mode=mode,
                            transform=transform,
                            num_seq=args.num_seq,
                            downsample=args.ds,
                            which_split=args.split)
    else:
        raise ValueError('dataset not supported')

    sampler = data.RandomSampler(dataset)
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=16,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=16,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=2,
                                      pin_memory=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}-\
sp{args.split}_{0}_{args.model}_bs{args.batch_size}_lr{1}_seq{args.num_seq}_train-{args.train_what}{2}'.format(
            '%s' % args.net[10::], \
            args.old_lr if args.old_lr is not None else args.lr, \
            '_best_epoch' if args.pretrain else '', \
            args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path


if __name__ == '__main__':
    main()
