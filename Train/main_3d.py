import sys
import time
import argparse
import matplotlib.pyplot as plt

plt.switch_backend('agg')

sys.path.append('../Utils')
from Train.dataset_3d import *
from Train.model_3d import *
from Train import contrastive_loss
from Utils.augmentation import *
from Utils.utils import AverageMeter, save_checkpoint, calc_topk_accuracy, neq_load_customized

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import transforms

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='uniformer_small', type=str)
parser.add_argument('--model', default='CVC', type=str)
parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument('--split', default=1, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--num_seq', default=8, type=int, help='number of video frames')
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=1, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--print_freq', default=50, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--cluster_num', default=101, type=int, help='number of clusters')
parser.add_argument('--memory_reset_epoch', default=-1, type=int, help='the epoch to reset memory bank')
parser.add_argument('--representation_dim', default=512, type=int, help='dimension of final representation vector')
parser.add_argument('--memory_setup_epoch', default=50, type=int, help='number of epochs for setting up memory bank')
parser.add_argument('--update_ratio', default=0.005, type=float, help='the update ratio of cluster memory feature')
parser.add_argument('--instance_temperature', default=1, type=float, help='temperature for instance CL')
parser.add_argument('--cluster_temperature', default=1, type=float, help='temperature for cluster CL')
parser.add_argument('--cluster_ratio', default=1, type=float, help='percentage of cluster CL contribution')
parser.add_argument('--cluster_theta', default=0.8, type=float, help='lower bound for Gathering')
parser.add_argument('--cluster_sigma', default=0.4, type=float, help='upper bound for Penalizing')


def main():
    torch.manual_seed(233)
    np.random.seed(0)
    global args;

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda;
    cuda = torch.device('cuda')

    if args.model == 'CVC':
        model = CVC_3d(sample_size=args.img_dim,
                       network=args.net,
                       representation_dim=args.representation_dim,
                       cluster_num=args.cluster_num,
                       with_aug=True)
        model_perturbed = CVC_3d(sample_size=args.img_dim,
                                 network=args.net + '_perturbed',
                                 representation_dim=args.representation_dim,
                                 cluster_num=args.cluster_num,
                                 with_aug=False)
    else:
        raise ValueError('wrong model!')

    model = model.to(cuda)
    model_perturbed = model_perturbed.to(cuda)

    global memory_bank;
    memory_bank = torch.zeros(args.cluster_num, args.representation_dim).to(cuda)

    '''
    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')
    '''
    if args.resume or args.pretrain:
        for name, param in model_perturbed.named_parameters():
            param.requires_grad = False

    params = model.parameters()
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wd, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    args.old_lr = None

    best_loss = 100

    ### restart training ###
    if args.resume:
        if os.path.isfile(args.resume):
            args.old_lr = args.lr
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            memory_bank = checkpoint['memory']
            memory_bank = memory_bank.to(cuda)
            model.load_state_dict(checkpoint['state_dict'])
            if not args.reset_lr:  # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
                # pass
            else:
                print('==== Change lr from %f to %f ====' % (args.old_lr, args.lr))
            print("=> loaded resumed checkpoint '{}' (epoch {}) with best_loss {}".format(args.resume, checkpoint['epoch'], best_loss))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))

    global memoryManager
    memoryManager = contrastive_loss.MemoryManager(memory_bank, args.update_ratio)

    global criterion_instance, criterion_entropy, criterion_cluster
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, cuda).to(cuda)
    criterion_entropy = contrastive_loss.EntropyLoss(args.batch_size, args.cluster_num).to(cuda)
    criterion_cluster = contrastive_loss.ClusterLoss(memory_bank, args.cluster_temperature).to(cuda)

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    ### load data ###
    if args.dataset == 'ucf101' or args.dataset == 'hmdb51':
        # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
        transform = transforms.Compose([
            GroupCenterCrop(size=224),
            GroupScale(int(args.img_dim)),
            ToTensor(),
            Normalize()
        ])
        transform_aug = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomCrop(size=128, consistent=True),
            GroupScale(int(64)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    elif args.dataset == 'k400':
        # designed for kinetics400, short size=144, rand crop to 128x128
        transform = transforms.Compose([
            GroupCenterCrop(size=224),
            GroupScale(int(args.img_dim)),
            ToTensor(),
            Normalize()
        ])
        transform_aug = transforms.Compose([
            RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            RandomHorizontalFlip(consistent=True),
            GroupScale(int(64)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])

    train_loader = get_data(transform, transform_aug, 'train')

    # setup tools
    global img_path;
    img_path, model_path = set_path(args)

    print('-- start main loop --')
    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):

        # update perturbed model every epoch
        params = model.state_dict()
        model_perturbed.load_state_dict(params)

        train_loss = train(train_loader, model, model_perturbed, optimizer, epoch)

        scheduler.step()
        print('\t Epoch: ', epoch, 'with lr: ', scheduler.get_last_lr())

        if epoch % 5 == 0:
            # save check_point
            is_best = train_loss < best_loss;
            best_loss = min(train_loss, best_loss)
            save_checkpoint({'epoch': epoch + 1,
                             'net': args.net,
                             'state_dict': model.state_dict(),
                             'best_loss': best_loss,
                             'optimizer': optimizer.state_dict(),
                             'memory': memory_bank
                             }, is_best, gap=5, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch + 1)),
                            keep_all=False)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))


def train(data_loader, model, model_perturbed, optimizer, epoch):
    losses = AverageMeter()
    model.train()
    model_perturbed.eval()
    global memoryManager, memory_bank

    for idx, (input_seq, input_aug_seq) in enumerate(data_loader):
        tic = time.time()
        input_seq = input_seq.to(cuda, non_blocking=True)
        input_aug_seq = input_aug_seq.to(cuda, non_blocking=True)
        B = input_seq.size(0)
        [instance_m_, cluster_m_, instance_aug_m_] = model(input_seq, input_aug_seq)

        loss_instance = loss_cluster = loss_entropy = None
        top1 = top3 = extra = None
        if epoch <= 190:
            loss_instance, top1, top3 = criterion_instance(instance_m_, instance_aug_m_)
            loss = loss_instance
        elif 190 < epoch <= args.memory_setup_epoch or epoch in [args.memory_reset_epoch]:
            loss_instance, top1, top3 = criterion_instance(instance_m_, instance_aug_m_)
            loss_entropy, extra = criterion_entropy(cluster_m_)
            loss = loss_instance + loss_entropy
        else:
            # re-process instances being assigned with low confidence
            cluster_max, _ = cluster_m_.max(dim=1)
            w_instance_indices = []
            for i in range(B):
                if cluster_max[i] < args.cluster_sigma:
                    w_instance_indices.append(i)
            [_, cluster_perturbed_, _] = model_perturbed(input_seq[w_instance_indices])
            re_cluster_max, re_cluster_indices = cluster_perturbed_.max(dim=1)
            penalize_instance_indices = []
            penalize_cluster_indices = []
            for i in range(len(w_instance_indices)):
                if re_cluster_max[i] > args.cluster_theta:
                    penalize_instance_indices.append(w_instance_indices[i])
                    penalize_cluster_indices.append(re_cluster_indices[i])

            loss_instance, top1, _ = criterion_instance(instance_m_, instance_aug_m_)
            loss_entropy, extra = criterion_entropy(cluster_m_)
            loss_cluster, top3, _ = criterion_cluster(instance_m_, cluster_m_, penalize_instance_indices, penalize_cluster_indices)
            loss = loss_instance + loss_entropy + args.cluster_ratio * loss_cluster

        del input_seq, input_aug_seq

        losses.update(loss.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f}) Time:{3:.2f}\t'.format(
                epoch, idx, len(data_loader), time.time() - tic, loss=losses))
            print('Loss Detail: instance={0}, entropy={1}, cluster={2}; top1_acc={3}, top3_acc={4}'.format(
                loss_instance, loss_entropy, loss_cluster, top1, top3))

            cluster_max, indices = cluster_m_.max(dim=1)
            print('\tMax cluster = ', cluster_max[:50], end='\t@@ ')
            # print('\t', indices)
            print('\tclusters counts =', indices.unique().shape, 'extra =', extra)

        del loss_instance, loss_entropy, loss_cluster, loss

        with torch.no_grad():
            if 500 < epoch <= args.memory_setup_epoch or epoch in [args.memory_reset_epoch]:
                memoryManager.setup_memory(instance_m_, cluster_m_)
            elif epoch > args.memory_setup_epoch:
                memoryManager.update_memory_bank(instance_m_, cluster_m_)

        del instance_m_, cluster_m_, instance_aug_m_

    return losses.local_avg


def get_data(transform, transform_aug=None, mode='train'):
    print('Loading video data for "%s" ...' % mode)
    if args.dataset == 'k400':
        dataset = Kinetics(mode=mode,
                           transform=transform,
                           transform_aug=transform_aug,
                           num_seq=args.num_seq,
                           downsample=args.ds)
    elif args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                            transform=transform,
                            transform_aug=transform_aug,
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

    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  sampler=sampler,
                                  shuffle=False,
                                  num_workers=16,
                                  pin_memory=True,
                                  drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_' \
                   'bs{args.batch_size}_lr{1}_seq{args.num_seq}_train-{args.train_what}{2}'\
            .format(
            '%s' % args.net[10:],
            args.old_lr if args.old_lr is not None else args.lr,
            '_pt=%s' % args.pretrain.replace('/', '-') if args.pretrain else '',
            args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path


if __name__ == '__main__':
    main()
