import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from importlib import import_module
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
import math

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10_whitened',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='cifar10',
                    help='model architecture: alexnet | resnet | ... '
                         '(default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=15000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')


def input_minimum(self, input, output):
    abs_vals = input[0].data.abs()
    zeros = abs_vals.lt(1e-38).float().sum(0)[0]
    # self.min_vals = getattr(self, 'min_vals', input[0].data[0,
    # :].clone().fill_(999.))
    self.zeros = getattr(self, 'zeros', input[0].data[0, :].clone().zero_())
    self.zeros.add_(zeros)
    # self.min_vals = torch.min(self.min_vals, abs_vals.min(0)[0])
    if hasattr(self, 'min_vals'):
        self.min_vals = torch.cat([self.min_vals, abs_vals])
    else:
        self.min_vals = abs_vals
    self.min_vals, _ = self.min_vals.sort(0)
    self.min_vals = self.min_vals[:10, :]
    abs_min = abs_vals.min()
    curr_min = getattr(self, 'min_input', 999)
    if curr_min > abs_min:
        self.min_input = abs_min


def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.info("run arguments: %s", args)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = import_module('.' + args.model, 'models').model
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)
    target_transform = lambda x: -1. if x % 2 else 1.
    if args.dataset == 'tinyImagenet':
        target_transform = lambda x: -1. if x < 500 else 1.

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=False),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})
    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.MSELoss)()
    criterion.type(args.type)
    model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform[
                           'eval'], target_transform=target_transform)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    train_data = get_dataset(args.dataset, 'train', transform[
                             'train'], target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # logging.info('training regime: %s', regime)

    model.relu.register_forward_hook(input_minimum)

    for epoch in range(args.start_epoch, args.epochs):
        model.relu.min_input = 999
        if hasattr(model.relu, 'min_vals'):
            model.relu.min_vals.fill_(999)
        if hasattr(model.relu, 'zeros'):
            model.relu.zeros.zero_()
        optimizer = adjust_optimizer(optimizer, epoch, regime)

        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer)

        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1
        }, is_best, path=save_path)
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1))
        # z = {'zeros_activation_%s' % i: model.relu.zeros[
        #     i] for i in range(model.relu.zeros.size(0))}
        # z2 = {'min_val_%s' % i: model.relu.min_vals[
        # i] for i in range(model.relu.min_vals.size(0))}
        # print(zip(range(model.relu.min_vals.size(0)), range(model.relu.min_vals.size(1))))
        z = {'activation_%s_min_%s' % (k, i): math.log(max(model.relu.min_vals[i, k], 1e-38))
             for i in range(model.relu.min_vals.size(0))
             for k in range(model.relu.min_vals.size(1))}

        # z = dict(z.items() + z2.items())
        mean_lr = torch.Tensor([p['lr']
                                for p in optimizer.param_groups]).mean()
        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    training=100. - train_prec1, validation=100 - val_prec1, learning_rate=mean_lr,
                    relu_input_min=math.log(max(model.relu.min_input, 1e-38)), num_zeros=model.relu.zeros.sum(),
                    log_training=math.log(max(100 - train_prec1, 1e-38)), **z)
        results.plot(x='epoch', y=['training', 'validation'],
                     title='Error', ylabel='error %')
        results.plot(x='epoch', y=['training'],
                     title='Training Error', ylabel='error %')
        results.plot(x='epoch', y=['log_training'],
                     title='Log Training Error', ylabel='error %')
        results.plot(x='epoch', y=['learning_rate'],
                     title='Learning Rate used for training', ylabel='value')
        results.plot(x='epoch', y=['relu_input_min'],
                     title='Log Abs minimum input to relu', ylabel='value')
        results.plot(x='epoch', y=['num_zeros'],
                     title='number of zeros (epoch)', ylabel='value')
        results.plot(x='epoch', y=['train_loss', 'val_loss'],
                     title='Loss', ylabel='loss')
        # for k in range(model.relu.zeros.size(0)):
        # results.plot(x='epoch', y=['zeros_feat_%s' % k],
        #  title='zeros num', ylabel='num')
        # results.plot(x='epoch', y=['min_val_%s' % k],
        #  title='minimum val at input', ylabel='num')
        for k in range(model.relu.min_vals.size(1)):
            ys = ['activation_%s_min_%s' %
                  (k, i) for i in range(model.relu.min_vals.size(0))]
            results.plot(
                x='epoch', y=ys, title='min values for activation %s' % k, ylabel='log value')
        results.save()


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.float().cuda(async=True)
        input_var = Variable(inputs, volatile=not training).type(args.type)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1 = (output.data * target).gt(0.0).float().mean() * 100.
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1, inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)


if __name__ == '__main__':
    main()
