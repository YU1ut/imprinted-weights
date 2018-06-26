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
import torchvision.transforms as transforms
import models
import loader

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='CUB_200_2011',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-c', '--checkpoint', default='imprint_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--method', '-m', metavar='METHOD', default='imprint',
                    choices=['imporint, random'])
parser.add_argument('--num-sample', default=1, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    model = models.Net().cuda()


    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch']))
    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_dataset = loader.ImageLoader(
        args.data,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), train=True, num_classes=200, num_train_sample=args.num_sample, novel_only=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        loader.ImageLoader(args.data, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), num_classes=200),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    imprint(train_loader, model)
    validate(val_loader, model)


def imprint(train_loader, model):

    # switch to evaluate mode
    model.eval()

    for batch_idx, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # compute output
        output = model.extractor(input)

        if batch_idx == 0:
            output_stack = output
            target_stack = target
        else:
            output_stack = torch.cat((output_stack, output), 0)
            target_stack = torch.cat((target_stack, target), 0)

    new_weight = torch.zeros(100, 2048).cuda()
    for i in range(len(target_stack)):
        tmp = output_stack[target_stack == (i + 100)].mean(0)
        new_weight[i] = tmp / tmp.norm(p=2)
    weight = torch.cat((model.classifier.fc.weight.data, new_weight))
    model.classifier.fc = nn.Linear(2048, 200, bias=False)
    model.classifier.fc.weight.data = weight

def validate(val_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    bar = Bar('Processing', max=len(val_loader))
    with torch.no_grad():
        end = time.time()
        for batch_idx, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda()
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

             # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()