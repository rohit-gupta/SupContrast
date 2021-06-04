from __future__ import print_function

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn 
import numpy as np


from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier

from util import fgsm_attack

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    # adversarial attack
    parser.add_argument('--epsilons', metavar='N', type=int, nargs='+',
                        help='adversarial attack epsilons')

    # for visualization
    parser.add_argument('--eval', action='store_true',
                        help='evaluate pretrained model')
    parser.add_argument('--viz', action='store_true',
                        help='visualize features')


    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        state_dict = { k.replace("module.encoder.", "encoder.module."):v for k, v in state_dict.items() }

        model.load_state_dict(state_dict, strict=False)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    if opt.viz:
        allfeatures = []
        alllabels = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            features = model.encoder(images)
            output = classifier(features)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
            

            if opt.viz:
                allfeatures += [features]
                alllabels += [labels]



    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))


    if opt.viz:

        # Features Visualization
        # features = torch.cat(allfeatures,dim=0).cpu().numpy()
        # labels = torch.cat(alllabels,dim=0).cpu().numpy()
        # print("dimensions:", features.shape, labels.shape)
        # from sklearn.manifold import TSNE
        # tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=50)
        # X_2d = tsne.fit_transform(features)
        # target_ids = range(10)
        # colors = 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
        # cifar10classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # import matplotlib
        # matplotlib.use('Agg')
        # from matplotlib import pyplot as plt
        # f = plt.figure(figsize=(6, 5))
        # for i, c, label in zip(target_ids, colors, cifar10classes):
        #     plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], c=c, label=label)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5)
        # plt.xticks([])
        # plt.yticks([])
        # f.savefig(opt.ckpt.split("/")[-2].split("_")[0] +  "_test_pca_p50.pdf", bbox_inches='tight')

        # Distance visualization
        from scipy.spatial.distance import pdist, squareform
        with torch.no_grad():
            features = torch.cat(allfeatures,dim=0)
            labels = torch.cat(alllabels,dim=0)

            labelsc = labels.contiguous().view(-1, 1)
            same_class_mask = torch.eq(labelsc, labelsc.T).float() - torch.eye(labelsc.shape[0]).float().cuda()
            diff_class_mask = 1. - torch.eq(labelsc, labelsc.T).float() 
            #cosdist = nn.CosineSimilarity(dim=-1, eps=1e-6)
            #distances = cosdist(features.unsqueeze(0), features.unsqueeze(1))
            features = features.cpu().numpy()
            distances = 1 - squareform(pdist(features, metric='cosine'))

            same_class_mask = same_class_mask.cpu().numpy()
            matching_pairs_count = np.sum(same_class_mask)
            same_class_dist = np.sum(same_class_mask*distances)/matching_pairs_count


            diff_class_mask = diff_class_mask.cpu().numpy()
            mismatching_pairs_count = np.sum(diff_class_mask)
            diff_class_dist = np.sum(diff_class_mask*distances)/mismatching_pairs_count

            labels = labels.cpu().numpy()

            for x in range(10):
                for y in range(10):
                    # if x != y:
                    print(x, y, np.mean(distances[(labels == x), (labels == y)]))
                    print(x, y, np.sum((labels == x)), np.sum((labels == y)), (distances[(labels == x), (labels == y)]).size)

                    # else:
                    #     count = np.sum(labels == x)*np.sum(labels == x) - np.sum(labels == x)
                    #     print(x, y, count)
                    #     print(x, y, np.sum((distances*same_class_mask)[labels == x, labels == y])/count)





            print(same_class_mask.shape, distances.shape)
            print(same_class_mask[:10,:10])
            print(distances[:10,:10])

            print(same_class_dist, diff_class_dist)
        




    return losses.avg, top1.avg


def adveval(val_loader, model, classifier, criterion, opt, epsilon):
    """adversarial robustness evaluation"""
    model.eval()
    classifier.eval()

    if opt.dataset == 'cifar10':
        means = [0.4914, 0.4822, 0.4465]
        stds = [0.2023, 0.1994, 0.2010]
    elif opt.dataset == 'cifar100':
        means = [0.5071, 0.4867, 0.4408]
        stds = [0.2675, 0.2565, 0.2761]

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    advtop1 = AverageMeter()

    # with torch.no_grad():
    end = time.time()
    for idx, (images, labels) in enumerate(val_loader):
        images = images.float().cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]

        # for generating adversarial perturbation
        images.requires_grad = True

        # forward
        output = classifier(model.encoder(images))
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # Zero out model gradients
        model.encoder.zero_grad()
        classifier.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward(retain_graph=True)

        # Collect datagrad
        gradients = images.grad.data

        # Call FGSM Attack
        perturbed_images = fgsm_attack(images, float(epsilon)/255., gradients, torch.FloatTensor(means), torch.FloatTensor(stds))

        # Adversarial prediction
        advoutput = classifier(model.encoder(perturbed_images))

        advacc1, _ = accuracy(advoutput, labels, topk=(1, 5))
        advtop1.update(advacc1[0], bsz)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'
                  'AdvAcc@1 {advtop1.val:.3f} ({advtop1.avg:.3f})'.format(
                   idx, len(val_loader), batch_time=batch_time,
                   loss=losses, top1=top1, advtop1=advtop1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg, advtop1.avg



def main():
    best_acc = 0
    best_classifier = None  
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)
    best_classifier = classifier

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    if opt.eval:
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt) 
    else:
        # training routine
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train(train_loader, model, classifier, criterion,
                              optimizer, epoch, opt)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2 - time1, acc))

            # eval for one epoch
            loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
            if val_acc > best_acc:
                best_acc = val_acc
                best_classifier = classifier

        print('best accuracy: {:.2f}'.format(best_acc))

    for epsilon in opt.epsilons:
        loss, acc, adv_acc = adveval(val_loader, model, best_classifier, criterion, opt, epsilon)
        print('adv accuracy at epsilon {:.2f}: {:.2f}'.format(epsilon, adv_acc))



if __name__ == '__main__':
    main()
