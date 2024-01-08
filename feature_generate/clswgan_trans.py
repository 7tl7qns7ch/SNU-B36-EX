from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import math
import util 
import classifier
import classifier2
import sys
import model
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SNU36', help='SNU36')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='CNN_fea')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=500, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--feaSize', type=int, default=128, help='size of visual features')
parser.add_argument('--attSize', type=int, default=7, help='size of semantic features')
parser.add_argument('--nz', type=int, default=9, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=512, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=512, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--resultsroot', help='resultsroot')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=10)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=20190927, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=195, help='number of all classes')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if os.path.isdir(opt.resultsroot) == True:
    pass
else:
    os.makedirs(opt.resultsroot)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# initialize generator and discriminator
netG = model.Generator_trans(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model.Critic_trans(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()

input_fea = torch.FloatTensor(opt.batch_size, opt.feaSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input_fea, input_att, noise = input_fea.cuda(), input_att.cuda(), noise.cuda()
    one, mone = one.cuda(), mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_fea.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.feaSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)

    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG.forward(syn_noise, syn_att)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_(True)

    if opt.cuda:
        interpolates = interpolates.cuda()

    #interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, input_att)

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=ones, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    #print(autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0])

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

# train a classifier on seen classes, obtain \theta of Equation (4)
#train_features = data.train_feature.cpu().detach().numpy()
train_label = data.train_label.cpu().detach().numpy()
seenclassesss = data.seenclasses.cpu().detach().numpy()
a = util.map_label(data.train_label, data.seenclasses).cpu().detach().numpy()

map_labels = util.map_label(data.train_label, data.seenclasses)
pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses),
                                     data.seenclasses.size(0), opt.feaSize, opt.cuda, 0.001, 0.5, 50, 100,
                                     opt.pretrain_classifier)

acc = pretrain_cls.val(data.test_seen_feature, data.test_seen_label, data.seenclasses)
print(acc)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters(): # set requires_grad to False
    p.requires_grad = False

loss_D = []
loss_G = []
wasserstein_dist = []
c_err = []
unseen_acc = []
seen_acc = []
h_acc = []
output_seen = []
output_unseen = []

for epoch in range(opt.nepoch):
    FP = 0 
    mean_lossD = 0
    mean_lossG = 0
    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()
            # train with realG
            # sample a mini-batch
            #sparse_real = opt.feaSize - input_fea[1].gt(0).sum()
            #input_resv = Variable(input_fea)
            #input_attv = Variable(input_att)

            criticD_real = netD(input_fea, input_att)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            # train with fakeG
            noise.normal_(0, 1)
            #noisev = Variable(noise)
            fake = netG(noise, input_att)
            fake_norm = fake.data[0].norm()
            #sparse_fake = fake.data[0].eq(0).sum()
            criticD_fake = netD(fake.detach(), input_att)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_fea, fake.data, input_att)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty

            optimizerD.step()

        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation

        netG.zero_grad()
        #input_attv = Variable(input_att)
        noise.normal_(0, 1)
        #noisev = Variable(noise)
        fake = netG(noise, input_att)
        criticG_fake = netD(fake, input_att)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        # classification loss
        c_errG = cls_criterion(pretrain_cls.model(fake), input_label)
        errG = G_cost + opt.cls_weight*c_errG
        errG.backward()
        optimizerG.step()

    mean_lossG /=  data.ntrain / opt.batch_size 
    mean_lossD /=  data.ntrain / opt.batch_size 
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f'
              % (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(), c_errG.item()))
    loss_D.append(D_cost.item())
    loss_G.append(G_cost.item())
    wasserstein_dist.append(Wasserstein_D.item())
    c_err.append(c_errG.item())

    if epoch % opt.save_every == 0:
        # evaluate the model, set G to evaluation mode
        netG.eval()
        # Generalized zero-shot learning
        if opt.gzsl:
            model_path = opt.resultsroot + '/netG' + '/netG' + str(epoch) + '.pth'
            model_dir = opt.resultsroot + '/netG'
            if os.path.isdir(model_dir) == True:
                pass
            else:
                os.makedirs(model_dir)
            print("Model saved")
            torch.save(netG.state_dict(), model_path)
            syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_Y = torch.cat((data.train_label, syn_label), 0)
            nclass = opt.nclass_all
            cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
            unseen_acc.append(cls.acc_unseen)
            seen_acc.append(cls.acc_seen)
            h_acc.append(cls.H)
            output_seen.append(cls.output_seen)
            output_unseen.append(cls.output_unseen)
        # Zero-shot learning
        else:
            syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
            cls = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
            acc = cls.acc
            print('unseen class accuracy= ', acc)

        # reset G to training mode
        netG.train()

loss_D = np.array(loss_D).astype(float)
loss_G = np.array(loss_G).astype(float)
wasserstein_dist = np.array(wasserstein_dist).astype(float)
c_err = np.array(c_err).astype(float)
unseen_acc = np.array(unseen_acc).astype(float)
seen_acc = np.array(seen_acc).astype(float)
h_acc = np.array(h_acc).astype(float)

output_seen = np.array(output_seen).astype(float)
output_unseen = np.array(output_unseen).astype(float)

results_name = 'sn' + str(opt.syn_num) + 'nz' + str(opt.nz) + 'gh' + str(opt.ngh)

os.chdir(opt.resultsroot)
if os.path.isdir(results_name) == True:
    pass
else:
    os.makedirs(results_name)

os.chdir(results_name)

loss_D_dir = 'loss_D.npy'
loss_G_dir = 'loss_G.npy'
wasserstein_dist_dir = 'wasserstein_dist.npy'
c_err_dir = 'c_err.npy'
unseen_acc_dir = 'unseen_acc.npy'
seen_acc_dir = 'seen_acc.npy'
h_acc_dir = 'h_acc.npy'

logits_seen_name = 'logits_seen.npy'
logits_unseen_name = 'logits_unseen.npy'

np.save(loss_D_dir, loss_D)
np.save(loss_G_dir, loss_G)
np.save(wasserstein_dist_dir, wasserstein_dist)
np.save(c_err_dir, c_err)
np.save(unseen_acc_dir, unseen_acc)
np.save(seen_acc_dir, seen_acc)
np.save(h_acc_dir, h_acc)

np.save(logits_seen_name, output_seen)
np.save(logits_unseen_name, output_unseen)

