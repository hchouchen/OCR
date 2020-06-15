import copy,os
import torch
import utils
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
import models.crnn as crnn

from PIL import Image
from skimage import io
from keys import getAlphabet
from dataset import *
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
    

def train(args):
    device = torch.device("cuda:0")
    
    alphabet = getAlphabet(args.language)
    nclass = len(alphabet) + 1
    
    loss_avg = utils.averager()
    converter = utils.strLabelConverter(alphabet)
    criterion = nn.CTCLoss()
    criterion = criterion.cuda()
    
    train_dataset = ImageDataset(args,train=True)
    train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True)
    val_dataset = ImageDataset(args,train=False)
    val_loader = DataLoader(dataset=val_dataset,batch_size=args.batch_size,shuffle=True)
    
    model = crnn.CRNN(args.imgH, 1, nclass, 256, 1)
    model = model.to(device)
    
    if pretrained=='true':
        model.load_state_dict(torch.load(args.weight_path))
    
    optimizer = optim.SGD(model.parameters(),lr = 0.1,momentum=0.9,weight_decay=0.00004)
    #optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
    
    losses = [] 
    acces = []
    eval_losses = []
    eval_acces = []
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1,gamma=0.5)
    
    for echo in range(args.epoch):
        train_loss = 0
        train_acc = 0
        model.train()        
        if np.mod(echo,5) == 4:
            scheduler.step()
        for i,(X,label) in enumerate(train_loader):
            X = X.to(device)
            text, length = converter.encode(label)
            preds = model(X)
            #out = F.log_softmax(preds)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))
            cost = criterion(preds, text, preds_size, length)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            loss_avg.add(cost)
        losses.append(loss_avg.val())
        print("echo:"+' ' +str(echo))
        print("train-loss:" + ' ' + str(loss_avg.val()))
        #trloss, = plt.plot(losses)
        loss_avg.reset()
        
        model.eval()
        n_correct = 0
        for i,(X,label) in enumerate(val_loader):
            img = X
            X = Variable(X).cuda()
            text, length = converter.encode(label)
            preds = model(X)
            #out = F.log_softmax(preds)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))
            cost = criterion(preds, text, preds_size, length)
            loss_avg.add(cost)
            
            _, preds = preds.max(2)
            #preds = preds.squeeze(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            #print(sim_preds)
            for pred, target in zip(sim_preds, label):
                if pred.strip() == target.strip():
                    n_correct += 1
    
        accuracy = n_correct / float(len(val_loader) * BATCH_SIZE)
        eval_losses.append(loss_avg.val())
        eval_acces.append(accuracy)
        print("test-loss:" + ' ' + str(loss_avg.val()))
        print("accuracy:"+' '+str(accuracy))
        loss_avg.reset()
        
        #teloss, = plt.plot(eval_losses)
        #plt.legend(handles=[trloss,teloss],labels=['train-loss','test-loss'],loc='upper right')
        #plt.show()
        #plt.plot(eval_acces)
        #plt.show()
        

def test(args):
    device = torch.device("cuda:0")
    print(device)
    alphabet = getAlphabet(args.language)
    nclass = len(alphabet) + 1
    
    converter = utils.strLabelConverter(alphabet)
    criterion = nn.CTCLoss()
    criterion = criterion.to(device)
    
    test_dataset = testDataset(root=args.test_root)
    test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)
    
    test_model = crnn.CRNN(args.imgH, 1, nclass, 256, 1)
    test_model = test_model.to(device)
    
    test_model.load_state_dict(torch.load(args.weight_path))
    
    result = []
    model.eval()
    for i,(X,label) in enumerate(test_loader):
        X = X.to(device)
        preds = model(X)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))
        _, preds = preds.max(2)
        #preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        result.append(sim_preds)
  
    print(result)
    

def main(args):
    if args.mode == 'train':
        train(args)
    else:
        test(args)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR inference pipeline')
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--language', default='Russian', type=str)
    parser.add_argument('--pretrained', default='true', type=str)
    parser.add_argument('--weight_path', default=r'./Russian-weight.pth', type=str)
    parser.add_argument('--imgH', default=32, type=int)
    parser.add_argument('--img_path', default=r'./Russian-data.npy', type=str)
    parser.add_argument('--label_path', default=r'./Russian-label.csv', type=str)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--test_root', default='./test', type=str)
    
    args = parser.parse_args()
    main(args)
