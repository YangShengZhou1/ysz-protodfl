import copy
import logging
import time
import pdb
import numpy as np
import torch
from torch import nn
import torch.nn as nn
from fedml_api.model.cv.cnn_meta import Meta_net

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args=args
        self.logger = logger

    def set_masks(self, masks):
        self.masks=masks
        # self.model.set_masks(masks)

    def get_model_params(self,device):
        self.model.to(device)
        return copy.deepcopy(self.model.state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_trainable_params(self,device):
        dict= {}
        for name, param in self.model.named_parameters():
            dict[name] = param.to(device)
        return dict
    
    def train_rep(self, train_data, device,  args):
        model = self.model
        model.to(device)
        model.eval()
        for _,(x, y) in enumerate(train_data):
            if type(x) == type([]):
                x[0] = x[0].to(device)
            else:
                x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                rep = model.base(x).detach()
            break
        return rep
    
    def train(self, train_data, device,  args,round, old_running_mean, momentums):
        # torch.manual_seed(0)
        model = self.model
        model.to(device)
        model.train()
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr* (args.lr_decay**round), momentum=args.momentum,weight_decay=args.wd)
        #loss = nn.CrossEntropyLoss()
        for epoch in range(args.epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                labels = labels.to(device)
                model.zero_grad()
                rep = model.base(x)
                running_mean = torch.mean(rep, dim=0)
                old_running_mean = (1-momentums) * old_running_mean + momentums * running_mean
                output = model.head(rep)
                losses = criterion(output, labels.long())
                optimizer.zero_grad()
                losses.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                old_running_mean.detach_()
                epoch_loss.append(losses.item())
                #for name, param in self.model.named_parameters():
                #    if name in self.masks:
                #        param.data *= self.masks[name].to(device)
            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
        return old_running_mean
    
    def test(self, test_data, device, args):
        model = self.model
        model.to(device)
        model.eval()
        metrics = {
            'test_correct': 0,
            'test_acc':0.0,
            'test_loss': 0,
            'test_total': 0
        }
        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                rep = model.base(x)
                pred = model.head(rep)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']
        return metrics
    
    def val(self, val_data, val_running_mean, momentums, device, args):#通过验证集计算验证集的原型向量，从而得到KL散度距离的阈值
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_acc':0.0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)
        val_running_mean = val_running_mean.zero_().to(device)
        # Reduce the batch size by a factor of 10
        original_batch_size = val_data.batch_size
        reduced_batch_size = max(1, original_batch_size // 10)  # Ensure batch size is at least 1

        reduced_val_data = torch.utils.data.DataLoader(val_data.dataset, batch_size=reduced_batch_size, shuffle=False, num_workers=val_data.num_workers)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(reduced_val_data):
                x = x.to(device)
                target = target.to(device)
                rep = model.base(x)
                # Update validation running mean
                running_mean = torch.mean(rep, dim=0)
                val_running_mean = (1 - momentums) * val_running_mean + momentums * running_mean

        return val_running_mean
    
    '''

    def train(self, train_data,  device,  args, round):
        # torch.manual_seed(0)
        model = self.model
        model.to(device)
        model.train()
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr* (args.lr_decay**round), momentum=args.momentum,weight_decay=args.wd)
        for epoch in range(args.epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()

                log_probs = model.forward(x)
                loss = criterion(log_probs, labels.long())
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()
                # self.logger.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                epoch_loss.append(loss.item())

            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))



    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics
        '''

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

