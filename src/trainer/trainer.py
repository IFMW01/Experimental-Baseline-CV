import torch
from tqdm import tqdm
from copy import deepcopy
import time
import pandas as pd
from torchmetrics.classification import MulticlassCalibrationError

# Trainer class used to train base and Naive models
class Trainer():
    def __init__(self, model, train_loader, test_loader, optimizer, criterion, device, n_epoch,n_classes):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.n_epoch = n_epoch
        self.n_classes = n_classes

# Evaluates performance obtaining loss, acc and ece
    def evaluate(self,dataloader):
        self.model.eval()
        model_loss = 0.0
        correct = 0
        total = 0
        ece = 0
        ece = MulticlassCalibrationError(self.n_classes, n_bins=15, norm='l1')
        for data, target in dataloader:
            with torch.no_grad():
                if data.device != self.device:
                    data = data.to(self.device) 
                if target.device != self.device:
                    target = target.to(self.device) 
                output = self.model(data)
                loss = self.criterion(output, target)
                ece.update(torch.softmax(output, dim=1),target)
                model_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        ece = ece.compute().item()
        model_loss /= len(dataloader)
        accuracy = 100 * correct / total
        return accuracy,model_loss, ece
    
    def evaluate_test(self,dataloader):
        self.model.eval()
        model_loss = 0.0
        correct = 0
        total = 0
        ece = 0
        predictions = pd.DataFrame()
        temp = pd.DataFrame()
        ece = MulticlassCalibrationError(self.n_classes, n_bins=15, norm='l1')
        for data, target in dataloader:
            with torch.no_grad():
                if data.device != self.device:
                    data = data.to(self.device) 
                if target.device != self.device:
                    target = target.to(self.device) 
                output = self.model(data)
                loss = self.criterion(output, target)
                softmax = torch.softmax(output, dim=1)
                temp = pd.DataFrame(softmax.cpu().numpy()) 
                predictions = pd.concat([predictions, temp], ignore_index=True)
                ece.update(softmax,target)
                model_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        ece = ece.compute().item()
        model_loss /= len(dataloader)
        accuracy = 100 * correct / total
        return accuracy,model_loss, ece, predictions
    
# Training of the model
    def train(self):
        training_sequence = {}
        train_ece = 0 
        test_ece = 0
        training_time = 0
        epoch_results = {}
        training_sequence = {}
        for epoch in tqdm(range(0, self.n_epoch)):
            epoch_time = 0
            start_time = time.time()
            self.model.train()

            for batch_idx, (data, target) in enumerate(self.train_loader):
                if data.device != self.device:
                    data = data.to(self.device) 
                if target.device != self.device:
                    target = target.to(self.device) 
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            end_time = time.time()
            epoch_time = end_time - start_time
            training_time +=  round(epoch_time, 3)

            
            train_accuracy,train_loss,train_ece = self.evaluate(self.train_loader)
            test_accuracy,test_loss, test_ece,predictions= self.evaluate_test(self.test_loader)

            epoch_results['train accuracy'] = train_accuracy
            epoch_results['train loss'] = train_loss
            epoch_results['train ece'] = train_ece
            epoch_results['test accuracy'] = test_accuracy
            epoch_results['test loss'] = test_loss
            epoch_results['test ece'] = test_ece
            epoch_results['training time'] = training_time
            epoch_results['function'] = predictions.to_numpy().tolist()
            training_sequence[f'{epoch}'] = epoch_results

            if  epoch%10==0:
                print(f"Epoch: {epoch}/{self.n_epoch}\tTrain accuracy: {train_accuracy:.2f}%\tTrain loss: {train_loss:.6f}\tTrain ECE {train_ece:.2f}")
                print(f'Test loss: {test_loss:.6f}, Test accuracy: {test_accuracy:.2f}%\tTest ECE {test_ece:.2f}"')

        return training_sequence,self.model
    