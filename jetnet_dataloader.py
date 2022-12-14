
import jetnet
from torch.utils.data import DataLoader
import pytorch_lightning as pl   
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from helpers import *
from nflows.distributions.base import Distribution
import matplotlib.pyplot as plt

class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)
    def inverse_transform(self,values):
        return (values *self.std)+self.mean
    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
    def to(self,dev):
        self.std=self.std.to(dev)
        self.mean=self.mean.to(dev)
        return self
  

class JetNetDataloader(pl.LightningDataModule):
    '''This is more or less standard boilerplate coded that builds the data loader of the training
       one thing to note is the custom standard scaler that works on tensors
       Currently only jets with 30 particles are used but this maybe changes soon'''
    def __init__(self,config): 
        super().__init__()
        self.config=config
        self.n_dim=config["n_dim"]

        self.batch_size=config["batch_size"]
    def setup(self,stage):
    # This just sets up the dataloader, nothing particularly important. it reads in a csv, calculates mass and reads out the number particles per jet
    # And adds it to the dataset as variable. The only important thing is that we add noise to zero padded jets
        
        data=jetnet.datasets.JetNet(self.config["parton"],normalize=False,train=True).data.float() 
        test_set=jetnet.datasets.JetNet(self.config["parton"],normalize=False,train=False).data.float()
        self.data=torch.cat((data,test_set),dim=0)
        
        # masks=np.sum(data.values[:,np.arange(3,120,4)],axis=1)
        masks = self.data[:,:,-1].bool()
        self.data=self.data[:,:,:-1].reshape(-1,90)
        self.n = masks.sum(axis=1).reshape(-1,1)
        
        # masks = ~masks

        if self.config["canonical"]:
            self.data=preprocess(self.data)        
        # calculating mass per jet
        self.m=mass(self.data[:,:self.n_dim],self.config["canonical"]).reshape(-1,1)  
      # Adding noise to zero padded jets.
        for i in torch.unique(self.n):
            i=int(i)
            self.data[self.data[:,-1]==i,3*i:90]=torch.normal(mean=torch.zeros_like(self.data[self.data[:,-1]==i,3*i:90]),std=1).abs()*1e-7

        #standard scaling 
        self.scaler=StandardScaler()
        self.data=torch.hstack((self.data,self.m))        
        self.scaler.fit(self.data)
        self.data=self.scaler.transform(self.data)
        self.min_m=self.scaler.transform(torch.zeros((1,91)))[0,-1]
        self.data=torch.hstack((self.data,self.n))
        
        #calculating mass dist in different bins, this is needed for the testcase where we need to generate the conditoon
        if self.config["variable"]:
            self.mdists={}
            
            for i in torch.unique(self.n):
                if len(self.data[self.n[:,0]==i])==1:
                    
                    self.mdists[int(i)]=[lambda x:self.data[self.n[:,0]==i,-2],lambda x:self.data[self.n[:,0]==i,-2]]
                else:
                    self.mdists[int(i)]=F(self.data[self.n[:,0]==i,-2])    
        self.data,self.test_set=train_test_split(self.data.cpu().numpy(),test_size=0.3)
        self.n_train=self.data[:,-1]
        self.n_test=self.test_set[:,-1]
        
            
        self.test_set=torch.tensor(self.test_set).float()
        self.data=torch.tensor(self.data).float()
        assert self.data.shape[1]==92
        assert (torch.isnan(self.data)).sum()==0

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=len(self.test_set))