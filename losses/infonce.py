from doctest import OutputChecker
from operator import concat
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class InfoNCE(nn.Module):
    def __init__(self):
        super(InfoNCE, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)


    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # labels = labels.to(self.args.device)
        # print(features)
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

                  
        mask = torch.eye(labels.shape[0], dtype=torch.bool)                                     
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long)

        logits = logits / temperature
        return logits, labels

    def forward(self,zis, zjs):
        zis=zis.cpu() 
        zjs=zjs.cpu() 
        
        zis=zis.detach().numpy()
        zjs=zjs.detach().numpy()
        
        zis=np.reshape(zis,(zis.shape[0],-1))
        zjs=np.reshape(zjs,(zjs.shape[0],-1))
        # print(y)
        # print(np.shape(y))
        zis = torch.from_numpy(zis)
        zjs = torch.from_numpy(zjs)
      

        fc = nn.Linear(zis.shape[1],128)#投射
        zis = fc(zis)
        zjs = fc(zjs)



        zis=zis.cpu() 
        zjs=zjs.cpu() 
        
        zis=zis.detach().numpy()
        zjs=zjs.detach().numpy()

        features = concat(zis,zjs)
         
        features = torch.from_numpy(features)

        logits, labels = self.info_nce_loss(features)
        loss = self.criterion(logits, labels)
        return loss, {'InfoNCE Loss': loss.item()}
        
