from re import X
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# zis = torch.ones(8, 65,778)
# zjs = torch.ones(8, 65,778)
# # print(zis,zjs)
# # print(zis.size(),zjs.size())

# zis = F.normalize(zis, p=2, dim=1)
# zjs = F.normalize(zjs, p=2, dim=1)
# # print(zis,zjs)
# # print(zis.size(),zjs.size())

            
# hidden1, hidden2 = zis, zjs
# batch_size = hidden1.shape[0]
# # print(batch_size)

# hidden2_large = hidden2

# labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
# masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
# #两个对角阵
# print(labels,masks)  
# print(labels.size(),masks.size())        
# temperature = 0.1
# # logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large,0, 1)) / temperature

# logits_ab = torch.matmul(hidden1, hidden2_large.T) / temperature
#        #hidden1=[8,65,778]
#        #torch.transpose(hidden2_large,0, 1)=[65,8,778]
# # logprobs = torch.nn.functional.log_softmax(logits_ab, dim = 1)#log(softmax(logits))
# # loss_a = -(labels * logprobs).sum() / logits_ab.shape[0] # - target位置上的logprobs相加，取平均
        
       
    
# # print (loss_a, {'InfoNCE Loss': loss_a.item()})     


         


# x=torch.ones(4,2,3)
# y=torch.sigmoid(x)
y=np.ones((4,2,3))
y=np.reshape(y,(y.shape[0],-1))
y=torch.from_numpy(y)#[4,6]
y = torch.tensor(y, dtype=torch.float32)
model = nn.Linear(y.shape[0],3)
y=model(y)
print(y)
# print(np.shape(y))
print(y.size())



