import torch
import torch.nn.functional as F
import torch.nn as nn

class InfoNCE(nn.Module):
    def __init__(self):
        super(InfoNCE, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def softXEnt(self, target, logits):    #crossentropy loss
        logprobs = torch.nn.functional.log_softmax(logits, dim = 1)#log(softmax(logits))
        loss = -(target * logprobs).sum() / logits.shape[0] # - target位置上的logprobs相加，取平均
        return loss

    def forward(self, zis, zjs, norm=True, weights=1.0, temperature = 0.1):#需要看论文多少合适
        #zis, zjs = [8,65,778]

        # temperature = self.temperature
        # alpha = self.alpha_weight

        LARGE_NUM = 1e9

         # Get (normalized) hidden1 and hidden2.
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)
            #pooling之后的L2 norm，将值进行归一化到01之间
            
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        # hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        # labels = labels.to(self.device)
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        
        """
        Different from Image-Image contrastive learning
        In the case of Image-Text contrastive learning we do not compute the similarity function between the Image-Image and Text-Text pairs  
        """
        # logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large,0, 1)) / temperature
        # logits_aa = logits_aa - masks * LARGE_NUM
        # logits_bb = torch.matmul(hidden2,  torch.transpose(hidden2_large,0, 1)) / temperature
        # logits_bb = logits_bb - masks * LARGE_NUM

        #这种不需要mask
        #矩阵相乘矩阵转置
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large,0, 1)) / temperature
        # logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large,0, 1)) / temperature
       
        loss_a = self.softXEnt(labels, logits_ab)
        # loss_b = self.softXEnt(labels, logits_ba)
        #只学侧→正

        # return alpha*loss_a + (1-alpha)*loss_b
        return loss_a, {'InfoNCE Loss': loss_a.item()}


    