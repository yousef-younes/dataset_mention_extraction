
import torch
import torch.nn as nn
import torch.nn.functional as F

#this ist he MLP network to be used for classification
class MLP(nn.Module):
    def __init__(self,hidden_size1, hidden_size2, output_dim, dropout,gamma,alpha):
        super().__init__()

        self.num_classes = output_dim
        self.gamma = gamma
        self.alpha = alpha


        self.classifier = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size2, self.num_classes),
        )



    def forward(self, x,labels=None):

        logits = self.classifier(x)

        loss = None
        if labels is not None:
            # if labels is not None:
            assert self.num_classes == 2, f'Expected 2 labels but found {self.num_labels}'
            if self.alpha != None:
                alpha_tensor = torch.Tensor([self.alpha, 1 - self.alpha])
                loss_fct = F.cross_entropy(logits.view(-1, self.num_classes), labels.view(-1), weight=alpha_tensor)
            else:
                loss_fct = F.cross_entropy(logits.view(-1, self.num_classes), labels.view(-1))

            pt = torch.exp(-loss_fct)
            #loss = (((1 - pt) ** self.gamma * loss_fct)+loss_fct)/2
            loss = ((1 - pt) ** self.gamma * loss_fct)


        return loss,torch.sigmoid(logits)
