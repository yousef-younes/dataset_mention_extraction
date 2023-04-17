
import math
import os

import torch
from sklearn.metrics import recall_score, classification_report,f1_score

from tqdm import tqdm

import general_util as gu

import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from MLP_2 import MLP

os.environ["CUDA_VISIBLE_DEVICES"]="1"


class Meta_MLP(nn.Module):
    def __init__(self,hidden_size1, output_dim):
        super().__init__()

        self.num_classes = output_dim
        self.classifier = nn.Sequential(
            nn.Linear(6, 32),
            #nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, self.num_classes),
            nn.ReLU()
        )
        #self.classifier = nn.Linear(hidden_size1,self.num_classes)

    def forward(self, x,labels=None):

        logits = self.classifier(x)

        loss = None
        if labels is not None:
            # if labels is not None:
            assert self.num_classes == 2, f'Expected 2 labels but found {self.num_labels}'
            loss = F.cross_entropy(logits.view(-1, self.num_classes), labels.view(-1))

        return loss,torch.sigmoid(logits)

def accuracy(probs, target):
    winners = probs.argmax(dim=1)
    corrects = (winners == target)
    accuracy = corrects.sum().float() / float(target.size(0))
    return accuracy

def compute_recall(true_label,probs):
    preds = probs.argmax(dim=1)
    return recall_score(true_label,preds)

def compute_f1(true_label,probs):
    preds = probs.argmax(dim=1)
    return f1_score(true_label,preds)

def train(models,meta_model,trainDataLoader, optimizer,scheduler):
    epoch_loss = 0
    epoch_acc = 0
    epoch_recall = 0
    batch_num = len(trainDataLoader)  # number of batches in the data
    
    meta_model.train()
    
    weights =[0.4,0.3,0.3]
    
    for step, batch in enumerate(tqdm(trainDataLoader)):
        batch_preds_tensor = None
        for idx, model in enumerate(models):
            model.eval()
            loss,predictions = model(batch[0],batch[1])
            if batch_preds_tensor == None:
                batch_preds_tensor = torch.mul(predictions,weights[idx])
            else:
                batch_preds_tensor = torch.cat((batch_preds_tensor,torch.mul(predictions,weights[idx])),dim=1)
                
            #print(batch_preds_tensor.shape)
    
        optimizer.zero_grad()

        loss,meta_predictions = meta_model(batch_preds_tensor, batch[1])


        acc = accuracy(meta_predictions, batch[1])

        recall = compute_recall(batch[1], meta_predictions)

        # perform backpropagation
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_recall += recall.item()

        #change the learning rate every 50 batches
        #if step%50 == 0:
            #scheduler.step(epoch_loss/step)

    avg_loss, acc, recall = epoch_loss / batch_num, epoch_acc / batch_num, epoch_recall / batch_num
    scheduler.step(avg_loss)

    return avg_loss,acc,recall


    
def evaluate(models,meta_model,batch_size,val_dataloader):
    print("ensemble evaluation started.....")
    epoch_loss = 0
    epoch_acc = 0
    epoch_recall = 0
    epoch_f1 = 0
    batch_num = len(val_dataloader) # number of batches
    

    meta_model.eval()
    
    all_preds = []
    all_labels = []
    batch_size_tensor = torch.Tensor([batch_size])
    weight_list = [0.4,0.3,0.3] 
    with torch.no_grad():
        #for input, labels, batch_num in batch_iter(x_val, y_val, batch_size, shuffle=True):
        for step, batch in enumerate(tqdm(val_dataloader)):
            batch_preds_tensor = None
            for indd, model in enumerate(models):
                model.eval()
                loss,predictions = model(batch[0],batch[1])
                if batch_preds_tensor == None:
                    batch_preds_tensor = torch.mul(predictions,weight_list[indd]) 
                else:
                    batch_preds_tensor = torch.cat((batch_preds_tensor,torch.mul(predictions,weight_list[indd])),dim=1)
                    #batch_preds_tensor = torch.cat((batch_preds_tensor,predictions),dim=1)
                    #batch_preds_tensor = torch.add(batch_preds_tensor,predictions)

            #batch_avg_tensor = torch.div(batch_preds_tensor,batch_size_tensor)
            
            #print(batch_preds_tensor)
            #print(batch_avg_tensor)
            
            loss,meta_predictions = meta_model(batch_preds_tensor, batch[1])

            all_labels.extend(batch[1].tolist())
            all_preds.extend(torch.argmax(meta_predictions, dim=1).tolist())


            acc = accuracy(meta_predictions, batch[1])

            recall = compute_recall(batch[1], meta_predictions)

            f1 = compute_f1(batch[1],meta_predictions)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_recall += recall.item()
            epoch_f1 += f1.item()

    print(classification_report(all_labels, all_preds))

    return epoch_loss / batch_num, epoch_acc / batch_num, epoch_recall / batch_num, epoch_f1 / batch_num

def run_train(epochs, models,meta_model, train_dataloader, val_dataloader, optimizer, scheduler, fold_id,batch_size):
    best_valid_loss = float('+inf')
    best_valid_recall = float('-inf')
    best_valid_f1 = float('-inf')


    for epoch in range(epochs):

        print(f"epoch {epoch}: \n")

        # train the model
        train_loss, train_acc, train_recall = train(models,meta_model, train_dataloader, optimizer, scheduler)

        # evaluate the model
        valid_loss, valid_acc, valid_recall , valid_f1 = evaluate(models,meta_model,batch_size, val_dataloader)

        # save the best model
        if valid_recall>= best_valid_recall:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Here is saved@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            best_valid_recall = valid_recall
            torch.save(meta_model.state_dict(), f'new_meta_model_5_models/saved_weights_{fold_id}.pt')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train Recall: {train_recall * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% | Val. Recall: {valid_recall * 100:.2f}% | Val. F1: {valid_f1*100:.2f}%')


def evaluate_ensemble(models,batch_size,val_dataloader):
    print("ensemble evaluation started.....")
    epoch_loss = 0
    epoch_acc = 0
    epoch_recall = 0
    epoch_f1 = 0
    batch_num = len(val_dataloader) # number of batches
    
    
    all_preds = []
    all_labels = []
    batch_size_tensor = torch.Tensor([batch_size])
    weight_list = [0.4,0.2,0.4] #I want to find the right weigts that gives the best predictions
    with torch.no_grad():
        #for input, labels, batch_num in batch_iter(x_val, y_val, batch_size, shuffle=True):
        for step, batch in enumerate(tqdm(val_dataloader)):
            batch_preds_tensor = None
            for indd,model in enumerate(models):
                model.eval()
                loss,predictions = model(batch[0],batch[1])
                if batch_preds_tensor == None:
                    batch_preds_tensor = predictions   
                else:
                    #batch_preds_tensor = torch.cat((batch_preds_tensor,predictions),dim=1)
                    batch_preds_tensor = torch.add(batch_preds_tensor,torch.mul(predictions,weight_list[indd]))

            batch_avg_tensor = torch.div(batch_preds_tensor,batch_size_tensor)
            
            #print(batch_preds_tensor)
            #print(batch_avg_tensor)
            
            #loss,meta_predictions = meta_model(batch_preds_tensor, batch[1])

            all_labels.extend(batch[1].tolist())
            all_preds.extend(torch.argmax(batch_avg_tensor, dim=1).tolist())


            acc = accuracy(batch_avg_tensor, batch[1])

            recall = compute_recall(batch[1], batch_avg_tensor)

            f1 = compute_f1(batch[1],batch_avg_tensor)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_recall += recall.item()
            epoch_f1 += f1.item()

    print(classification_report(all_labels, all_preds))

    return epoch_loss / batch_num, epoch_acc / batch_num, epoch_recall / batch_num, epoch_f1 / batch_num
def eval_main():
    lr = 5e-5
    batch_size = 100
    num_classes = 2

    hidden_size1 = 768
    hidden_size2 = 1024
    
    num_epochs=50
    
    for fold_id in [0,1,2,3,4]:
        print(f"\n*********************{fold_id}***************************\n")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time = ",current_time)
        print("\n")
        
        #meta_model =Meta_MLP(8,2)

         # optimization algorithm
        #optimizer = torch.optim.Adam(meta_model.parameters(), lr=lr)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,verbose=True)
        
        #biased model dropout 0.5, alpha=0.10, gamma=4
        model_1 = MLP(hidden_size1, hidden_size2, num_classes, 0.5,alpha=0.1,gamma=4)
        model_1.load_state_dict(torch.load(f'fresh_filter/saved_weights_{fold_id}.pt'))

        #second model dropout=0.3, alpha=None, gamma=0
        model_2 = MLP(hidden_size1, hidden_size2,  num_classes, 0.3,alpha=None,gamma=0)
        model_2.load_state_dict(torch.load(f'ours_dorpout_0.3_alpha_none_gamma_0/saved_weights_{fold_id}.pt'))

        
        model_3 = MLP(hidden_size1, hidden_size2,  num_classes, 0.5,alpha=0.3,gamma=0)
        model_3.load_state_dict(torch.load(f'ours_dorpout_0.5_alpha_0.3_gamma_0/saved_weights_{fold_id}.pt'))

        model_4 = MLP(hidden_size1, hidden_size2,  num_classes, 0.5,alpha=0.3,gamma=0)
        model_4.load_state_dict(torch.load(f'ours_dorpout_0.5_alpha_0.7_gamma_4/saved_weights_{fold_id}.pt'))

        model_epochs_50 = MLP(hidden_size1, hidden_size2,  num_classes, 0.3,alpha=None,gamma=0)
        model_epochs_50.load_state_dict(torch.load(f'very_long_single_model_training/saved_weights_{fold_id}.pt'))
        
        #models= [model_1,model_2,model_3,model_4]
        models = [model_1,model_4,model_epochs_50]

        x_train, y_train = gu.get_train_data(fold_id)
        train_dataloader = gu.get_dataloader(x_train,y_train,batch_size)
        #train_dataloader = get_dataloader_with_contrastive_embeding(x_train,y_train,batch_size)

        x_val,y_val= gu.get_val_data(fold_id)
        val_dataloader = gu.get_dataloader(x_val,y_val,batch_size)


        evaluate_ensemble(models,batch_size,val_dataloader)
        # train and evaluate
        #run_train(num_epochs,models,meta_model,train_dataloader,val_dataloader,optimizer,scheduler,fold_id,batch_size)

def main():
    lr = 5e-5
    batch_size = 100
    num_classes = 2

    hidden_size1 = 768
    hidden_size2 = 1024
    
    num_epochs=10
    
    for fold_id in [0,1,2,3,4]:
        print(f"\n*********************{fold_id}***************************\n")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time = ",current_time)
        print("\n")
        
        meta_model =Meta_MLP(6,2)

         # optimization algorithm
        optimizer = torch.optim.Adam(meta_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,verbose=True)
        
        #biased model dropout 0.5, alpha=0.10, gamma=4
        model_1 = MLP(hidden_size1, hidden_size2, num_classes, 0.5,alpha=0.1,gamma=4)
        model_1.load_state_dict(torch.load(f'fresh_filter/saved_weights_{fold_id}.pt'))

        #second model dropout=0.3, alpha=None, gamma=0
        model_2 = MLP(hidden_size1, hidden_size2,  num_classes, 0.3,alpha=None,gamma=0)
        model_2.load_state_dict(torch.load(f'ours_dorpout_0.3_alpha_none_gamma_0/saved_weights_{fold_id}.pt'))

        #model_3 = MLP(hidden_size1, hidden_size2,  num_classes, 0.5,alpha=0.3,gamma=0)
        #model_3.load_state_dict(torch.load(f'ours_dorpout_0.5_alpha_0.3_gamma_0/saved_weights_{fold_id}.pt'))

        #model_4 = MLP(hidden_size1, hidden_size2,  num_classes, 0.5,alpha=0.3,gamma=0)
        #model_4.load_state_dict(torch.load(f'ours_dorpout_0.5_alpha_0.7_gamma_4/saved_weights_{fold_id}.pt'))
        
        model_5 = MLP(hidden_size1, hidden_size2,  num_classes, 0.5,alpha=0.3,gamma=2)
        model_5.load_state_dict(torch.load(f'very_long_single_model_training/saved_weights_{fold_id}.pt'))

        models= [model_1,model_2,model_5]

        x_train, y_train = gu.get_train_data(fold_id)
        train_dataloader = gu.get_dataloader(x_train,y_train,batch_size)
        #train_dataloader = get_dataloader_with_contrastive_embeding(x_train,y_train,batch_size)

        x_val,y_val = gu.get_val_data(fold_id)
        val_dataloader = gu.get_dataloader(x_val,y_val,batch_size)

        #evaluate_ensemble([model_1,model_2,model_3,model_4],val_dataloader)
        # train and evaluate
        run_train(num_epochs,models,meta_model,train_dataloader,val_dataloader,optimizer,scheduler,fold_id,batch_size)

main()
#eval_main()

