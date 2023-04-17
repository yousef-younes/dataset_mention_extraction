import os

#from torch_lr_finder import LRFinder
import torch
from sklearn.metrics import recall_score, classification_report,f1_score

from tqdm import tqdm

import general_util as gu

import torch.nn as nn
from datetime import datetime
from MLP_2 import MLP

os.environ["CUDA_VISIBLE_DEVICES"]="2"

path="./"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#data_path = "../data/final_data/"

def accuracy(probs, target):
    winners = probs.argmax(dim=1)
    corrects = (winners == target)
    accuracy = corrects.sum().float() / float(target.size(0))
    return accuracy

def compute_recall(true_label,probs):
    preds = probs.argmax(dim=1)
    return recall_score(true_label,preds)

def comput_f1(true_label,probs):
    preds = probs.argmax(dim=1)
    return f1_score(true_label,preds)

def train(model, trainDataLoader, optimizer,scheduler):
    epoch_loss = 0
    epoch_acc = 0
    epoch_recall = 0
    batch_num = len(trainDataLoader)  # number of batches in the data

    for step, batch in enumerate(tqdm(trainDataLoader)):
        optimizer.zero_grad()

        loss,predictions = model(batch[0], batch[1])


        acc = accuracy(predictions, batch[1])
       

        recall = compute_recall(batch[1], predictions)

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


def evaluate(model,val_dataloader):
    epoch_loss = 0
    epoch_acc = 0
    epoch_recall = 0
    epoch_f1 = 0
    batch_num = len(val_dataloader) # number of batches
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        #for input, labels, batch_num in batch_iter(x_val, y_val, batch_size, shuffle=True):
        for step, batch in enumerate(tqdm(val_dataloader)):
            loss,predictions = model(batch[0],batch[1])

            all_labels.extend(batch[1])
            all_preds.extend(torch.argmax(predictions, dim=1).tolist())

            acc = accuracy(predictions, batch[1])

            recall = compute_recall(batch[1], predictions)

            f1 = comput_f1(batch[1],predictions)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_recall += recall.item()
            epoch_f1 += f1.item()

    print(classification_report(all_labels, all_preds))

    return epoch_loss / batch_num, epoch_acc / batch_num, epoch_recall / batch_num, epoch_f1 / batch_num


def run_train(epochs, model, train_dataloader, val_dataloader, optimizer, scheduler, fold_id):
    best_valid_loss = float('+inf')
    best_valid_recall = float('-inf')
    best_valid_f1 = float('-inf')


    for epoch in range(epochs):

        print(f"epoch {epoch}: \n")

        # train the model
        train_loss, train_acc, train_recall = train(model, train_dataloader, optimizer, scheduler)

        # evaluate the model
        valid_loss, valid_acc, valid_recall , valid_f1 = evaluate(model, val_dataloader)

        # save the best model
        if valid_loss <= best_valid_loss:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Here is saved@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'mpl_model/saved_weights_{fold_id}.pt')

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train Recall: {train_recall * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% | Val. Recall: {valid_recall * 100:.2f}% | Val. F1: {valid_f1*100:.2f}%')



def main():
    print("This experiment is using dropout=0.5, alpha=0.3, gamma=2")
    # hyper-parameters:
    lr = 5e-5
    lrs=[1.86E-02,6.93E-03,5.42E-03,2.29E-03,8.31E-03]
    batch_size = 50
    dropout_keep_prob = 0.5
    num_classes = 2

    hidden_size1 = 768
    hidden_size2 = 1024
    num_epochs = 50

    # loss function
    loss_func = nn.CrossEntropyLoss()

    for exp_no in range(5):

        print(f"\n*********************{exp_no}***************************\n")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time = ",current_time)
        print("\n")
        #lr = lrs[exp_no]
        # Build the model
        model = MLP(hidden_size1, hidden_size2,  num_classes, dropout_keep_prob,alpha=0.3,gamma=2)
       
        
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(f"No. Of all model parameters:{pytorch_total_params}\n")
        #trainable parameters
        pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"No. Of model trainable parameters: {pytorch_total_params}\n")

        # optimization algorithm
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,verbose=True)

        x_train, y_train = gu.get_train_data(exp_no)
        train_dataloader = gu.get_dataloader(x_train,y_train,batch_size,embed_type='bert')#'tfidf'
        #train_dataloader = gu.get_dataloader_with_contrastive_embeding(x_train,y_train,batch_size)

        x_val,y_val = gu.get_val_data(exp_no)
        val_dataloader = gu.get_dataloader(x_val,y_val,batch_size,embed_type='bert') #'tfidf'
        #val_dataloader = gu.get_dataloader_with_contrastive_embeding(x_val,y_val,batch_size)

        #pick_learning_rate(exp_no,train_dataloader,val_dataloader,model,loss_func,optimizer)

        # train and evaluate
        run_train(num_epochs,model,train_dataloader,val_dataloader,optimizer,scheduler,exp_no)

'''
#this function is to pick the learning rate for the fold
def pick_learning_rate(fold,train_dataloader,valid_dataloader,model,criterion,optimizer):
    desired_batch_size, real_batch_size = 64, 64
    accumulation_steps = desired_batch_size // real_batch_size

    model = model
    criterion = criterion
    optimizer = optimizer


    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")#cpu
    lr_finder.range_test(train_dataloader,valid_dataloader, end_lr=10, num_iter=100, step_mode="exp", accumulation_steps=accumulation_steps)
    lr_finder.plot()
    lr_finder.reset()
'''

main()




