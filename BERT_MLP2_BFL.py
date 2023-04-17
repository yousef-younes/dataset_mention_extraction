
import gc
import torch.cuda
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import sys
sys.path.append('../')
from BertWithMLP2Head import BertForSequenceClassificationWithFL
import os
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
import numpy as np
import time
import datetime

from sklearn.metrics import recall_score, fbeta_score, classification_report
import general_util as gu

kaggle_data_path = '../data/final_data/'


def get_available_devices():
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# function to calculate the recall
def compute_recall(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return recall_score(labels_flat, pred_flat)
#function to calculate the f_beta score
def compute_fbeta(preds,labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return fbeta_score(labels_flat, labels_flat,beta=3)

def format_time(elapsed):

    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

acc_res = []
recall_list = []
fbeta_list = []

best_recall = -1

def get_data_prepared(fold_no,tokenizer,gpu_count,data_type = "train"):

    # Get the lists of sentences and their labels.
    if data_type=="train":
        cur_txts, cur_labels = gu.get_train_data(fold_no)
    elif data_type=="val":
        cur_txts, cur_labels = gu.get_val_data(fold_no)

    cur_inputs = []
    cur_masks = []
    for txt in cur_txts:
        tmp = tokenizer(txt,padding='max_length',truncation=True,max_length=512)
        cur_inputs.append(tmp["input_ids"])
        cur_masks.append((tmp['attention_mask']))

    cur_inputs = torch.tensor(cur_inputs)
    cur_masks = torch.tensor(cur_masks)
    cur_labels = torch.tensor(cur_labels)
    cur_labels = cur_labels.to(torch.int64)

    batch_size = 16 * max(1, 1)

    # train data loader
    cur_dataSet = TensorDataset(cur_inputs, cur_masks, cur_labels)
    cur_sampler = RandomSampler(cur_dataSet)
    cur_dataLoader = DataLoader(cur_dataSet, batch_size=batch_size, sampler=cur_sampler)

    return cur_dataLoader



def train(output_dir,fold_no,writer):

    device, gpu_ids = get_available_devices()

    global acc_res ,recall_list, fbeta_list
    acc_res= []
    recall_list=[]
    fbeta_list = []


    # Load the BERT tokenizer and model.
    print('Loading Bert tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
    model = BertForSequenceClassificationWithFL.from_pretrained('bert-base-uncased', num_labels=2,
                                                                output_attentions=False, output_hidden_states=False)
    model.gamma = 4
    model.alpha = 0.1

    for param in model.bert.parameters():
        param.requires_grad = True

    # trainable parameters
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_trainable_params)



    trainDataLoader = get_data_prepared(fold_no,tokenizer,len(gpu_ids),"train")


    model = nn.DataParallel(model, gpu_ids)
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    epochs = 5

    total_num_steps = len(trainDataLoader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_num_steps)

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # recall
    # eval_recall = [0,0]

    # Store the average loss after each epoch so we can plot them.
    loss_values = []


    #this is added for tensorboard
    running_loss=0.0
    running_correct = 0.0

    # For each epoch...
    for epoch_i in range(0, epochs):

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.nan
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(tqdm(trainDataLoader)):


            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a backward pass
            model.zero_grad()

            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += torch.sum(loss)  # coss.item()

            # Perform a backward pass to calculate the gradients.
            loss.sum().backward()  # loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # torch.cuda.empty_cache()
            # gc.collect()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(trainDataLoader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss.item())

        print("")
        print("  Average training loss: {0:.3f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        print('loss so far')
        print(loss_values)
        #tensorboard
        writer.add_scalar("Avg Training Loss", avg_train_loss,epoch_i)

        #test the model on the validation set
        validation(output_dir,model,tokenizer,writer,gpu_ids,epoch_i,fold_no)



    # store training loss and validation acc in a file
    with open(output_dir + 'result.txt', 'w') as file:
        file.write('Avg Training loss\n')
        for item in loss_values:
            file.write("%s\t" % item)
        file.write("\n\n")

        file.write("Avg Validatin Accuracy\n")
        for item in acc_res:
            file.write("%s\t" % item)
        file.write("\n\n")

        file.write("Avg Validation recall: ")
        for item in recall_list:
            file.write("%s\t" % item)
        file.write(" \n\n ")

        file.write("Avg Fbeta score: ")
        for item in fbeta_list:
            file.write("%s\t"%item)
        file.write(" \n\n")

        file.close()

    print("")
    print("Training complete!")



def validation(output_dir, model,tokenizer,writer,gpu_ids,epoch_i,fold_no):

    print("")
    print("Running Validation...")

    #get val dataloader
    validationDataLoader = get_data_prepared(fold_no,tokenizer,len(gpu_ids),data_type="val")

    t0 = time.time()

    # tensorboard
    tb_labels = []  # store the labels for tensorboard
    tb_preds = []  # store predctions for tensorboard

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy, eval_recall,eval_fbeta = 0, 0, 0,0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validationDataLoader:
        # Add batch to GPU
        batch = tuple(t.cuda() for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.RobertaForSequenceClassification
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # tensorboard
        class_preds = [F.softmax(output, dim=0) for output in
                       outputs[0]]  # [F.softmax(outputs[0],dim=1)]#[F.softmax(output,dim=0) for output in outputs]
        labels_flat = label_ids.flatten()
        tb_labels.append(b_labels)
        tb_preds.append(class_preds)

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy


        # calculate the recall for this batch of validation sentence
        tmp_eval_recall = compute_recall(logits, label_ids)
        # accumulate the total recall
        eval_recall += tmp_eval_recall

        #calculatte the recall for this batch of validation examples
        temp_eval_fbeta = compute_fbeta(logits,label_ids)
        #accumulate the tottal fbeta
        eval_fbeta += temp_eval_fbeta

        # Track the number of batches
        nb_eval_steps += 1
    # print('recall values')
    # print('class A'+ str(eval_recall[0]/nb_eval_steps))
    # print('class B'+ str(eval_recall[1]/nb_eval_steps))

    # tensorboard: add pr_graph
    tb_preds = torch.cat([torch.stack(batch) for batch in tb_preds])
    tb_labels = torch.cat(tb_labels)


    print(classification_report(tb_labels.cpu(), tb_preds.argmax(dim=1).cpu()))

    classes = range(2)
    for i in classes:
        labels_i = tb_labels == i
        preds_i = tb_preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)

    # Report the final accuracy for this validation run.
    temp_acc = eval_accuracy / nb_eval_steps
    temp_recall = eval_recall / nb_eval_steps
    temp_fbeta = eval_fbeta/nb_eval_steps

    print("  Accuracy: {0:.3f}".format(temp_acc))
    print("  Recall  : {0:.3f}".format(temp_recall))
    print("  Fbeta   : {0:.3f}".format(temp_fbeta))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))


    if temp_recall > best_recall :
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print('first model saved')

    acc_res.append(temp_acc)
    recall_list.append(temp_recall)
    fbeta_list.append(temp_fbeta)

    print('AVG Accuracy so far:')
    print(acc_res)
    # tensorboard
    writer.add_scalar("Eval/Accuracy", acc_res[epoch_i], epoch_i)

    print("Recall so far: ")
    print(recall_list)
    writer.add_scalar("Eval/Recall", recall_list[epoch_i], epoch_i)

    print("fbeta so far: ")
    print(fbeta_list)
    writer.add_scalar("Eval/F_beta",fbeta_list[epoch_i],epoch_i)


    writer.close()


    print('validation of epoch '+str (epoch_i)+' completed')


def main():

    for i in range(5):
        print('')
        print('')
        print(f'€€€€€€€€€€€€€€€€€€€€€€€€€€€€ Training Exp {i} €€€€€€€€€€€€€€€€€€€€€€€€€€€€ ')

        # output directory to save the trained model
        output_dir = "bert_exp/exp_" + str(i) + "/"
        print(output_dir)
        #create summary writer for tensorboard
        tensorboard_file = 'save_tensorboard/bert_'+str(i)
        print(tensorboard_file)
        writer = SummaryWriter(tensorboard_file)
        #train and evaluate the model
        train(output_dir,i,writer)

        #clean memory
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()
