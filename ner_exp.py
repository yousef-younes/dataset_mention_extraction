import os
import numpy as np

from transformers import AutoTokenizer, default_data_collator, TrainingArguments, Trainer, \
    AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
import transformers
import evaluate
import general_util as gu


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["WANDB_DISABLED"]="true"

task = "ner"
#model_checkpoint = "bert-base-uncased"
#model_checkpoint = "microsoft/deberta-base"
model_checkpoint = "roberta-base"
batch_size = 16
label_all_tokens = True
gradient_acc = 2
label_list = ['O','B-DS','I-DS',]

logger = gu.get_my_logger("log_roberta_ner")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,add_prefix_space=True)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
metric = evaluate.load("seqeval")


def prepare_fold_data(fold_id):
    raw_datasets= gu.load_data(fold_id)

    raw_datasets = raw_datasets.remove_columns(['question','masked_context'])
    train_dataset = raw_datasets["train"].select(range(10))
    train_dataset=train_dataset.map(
        prepare_data_for_ner,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    validation_dataset = raw_datasets["validation"].select(range(10))
    validation_dataset=validation_dataset.map(
        prepare_data_for_ner,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )

    raw_datasets['train']=train_dataset
    raw_datasets['validation']=validation_dataset

    return raw_datasets



def prepare_data_for_ner(examples):
    out_tokens = []
    ner_tags = []
    out_id_list=[]

    for id,context,answers,cur_label in zip(examples['id'],examples['context'],examples['answers'],examples['label']) :

       cur_tokens= context.split(' ') #split the context
       tags = []
       if cur_label==0: #no dataset mention in the text since (lable = 0)
           tags=[0]*len(cur_tokens)
       else: #there is dataset mention in the text (label=1)
           datasets = answers['text']
           if len(datasets) > 1:
               datasets,_=gu.order_mentions_by_index(answers['text'],answers['answer_start'])
           j=0
           i=0
           while i < len(cur_tokens):
               if j<len(datasets):
                ds = datasets[j].split()
               else:
                   ds = ["*(__46)_$("] #set mention to a string that is impossible to find in normal text
               if cur_tokens[i] == ds[0]:
                   tags.append(1)
                   tags.extend([2]*(len(ds)-1))
                   i+=len(ds)
                   #for temp_i in range(len(ds)):
                   #    if ds[temp_i] == cur_tokens[i] and temp_i==0:
                   #       tags.append(1)
                   #    elif ds[temp_i]== cur_tokens[i]:
                   #        tags.append(2)
                   #    i+=1
                   #    temp_i+=1
                   if len(datasets)>1:
                       j+=1
                       continue #to avoid incrementing i at the end of the loop
                   else:
                       remaining_zeros = len(cur_tokens)-len(tags)
                       tags.extend([0]*remaining_zeros)
                       break
               else:
                   tags.append(0)
               i+=1


       #make sure the text and label lists are of the same length
       assert len(cur_tokens)==len(tags), f"{len(cur_tokens)}:{len(tags)}"
       out_id_list.append(id)
       out_tokens.append(cur_tokens)
       ner_tags.append(tags)


    inputs = {}
    inputs['id']=out_id_list
    inputs['tokens']=out_tokens
    inputs['ner_tags']=ner_tags
    return inputs


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    for k in [0,1,2,3,4]:
        print(f"\n**************{k} fold ********************\n") 
        logger.info(f"\n************************{k} fold ******************\n")
        datasets = prepare_fold_data(k)
        tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

        model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,num_labels=len(label_list))


        model_name = model_checkpoint.split("/")[-1]
        args = TrainingArguments(
            f"ner/{model_name}_k",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            gradient_accumulation_steps=2,
            #early_stopping_patience=8,
            weight_decay=0.01,
            push_to_hub=False,
        )


        data_collator = DataCollatorForTokenClassification(tokenizer)


        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        logger.info(trainer.evaluate())



main()
