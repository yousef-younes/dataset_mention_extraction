from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
from transformers import DataCollatorWithPadding
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score,classification_report
import general_util as gu


#put model checkpoint here
model_checkpoint = "bert-base-uncased"
#model_checkpoint = "allenai/scibert_scivocab_uncased"
batch_size = 16
lr = 2e-5
num_train_epochs = 3
logger = gu.get_my_logger("log_bin_classification_results")


class classification_experiment():
    def __init__(self):
        self.tokenizer = None

    def tokenize(self, batch):
        return self.tokenizer(batch["masked_context"], padding=True, truncation=True)


    def tokenize_val(self, batch):
        return self.tokenizer(batch["context"], padding=True, truncation=True)

    def load_epoch_data(self,fold_id):
        dataset = gu.load_data(fold_id)

        # remove unused columns
        dataset=dataset.remove_columns(['id', 'question', 'answers'])
        dataset["train"]=dataset["train"].remove_columns(['masked_context'])
        dataset["validation"]=dataset["validation"].remove_columns(["masked_context"])

        data_collator = DataCollatorWithPadding(self.tokenizer)
        # tokenize the text samples
        train_dataset = dataset["train"]#.select(range(10))
        train_dataset = train_dataset.map(self.tokenize_val, batched=True, remove_columns=['context'])

        val_datast = dataset["validation"]#.select(range(10))
        val_datast = val_datast.map(self.tokenize_val, batched=True, remove_columns=['context'])#dataset["validation"].column_names)

        train_dataset.set_format("torch")
        val_datast.set_format("torch")

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=batch_size
        )
        eval_dataloader = DataLoader(
            val_datast, collate_fn=data_collator,
            batch_size=batch_size
        )

        return train_dataloader, eval_dataloader, val_datast

    def my_compute_metrics(self, labels, predictions):

        f1 = f1_score(labels, predictions)
        recall = recall_score(labels, predictions)
        precision = precision_score(labels, predictions)
        acc = accuracy_score(labels, predictions)


        return {"f1": f1, "recall": recall, "precision": precision, "acc": acc}

    def train(self):

        for k in range(5):

            output_dir = f"exp_{k}/classification" + model_checkpoint

            #load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=2)

            train_dataloader,eval_dataloader, validation_dataset = self.load_epoch_data(k)

            optimizer = AdamW(model.parameters(), lr=lr)

            # parallalize the code using accelerator
            accelerator = Accelerator(mixed_precision='fp16')
            model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader
            )

            if accelerator.is_main_process:
                logger.info(f"\n\n*****************fold {k} **********************\n")


            num_update_steps_per_epoch = len(train_dataloader)
            num_training_steps = num_train_epochs * num_update_steps_per_epoch

            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
            )

            progress_bar = tqdm(range(num_training_steps))

            for epoch in range(num_train_epochs):
                # Training
                model.train()
                accelerator.print("\nTraining...\n")
                for step, batch in enumerate(train_dataloader):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)

                # Evaluation

                labels,preds= [],[]

                model.eval()

                accelerator.print("\nEvaluation!\n")
                for batch in tqdm(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)

                    logits = outputs.logits
                    predictions = torch.argmax(logits,dim=-1)
                    preds.append(accelerator.gather(predictions).cpu().numpy())
                    labels.append(accelerator.gather(batch["labels"]).cpu().numpy())


                labels = np.concatenate(labels)
                preds = np.concatenate(preds)



                metrics = self.my_compute_metrics(labels,preds)
                
                if accelerator.is_main_process:
                    logger.info(f"Validation results of epoch {epoch}:{metrics}")
                    report = classification_report(labels,preds)
                    logger.info("\n"+report)
                # Save and upload
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    self.tokenizer.save_pretrained(output_dir)
                    # repo.push_to_hub(
                    #    commit_message=f"Training in progress epoch {epoch}", blocking=False
                    # )



def main():

    exp = classification_experiment()
    exp.train()


if __name__ == '__main__':
    main()
