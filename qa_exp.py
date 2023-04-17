import gc
import torch.cuda

import numpy as np
import evaluate
from transformers import AutoModelForQuestionAnswering,AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
import collections
import general_util as gu

#model_checkpoint = "microsoft/deberta-base"
model_checkpoint = "roberta-base"
#model_checkpoint = "bert-base-uncased"
#model_checkpoint= "deepset/minilm-uncased-squad2"
#tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_length = 384
stride = 128
n_best = 20
max_answer_length = 75
predicted_answers = []
batch_size = 4
lr = 2e-5
num_train_epochs = 3
num_gradient_acc = 2

metric = evaluate.load("squad_v2")
# clone repo to local folder

queries = ["What data are used?","Is there any use of data collected from a survey?","Which dataset or database is used?","On which data is the study based?","Which data samples or images are used?"]

def prepare_epoch_data(fold_id):
    global raw_datasets
    raw_datasets= gu.load_data(fold_id)


    train_dataset = raw_datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    ).select(range(10))

    validation_dataset =  raw_datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    ).select(range(10))


    train_dataset.set_format("torch")
    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")


    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=batch_size

    )
    eval_dataloader = DataLoader(
        validation_set, collate_fn=default_data_collator, batch_size=batch_size
    )

    return train_dataloader, eval_dataloader, validation_dataset

def preprocess_training_examples(examples):
    #select the question to be used
    if query_index >0:
        questions = [queries[query_index]]*len(examples["question"])
    else:
        questions= [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_examples(examples):
    #select the question
    if query_index >0:
        questions = [queries[query_index]]*len(examples["question"])
    else:
        questions= [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    #if start_index == 0 and end_index==0:
                    #    answers.append({"text":"","logit_score" :start_logit[start_index]+end_logit[end_index]})
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    elif (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    else:
                        answers.append({
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                                       })
                    #answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"].strip(),"no_answer_probability": 0.0}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": "","no_answer_probability": 1.0})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def train():
    best_HasAns_f1 = float('-inf')
    
    accelerator = Accelerator(mixed_precision='fp16',gradient_accumulation_steps=num_gradient_acc)

    if accelerator.is_main_process:
        logger = gu.get_my_logger(f"roberta_base_normal_q3_with_extended_vocab")
    for q_i in [0,1,2,3,4]:
        global query_index
        query_index = q_i

        
        if accelerator.is_main_process:
            #logger = my_u.get_my_logger(f"deberta_grad_new_tok_q_{query_index}")
            logger.info(
                f"this experiment uses roberta  base with q{query_index} with handling impossible answers. It used batch size of 64 with gradient accumulatin of 2. lr=2e-5 and 3 epochs\n\n")

        for k in [0,1,2,3,4]:#range(5):
            output_dir = f"roberta_base_with_ext_vocab/q_{q_i}/fold_{k}/"+model_checkpoint
            #load the model and the tokenizer
            global tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

            # COMMIT THIS CODE WHEN USING Original TOKENIZER
            #modify tokenizer and model
            tmp_tokenizer = AutoTokenizer.from_pretrained("tokenizers/my_roberta_tokenizer") #HER PROVIDE THE CUSTOME TOKENIZER FOLDER
            original_vocab = tokenizer.vocab.keys()
            new_vocab = tmp_tokenizer.vocab.keys()
            tokenizer.add_tokens(list(set(new_vocab)-set(original_vocab)))
            model.resize_token_embeddings(len(tokenizer))


            train_dataloader,eval_dataloader, validation_dataset = prepare_epoch_data(k)

            optimizer = AdamW(model.parameters(), lr=lr)

            num_update_steps_per_epoch = len(train_dataloader)
            num_training_steps = num_train_epochs * num_update_steps_per_epoch

            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
            )

            # parallalize the code using accelerator
            model, optimizer, train_dataloader, eval_dataloader,lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader,lr_scheduler)

            if accelerator.is_main_process:
                logger.info(f"\n\n*****************fold {k} **********************\n")
                best_HasAns_f1 = float('-inf')

            #progress_bar = tqdm(range(num_training_steps))

            for epoch in range(num_train_epochs):
                #logger.info(f"\n*************epoch {epoch}******************\n")
                # Training
                model.train()
                accelerator.print("\nTraining...\n")
                for batch in tqdm(train_dataloader):
                    with accelerator.accumulate(model):
                        #inputs, targets = batch
                        outputs = model(**batch)
                        loss = outputs.loss
                        accelerator.backward(loss)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    #if accelerator.is_main_process:
                        #progress_bar.update(1)

                # Evaluation
                model.eval()
                start_logits = []
                end_logits = []
                accelerator.print("\nEvaluation!\n")
                for batch in tqdm(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)

                    start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
                    end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

                start_logits = np.concatenate(start_logits)
                end_logits = np.concatenate(end_logits)
                start_logits = start_logits[: len(validation_dataset)]
                end_logits = end_logits[: len(validation_dataset)]

                metrics = compute_metrics(
                    start_logits, end_logits, validation_dataset, raw_datasets["validation"]
                )
                if accelerator.is_main_process:
                    logger.info(f"\nValidation results of epoch {epoch}:{metrics}\n")
                    
                accelerator.wait_for_everyone()
                if metrics["HasAns_f1"]>best_HasAns_f1:
                    best_HasAns_f1 = metrics["HasAns_f1"]
                     # Save and upload
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                    tokenizer.save_pretrained(output_dir)

                gc.collect()
                torch.cuda.empty_cache()




train()

