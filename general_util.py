import numpy as np
import pandas as pd
import glob
import os
import sys

import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader

sys.path.append('../')

from sklearn.feature_extraction.text import TfidfVectorizer
from simpletransformers.language_representation import RepresentationModel
from datasets import load_dataset, concatenate_datasets
import logging

data_path = "data/exp_data/"
folder= "bert_centoids_as_cls"

os.environ['CUDA_VISIBLE_DEVICES']="0"

model = RepresentationModel(
    model_type="bert",
    model_name="bert-base-uncased",
    use_cuda=True
)

def load_data(fold):
    if fold == 0:
        dataset = load_dataset('json', data_files={
            'train': [data_path + 'subset-0.json', data_path + 'subset-1.json', data_path + 'subset-2.json',
                      data_path + 'subset-3.json'],
            'validation': data_path + 'subset-4.json'})
    elif fold == 1:
        dataset = load_dataset('json', data_files={
            'train': [data_path + 'subset-1.json', data_path + 'subset-2.json', data_path + 'subset-3.json',
                      data_path + 'subset-4.json'],
            'validation': data_path + 'subset-0.json'})
    elif fold == 2:
        dataset = load_dataset('json', data_files={
            'train': [data_path + 'subset-0.json', data_path + 'subset-2.json', data_path + 'subset-3.json',
                      data_path + 'subset-4.json'
                      ],
            'validation': data_path + 'subset-1.json'})
    elif fold == 3:
        dataset = load_dataset('json', data_files={
            'train': [data_path + 'subset-0.json', data_path + 'subset-1.json', data_path + 'subset-3.json',
                      data_path + 'subset-4.json'],
            'validation': data_path + 'subset-2.json'})
    elif fold == 4:
        dataset = load_dataset('json', data_files={
            'train': [data_path + 'subset-0.json', data_path + 'subset-1.json', data_path + 'subset-2.json',
                      data_path + 'subset-4.json'],
            'validation': data_path + 'subset-3.json'})

    return dataset

def load_fold_data(fold):
    train_df=[]
    test_df = []
    if fold == 0:
        fold_training_files = glob.glob(os.path.join(data_path,'subset-[0,1,2,3].json'))

        train_df = pd.concat((pd.read_json(f) for f in fold_training_files), ignore_index=True)

        test_df = pd.read_json(data_path + 'subset-4.json')
    elif fold == 1:
        fold_training_files = glob.glob(os.path.join(data_path, 'subset-[1,2,3,4].json'))

        train_df = pd.concat((pd.read_json(f) for f in fold_training_files), ignore_index=True)

        test_df = pd.read_json(data_path + 'subset-0.json')
    elif fold == 2:
        fold_training_files = glob.glob(os.path.join(data_path, 'subset-[0,2,3,4].json'))

        train_df = pd.concat((pd.read_json(f) for f in fold_training_files), ignore_index=True)

        test_df = pd.read_json(data_path + 'subset-1.json')
    elif fold == 3:
        fold_training_files = glob.glob(os.path.join(data_path, 'subset-[0,1,3,4].json'))

        train_df = pd.concat((pd.read_json(f) for f in fold_training_files), ignore_index=True)

        test_df = pd.read_json(data_path + 'subset-2.json')
    elif fold == 4:
        fold_training_files = glob.glob(os.path.join(data_path, 'subset-[0,1,2,4].json'))

        train_df = pd.concat((pd.read_json(f) for f in fold_training_files), ignore_index=True)

        test_df = pd.read_json(data_path + 'subset-3.json')

    train_df = train_df
    return train_df,test_df




    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)


def get_bert_embedding(contexts_to_embed=[]):
    sentence_vectors = model.encode_sentences(contexts_to_embed, combine_strategy="mean")
    print("bert embedding step is done.")
    return sentence_vectors

def get_train_data(fold):
    dataset = load_data(fold)
    #train_data = data_set["train"]

    #balance the data
    posi_sample = dataset["train"].filter(lambda example: example["label"]==1).select(range(5))
    neg_sample = dataset["train"].filter(lambda example: example["label"]==0 ).select(range(len(posi_sample)))#.select(range(5))

    train_data = concatenate_datasets([posi_sample, neg_sample])

    contexts = []
    labels = []
    for context, label in zip(train_data['context'], train_data['label']):
        contexts.append(context)
        labels.append(label)

    return contexts,np.array(labels)

def get_tain_data_embedded(fold):
    dataset = load_data(fold)
    #train_data = data_set["train"]

    posi_sample = dataset["train"].filter(lambda example: example["label"]==1)
    neg_sample = dataset["train"].filter(lambda example: example["label"]==0 ).select(range(len(posi_sample)))

    train_data = concatenate_datasets([posi_sample, neg_sample])

    p_contexts_to_embed = []
    p_contexts_to_embed_masked = []
    n_contexts_to_embed = []
    for context, masked_context, label in zip(train_data['context'], train_data['masked_context'], train_data['label']):
        if label == 0:
            n_contexts_to_embed.append(context)
        else:
            p_contexts_to_embed.append(context)
            p_contexts_to_embed_masked.append(masked_context)
    print(f"{len(p_contexts_to_embed)}:{len(p_contexts_to_embed_masked)}:{len(n_contexts_to_embed)}")

    # get the embedding
    p_embedded_samples = get_bert_embedding(contexts_to_embed=p_contexts_to_embed)
    p_masked_embedded_samples = get_bert_embedding(contexts_to_embed=p_contexts_to_embed_masked)
    n_embedded_samples = get_bert_embedding(contexts_to_embed=n_contexts_to_embed)

    return p_embedded_samples, p_masked_embedded_samples, n_embedded_samples

#get train data for xgboosting
def get_train_xgbt_data(fold):
    dataset = load_data(fold)
    #train_data = data_set["train"]

    #balance the data
    posi_sample = dataset["train"].filter(lambda example: example["label"]==1).select(range(10))
    neg_sample = dataset["train"].filter(lambda example: example["label"]==0 ).select(range(len(posi_sample)))

    train_data = concatenate_datasets([posi_sample, neg_sample])

    contexts = []
    labels = []
    for context, label in zip(train_data['context'], train_data['label']):
        contexts.append(context)
        labels.append(label)

    return contexts,np.array(labels)

def get_val_data(fold):
    dataset = load_data(fold)

    val_data = dataset["validation"].select(range(10))

    contexts = []
    labels = []
    for context, label in zip(val_data['context'],val_data['label']):
        contexts.append(context)
        labels.append(label)

    return contexts, np.array(labels)

#prepare data for spacy
def get_spacy_train_data(fold):
    dataset = load_data(fold)

    posi_samples = dataset["train"].filter(lambda example: example["label"]==1).select(range(10))
    neg_samples = dataset["train"].filter(lambda example: example["label"]==0).select(range(len(posi_samples)))
    train_data = concatenate_datasets([posi_samples,neg_samples])
    val_data = dataset["validation"]

    return train_data.data.to_pylist(),val_data.data.to_pylist()

def get_dataloader(x,y,batch_size,embed_type = "bert"):
    #get embedding
    if embed_type == "bert":
        x_train= get_bert_embedding(x)
    elif embed_type == "tfidf":
        tfidf_engine = get_tfidf_engine(x)

        # get tfidf embedding of train data
        x_train = tfidf_engine.transform(x)
        #transfer sparse matrix to numpy format
        x_train = x_train.toarray()
        x_train

    #convert to tensors
    x= torch.Tensor(x_train)
    y = torch.LongTensor(y)
    #create dataloader
    tmp_DataSet = TensorDataset(x, y)
    tmp_Sampler = RandomSampler(tmp_DataSet)
    _DataLoader = DataLoader(tmp_DataSet, batch_size=batch_size, sampler=tmp_Sampler)

    return _DataLoader

def get_tfidf_engine(train_data):
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', min_df=5, token_pattern=r'\w{1,}',
                                 stop_words="english") #,max_features=10000)
    tfidf_vect.fit(train_data)

    return tfidf_vect

#create logger with a given file
def get_my_logger(file_name):
    logger = logging.getLogger("Yousef's Logger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"{file_name}.log",mode="w")
    logger.addHandler(handler)

    formatter = logging.Formatter(
        "%(levelname)s - %(name)s - %(filename)s - %(funcName)s - %(lineno)d - %(asctime)s: %(message)s")
    handler.setFormatter(formatter)

    return logger


def sort_by_indexes(lst, indexes, reverse=False):
  return [val for (_, val) in sorted(zip(indexes, lst), key=lambda x: \
          x[0], reverse=reverse)]

'''
This function takes two lists the fist contains dataset mentions and the second contains their indexes in the text. It order the indexes
ascendingly and then order the mentions based on the order indexes. The returned lists contain the dataset mentions ordered by their apearance in the 
text
'''
def order_mentions_by_index(a_datasets,b_indexes):

    if len(a_datasets)== 0 :
        return [],[]

    ds = np.array(b_indexes)
    ordered_indices = np.argsort(ds)

    a_datasets = sort_by_indexes(a_datasets,ordered_indices)
    b_indexes = sort_by_indexes(b_indexes,ordered_indices)

    return a_datasets,b_indexes
