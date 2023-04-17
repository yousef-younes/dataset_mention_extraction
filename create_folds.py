
'''
This file contains functions to divide the dataset into 5 folds each with unique dataset IDs.
'''

import logging
from rank_bm25 import BM25Okapi

import pandas as pd
import json
import dataset_id_processings as idp
import general_util as gu

data_path = 'data\\exp_data\\'
logger = gu.get_my_logger("logs_create_folds")


'''
This function return the dataset ID with its support. 
Just for clarification purposes, you can activate line 31 to create a file that lists the calculated support. (-1) is used to indicate no dataset.
'''
def get_support_ds_id():
    df = pd.read_csv(data_path+'all_samples.csv',sep="$")

    id_count = df["dataset_id"].explode().value_counts()
    ids = list(id_count.index)
    freq = list(id_count.values)

    #save the dataset ID along with its support
    #id_count.to_csv(data_path+"support.csv",columns=['dataset_id'],sep="$")

    return ids,freq

'''
This function divides the dataset into 5 folds.
k is the number of folds to divide the data into
'''
def split_data(k):
    #get support
    dataset_ids, counts = get_support_ds_id()

    total_per =0
    for id,freq in zip(dataset_ids,counts):
        percentage = freq/sum(counts)
        logger.info(f"{id}:{percentage:.6f}")
        total_per += percentage
    #statistics
    total_num_of_samples =sum(counts)
    num_samples_per_fold = total_num_of_samples/k
    percent_of_neg_samples = 0.844
    percent_of_pos_samples = 0.156
    percent_of_neg_samples_per_fold = percent_of_neg_samples/k
    percent_of_pos_samples_per_fold = percent_of_pos_samples/k
    number_of_neg_samples_per_fold = num_samples_per_fold*percent_of_neg_samples
    number_of_pos_samples_per_fold = num_samples_per_fold*percent_of_pos_samples

    logger.info(f"Total number of samples: {total_num_of_samples}")
    logging.info(f"Percentage of samples without dataset mentions: {percent_of_neg_samples}")
    logger.info(f"Percentage of samples with dataset mentions: {percent_of_pos_samples}")
    logger.info(f"Number of samples per fold {num_samples_per_fold:.3f}")
    logger.info(f"Percentage of Neg samples per fold {percent_of_neg_samples_per_fold}")
    logger.info(f"Percentage of Pos samples per fold {percent_of_pos_samples_per_fold}")
    logger.info(f"Number of Neg samples per fold {number_of_neg_samples_per_fold:.2f}")
    logger.info(f"Number of Pos samples per fold {number_of_pos_samples_per_fold:.2f}")


    class_combin= divide_dataset_ids(dataset_ids[1:],counts[1:],k)

    for sub in class_combin:
        indices = [dataset_ids.index(x) for x in sub]

        sum_lit = [counts[x] for x in indices]

        print(sum(sum_lit))

    return class_combin

#This function divide the dataset ids into five disjoint sets
def divide_dataset_ids(ds_ids,freqs,num_of_folds):

    large_ids, mid_ids, small_ids = [], [], []
    large_freq, mid_freq, small_freq = [], [], []

    # divide the ids into three lists based on their support
    for id, count in zip(ds_ids, freqs):
        if count > 2000:
            large_ids.append(id)
            large_freq.append(count)
        elif count > 100 and count < 1000:
            mid_ids.append(id)
            mid_freq.append(count)
        else:
            small_ids.append(id)
            small_freq.append(count)

    # initial split
    class_combination = []

    for i in range(num_of_folds):
        subclass = []

        subclass.append(large_ids[i])

        if i > 1:
            subclass.extend(mid_ids[- 5:])
            mid_ids = mid_ids[:- 5]
            mid_freq = mid_freq[: - 5]
        else:
            subclass.extend(mid_ids[- 4:])
            mid_ids = mid_ids[:- 4]
            mid_freq = mid_freq[: - 4]

        subclass.extend(small_ids[-3:])
        small_ids = small_ids[:-3]
        small_freq = small_freq[:-3]

        class_combination.append(subclass)

    logger.info(class_combination)


    return class_combination

#This function finds the best question among our list of five questions that matches a given context using BM25
def get_best_matching_question(context):
    queries = ["What data are used?",
               "Is there any use of data collected from a survey?",
               "Which dataset or database is used?",
               "On which data the study is based?",
               "Which data samples or images are used?"]
    tokenized_questions = [q.split(" ") for q in queries]

    bm25 = BM25Okapi(tokenized_questions)

    tokenized_context = context.split(" ")
    question = bm25.get_top_n(tokenized_context, tokenized_questions, n=1)
    return " ".join(question[0])

#this function creates five folds that contains unique dataset IDs and has the best matching question for a context
def create_folds_with_unique_dataset_ids():
    df = pd.read_csv(data_path+"all_samples.csv",sep=str('$'))
    dataset_id_dict=idp.read_dataset_ids_file()
    #get combinations of dataset IDs for five folds
    class_combinations = split_data(5)
    data = []

    negs_df = df[df.dataset_id == -1]
    block_size = len(negs_df)/5

    counter_for_id=0

    for j,c_c in enumerate(class_combinations):
        neg_df = negs_df.loc[j * block_size: j * block_size + block_size:1]
        pos_df = df[df["dataset_id"].isin(c_c)]
        temp_df = [neg_df,pos_df]

        logger.info(f"{j}:{len(pos_df)}:{len(neg_df)}")
        cur_df = pd.concat(temp_df)
        #shuffle the frame
        cur_df = cur_df.sample(frac=1)

        for _, row in cur_df.iterrows():

            dataset_list = set()
            index_list = set()

            context = row["context"]
            ds_id = row['dataset_id']
            if ds_id != -1:
                dataset_list.add(row["dataset"])
                index_list.add(row["index"])

            #loop through the dictionary of dataset identifiers
            for k,v in dataset_id_dict.items():
                if k == ds_id:
                    continue

                tmp_list = list(dataset_id_dict[k])
                if len(tmp_list) > 1:
                    tmp_list.sort(key=len, reverse=True)

                #loop through the mention strings of a given dataset identifier
                for ds_mention in tmp_list:
                    ds_index = context.find(ds_mention)
                    if ds_index != -1:
                        dataset_list.add(ds_mention)
                        index_list.add(ds_index)
                        break

            context_without_ds_mentions = context
            for ds in dataset_list:
                if ds == 'None':
                    continue
                context_without_ds_mentions = context_without_ds_mentions.replace(ds, "")
            #get best question
            best_question = get_best_matching_question(context)
            data_item = {"id":f"{str(counter_for_id+1)}","context":f"{context}","question":f"{best_question}","answers":{'text':list(dataset_list),'answer_start':list(index_list)},"label":int(row['label']),"masked_context":f"{context_without_ds_mentions}"}
            counter_for_id+=1
            data.append(data_item)

        #serilize json
        json_string = json.dumps(data,indent=4)
        #write to json file
        with open(data_path+"subset-"+str(j)+".json", "w") as outfile:
            outfile.write(json_string)

        logger.info("file for subset {0} has been created".format(j))
        logger.info("size of file is {0}".format(len(data)))
        #clear data
        data = []


create_folds_with_unique_dataset_ids()







