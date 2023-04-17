
import json
import random
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords
import string
import dataset_id_processings as idp
import general_util as gu


data_path = 'data/'
num_words = 20

"this function is handle the kaggle training data and creates an csv file that contains contexts annotated with dataset mentioned in them"
def process_kaggle_data_with_id():
    df = pd.read_csv(data_path + 'train.csv')

    #add a new colum that Matches every dataset tile to its corresponding dataset Id
    df["dataset_id"] = df["dataset_title"].apply(idp.get_dataset_id)

    #xg = df[df.duplicated(["Id","dataset_id"],keep='first')]

    #remove duplicate papers which has more than one dataset mention
    df = df[~df.duplicated(["Id", "dataset_id"], keep='first')]

    #df = df[~df.duplicated(['file_id', 'sample_text', 'dataset', 'label'], keep='first')]

    # create dictionary of the structure {file_id:(pub_title,dataset_ID)}
    dic = {}
    j = 0
    pub = set()

    for i, row in df.iterrows():
        id = row['Id']
        if id not in dic.keys():
            j += 1

            if len(row['pub_title']) > 5:
                pub.add(row['pub_title'])

        # compute similarity
        dic[id] = (row['pub_title'], row["dataset_id"])

    #print(len(pub))
    #print(j)

    # get list of samples
    data = handle_files(dic)

    # create dataframe
    dframe = pd.DataFrame(data,
                          columns=['file_id', 'publication_name', 'section_title', 'context', 'dataset', 'index','label','dataset_id'])

    print(len(dframe))
    # remove all rows with section_txt less than 20 chars
    dframe = dframe[dframe.context.str.len().gt(20)]
    print(
        'number of samples after removing all sections whose text length is less than 20 chars: {}'.format(len(dframe)))
    # remove all repetitions (file_id, section_txt, dataset, label)
    dframe = dframe[~dframe.duplicated(['file_id', 'context', 'dataset', 'label'], keep='first')]
    print(
        'number of samples after removing all sections whose file_id,section_txt,dataset,label are the same: {}'.format(
            len(dframe)))

    # remove all repetitions (section_txt, dataset, label)
    dframe = dframe[~dframe.duplicated(['context', 'dataset', 'label'], keep='first')]
    print('number of samples after removing all sections whose section_txt,dataset,label are the same: {}'.format(
        len(dframe)))

    dframe = dframe[~dframe.duplicated(['context', 'label'], keep='first')]
    print('number of samples after removing all sections whose section_txt,label are the same: {}'.format(len(dframe)))

    # save dataframe to file
    dframe.to_csv(data_path + 'exp_data\\all_samples.csv', index=None, sep=str('$'))


dataset_id_dict = {}


'''
    this function will loop through the files, read the sections, and search for the dataset_mention_strings. It will retrun a list 
    of samples that represent the dataset.
'''
def handle_files(dic):
    # this list will hold all the samples while processing the dataset
    data = []

    #read the dict of dataset ids
    dataset_id_dict = idp.read_dataset_ids_file()

    for file_id, pub_metadata in dic.items():

        publication_name = pub_metadata[0]

        f = open(data_path + "\\train\\" + file_id + '.json')
        text_content = json.load(f)

        #get the dataset group of strings
        dataset_mention_strings = list(dataset_id_dict[pub_metadata[1]])
        # sort the dataset mentions in descending order with regard to length because we want to find the longest string pssible
        if len(dataset_mention_strings) >1:
            dataset_mention_strings.sort(key=len,reverse=True)
        #dataset_mention_string = sorted(dataset_mention_strings, key=len, reverse=False)

        # loop through the json pairs
        for section in text_content:
            section_title = section['section_title']

            section_txt = section['text']

            if len(section_txt) < 5:
                #print(section_txt)
                continue

            # the dollar sign will be used as separator so all of its occurances will be removed from the data
            section_txt = section_txt.replace('$', ' ')

            # look for the datasets starting from the one with the shortest length
            sample_to_add = do_the_matching(section_txt,dataset_mention_strings)

            if sample_to_add[1] == 'None':
                data_item =(file_id, publication_name, section_title, sample_to_add[0], sample_to_add[1], sample_to_add[2],sample_to_add[3],int(-1))
            else:
                data_item =(file_id, publication_name, section_title, sample_to_add[0], sample_to_add[1], sample_to_add[2],sample_to_add[3],int(pub_metadata[1]))

            data.append(data_item)

    return data


'''
search for datasets in a section
'''
def do_the_matching(section_txt,dataset_mention_strings):

    for ds_mention in dataset_mention_strings:
        sample_text = ""
        ds_index = section_txt.find(ds_mention)
        if ds_index != -1:
            before_mention = section_txt[:ds_index].strip().split()
            after_mention =  section_txt[ds_index + len(ds_mention):].strip().split()

            sample_text = " ".join(before_mention[-num_words:]) + " " + ds_mention + " " + " ".join(after_mention[:num_words])

            ds_index = sample_text.find(ds_mention)

            data_item = (sample_text,ds_mention,ds_index,int(1))

            return data_item

    sample_text = section_txt.split()
    sample_text = " ".join(sample_text[0:2*num_words+1])
    data_item = (sample_text,'None',ds_index,int(0))

    return data_item

'''
This function will search for the existence of datasets in one
'''
def compute_support_ds_id():
    df = pd.read_csv(data_path+'exp_data\\all_samples_manual_ids.csv',sep="$")

    id_count = df["dataset_id"].explode().value_counts()
    id_count.to_csv(data_path+"exp_data\\support.csv",sep="$")


def get_stats(file):

    print("Getting all statstics: ")
    train = pd.read_csv(file,sep=str('$'))

    all_len = len(train)
    pos_len = len(train[train.label == 1])
    neg_len = len(train[train.label == 0])
    print('number of samples: {}'.format(all_len))
    print('number of positive samples: {}'.format(pos_len))
    print('number of negative samples: {}'.format(neg_len))
    print('percentage of positive samples out of the whole training data {}%'.format(pos_len*100/ all_len))
    print('percentage of negative samples out of the whole training data {}%'.format(neg_len * 100 / all_len))
    train_datasets = set(train.dataset.unique())
    print('number of unique datasets: {}'.format(len(train_datasets)))
    #train_datasets_id = set(train.dataset_ID.unique())
    #print(' number of unique id is {}'.format(len(train_datasets_id)))
    print("datasets are:")
    print(train_datasets)
    print(train.dataset.apply(lambda x: len(x)).value_counts())

    samp = train["dataset"].explode().value_counts()
    samp.to_csv(data_path+"\\new_stats\\revers_new_support.csv",sep="$")


'''
read processed data and divide it controabbly into 20% test set and 80% train+dev sets such that there is no overlap between 
the two sets
'''
def controlled_divide_processed_data():

    df = pd.read_csv(data_path+'masked_2\\all_samples_wiout_masking.csv',sep=str('$'))

    datasets = list(set(df['dataset_ID']))
    random.shuffle(datasets)
    test_datasets = ['5','9','12','17','31','32','34','37','45'] #random.sample(datasets,int(len(datasets)*0.2))
    train_datasets = list(set(datasets)-set(test_datasets))
    print(len(test_datasets))
    print(len(train_datasets))
    print(len(datasets))
    print("******************")

    all_pos = df[df.label == 1 and df['dataset_ID'].isin(test_datasets) ]
    all_neg = df.drop(all_pos.index)

    #divide positive samples
    test = all_pos[all_pos['dataset_ID'].isin(test_datasets)]
    train = all_pos.drop(test.index)#df[df['dataset'] in train_datasets]

    #divide negative samples
    neg_test = all_neg.sample(frac=0.2)
    neg_train = all_neg.drop(neg_test.index)

    train = train.append(neg_train)
    test = test.append(neg_test)

    print(len(test))
    print(len(train))


    test.to_csv(data_path+'masked_2\\test.csv',index=None,sep=str('$'))
    train.to_csv(data_path+'masked_2\\train.csv',index=None,sep=str('$'))


def special_controlled_divide_processed_data():

    df = pd.read_csv(data_path+'masked_2\\all_samples_wiout_masking.csv',sep=str('$'))

    datasets = list(set(df['dataset_ID']))
    random.shuffle(datasets)
    test_datasets = ['5','9','12','17','31','32','34','37','45']
    train_datasets = list(set(datasets)-set(test_datasets))
    print(len(test_datasets))
    print(len(train_datasets))
    print(len(datasets))
    print("******************")

    test = df[df['dataset_ID'].isin(test_datasets)]
    rest_samples = df.drop(test.index)

    #divide negative samples
    for_test = rest_samples.sample(frac=0.19)
    train = rest_samples.drop(for_test.index)

    test = test.append(for_test)

    print(len(test))
    print(len(train))


    test.to_csv(data_path+'masked_2\\test.csv',index=None,sep=str('$'))
    train.to_csv(data_path+'masked_2\\train.csv',index=None,sep=str('$'))

def mask_taining_data():
    df = pd.read_csv(data_path + 'masked_2\\train.csv', sep=str('$'))

    data = []

    for i,row in df.iterrows():
        if row['dataset'] != 'None':
            dataset_mention_strings = list(idp.get_dataset_group_by_string(row['dataset']))
            if len(dataset_mention_strings) >1:
                dataset_mention_strings.sort(key=len,reverse=True)

            for x in dataset_mention_strings:
                new_string = row['sample_text'].replace(x,"DS_T")

            item = (row[0],row[1],row[2],row[3],row[4],new_string,row[6])
        else:
            item = row

        data.append(item)


    masked_df = pd.DataFrame(data,
                          columns=['file_id', 'publication_name', 'section_title', 'dataset', 'label', 'sample_text',
                                   "dataset_ID"])

    masked_df.to_csv(data_path+"masked_2\\masked_trian.csv",index=None,sep=str('$'))

    #if len(after_mention) > 0 and after_mention[0][0] == '(':
        #after_mention = after_mention[1:]

    # sample_text = " ".join(before_mention[-num_words:]) + " DS_T " + " ".join(after_mention[:num_words])

def further_preprocessing():
    df = pd.read_csv(data_path+"/exp_data/all_samples_manual_ids.csv",sep=str('$'))
    dataset_id_dict=idp.read_dataset_ids_file()

    data= []

    for i, row in df.iterrows():
        dataset_list = []
        index_list = []

        context = row["context"]
        ds_id = -1
        if row["dataset"] != 'None':
            ds_id = idp.get_dataset_id(row["dataset"])
        dataset_list.append(row["dataset"])
        index_list.append(row["index"])


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
                    dataset_list.append(ds_mention)
                    index_list.append(ds_index)
                    break

        #remove the 'None' dataset for the cased discoveed in the second path through the identifiers.
        if len(index_list) > 1 and index_list[0] == -1:
            dataset_list = dataset_list[1:]
            index_list = index_list[1:]
            row[6]=1

        if len(dataset_list) > 1:
            dataset_list,index_list=gu.order_mentions_by_index(dataset_list,index_list)

        context_without_ds_mentions = context
        for ds in dataset_list:
            if ds == 'None':
                continue
            context_without_ds_mentions=context_without_ds_mentions.replace(ds,"")

        data_item = (row[0],row[1],row[2],context,context_without_ds_mentions,dataset_list,index_list,row[6])
        data.append(data_item)

    output= pd.DataFrame(data,columns=["file_id","publication_name","section_title","context","NO_DS_context","dataset","index","label"])

    output.to_csv(data_path+"/final_version/NO_DS_further_processed_all_samples_manual_ids.csv",index=None,sep=str('$'))

#this function searches for further mataches in the datase and add them as new samples
def further_preprocessing_crossvalidation():

    df = pd.read_csv(data_path+"/original_data/all_samples_manual_ids.csv",sep=str('$'))
    dataset_id_dict=idp.read_dataset_ids_file()

    data= []

    for i, row in df.iterrows():
        dataset_list = []
        index_list = []

        context = row["context"]
        ds_id = -1
        if row["dataset"] != 'None':
            ds_id = idp.get_dataset_id(row["dataset"])
        #dataset_list.append(row["dataset"])
        #index_list.append(row["index"])

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
                    dataset_list.append(ds_mention)
                    index_list.append(ds_index)
                    break

        #remove the 'None' dataset for the cased discoveed in the second path through the identifiers.
        if len(index_list) > 1 and index_list[0] == -1:
            dataset_list = dataset_list[1:]
            index_list = index_list[1:]
            row[6]=1


        context_without_ds_mentions = context
        for ds in dataset_list:
            if ds == 'None':
                continue
            context_without_ds_mentions=context_without_ds_mentions.replace(ds,"")

        data_item = (row[0],row[1],row[2],context,context_without_ds_mentions,dataset_list,index_list,row[6])
        data.append(data_item)

    output= pd.DataFrame(data,columns=["file_id","publication_name","section_title","context","dataset","index","label",'dataset_id'])

    output.to_csv(data_path+"/crossValidationData/further_processed_all_samples_manual_ids.csv",index=None,sep=str('$'))


def sort_by_indexes(lst, indexes, reverse=False):
  return [val for (_, val) in sorted(zip(indexes, lst), key=lambda x: \
          x[0], reverse=reverse)]

def order_mentions_by_index(a_datasets,b_indexes):

    ds = np.array(b_indexes)
    ordered_indices = np.argsort(ds)

    a_datasets = sort_by_indexes(a_datasets,ordered_indices)
    b_indexes = sort_by_indexes(b_indexes,ordered_indices)

    return a_datasets,b_indexes


#this function splits the dataset into 5 equally-sized subsets with same class distribution like the original datasets
def split_the_dataset():
    # Load the dataset into a pandas dataframe.
    dataset = pd.read_csv(data_path + "exp_data/NO_DS_further_processed_all_samples_manual_ids.csv", delimiter='$')

    test_sizes = [0.2,0.25,0.33,0.5]
    i = 0
    for x in test_sizes:
        trainX, testX, trainy, testy = train_test_split(dataset,dataset["label"],test_size=x,random_state=2,stratify=dataset["label"])
        test_data = testX
        dataset = trainX

        test_data.to_csv(data_path+f"exp_data/subset-{i}.csv",index=None,sep=str('$'))
        i+=1
        if x == 0.5:
            dataset.to_csv(data_path+f"exp_data/subset-{i}.csv",index=None,sep=str('$'))

        # summarize
        train_0, train_1 = len(trainy[trainy == 0]), len(trainy[trainy == 1])
        test_0, test_1 = len(testy[testy == 0]), len(testy[testy == 1])
        print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))


#this function extract words from contexts and count their frequencies
def collect_salient_words():

    df = pd.read_csv(data_path+"exp_data/all_samples.csv",delimiter="$")
    #df = df[df["label"]==0]

    word_freq_pos = {}
    word_freq_neg = {}
    for i,row in df.iterrows():
        context = row["context"]
        context = context.translate(str.maketrans('', '', string.punctuation))
        context = context.lower()
        context = remove_stopwords(context)
        context = context.split()
        for word in context:
            if row["label"]==1:
                if len(word)<3 or word.isnumeric() or len(word)>12:
                    continue

                if word in word_freq_pos:
                    word_freq_pos[word]+=1
                else:
                    word_freq_pos[word]=1
            else:
                if len(word) < 3 or word.isnumeric():
                    continue

                if word in word_freq_neg:
                    word_freq_neg[word] += 1
                else:
                    word_freq_neg[word] = 1

    sorted_dict_pos = dict(sorted(word_freq_pos.items(), key=lambda item: item[1],reverse=True))
    sorted_dict_neg = dict(sorted(word_freq_neg.items(), key=lambda item: item[1],reverse=True))

    '''
    print("print common words")
    common = set(sorted_dict_pos.keys()).intersection(set(sorted_dict_neg.keys()))
    print(common)

    print("\n\nwords uniquess to positive contexts\n\n")
    print(set(sorted_dict_pos.keys()).difference(set(sorted_dict_neg.keys())))

    print("\n\nwords uniquess to negative contexts\n\n")
    print(set(sorted_dict_neg.keys()).difference(set(sorted_dict_pos.keys())))

    print("\n\npositive contexts statistis\n\n")
    '''
    counter = 0
    for k,v in sorted_dict_pos.items():
        counter += 1

        print(f"{k}:{v}\n")
        if counter == 1000:
            break

    print(counter)
    print(len(sorted_dict_pos))

    print("\n\nnegative contexts statistics\n\n")
    counter = 0
    for k, v in sorted_dict_neg.items():
        counter += 1

        print(f"{k}:{v}\n")
        if counter == 1000:
            break

    print(counter)
    print(len(sorted_dict_neg))


#mask_taining_data()
#special_controlled_divide_processed_data()
process_kaggle_data_with_id()
#get_stats(data_path+"final_version\\all_samples.csv")
#compute_support_ds_id()

#further_preprocessing()
#further_preprocessing_crossvalidation()
#split_the_dataset()

#collect_salient_words()



