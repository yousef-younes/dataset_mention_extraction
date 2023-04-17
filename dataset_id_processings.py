import pandas as pd

data_path = 'data/'
dataset_ids_file = data_path+"exp_data/datasets_final.txt"  #this is the source of the ids
#create and save dataset ids file
def create_dataset_ids_file():

    #read training data
    df = pd.read_csv(data_path + "train.csv")
    print(len(df))

    #remove duplicated titels-label pairs
    df = df.drop_duplicates(subset=["dataset_title", "dataset_label"])
    print(len(df))

    df = df[["dataset_title", "dataset_label"]]

    labels = set(df["dataset_label"])
    titles = set(df["dataset_title"])

    title_dict = {}
    for i, row in df.iterrows():
        key = row["dataset_title"]
        if key not in title_dict:
            title_dict[key] = set()
        title_dict[key].add(row["dataset_label"])
    print(len(title_dict))


    dataset_dict = {}
    count = 1

    for k, v in title_dict.items():
        dataset_dict[count] = set()

        dataset_dict[count] = v
        dataset_dict[count].add(k)

        count += 1

    print(len(dataset_dict))

    #get parentheses-enclosed acronyms
    for k, v in dataset_dict.items():
        to_be_removed = set()
        to_be_added = set()
        for item in v:
            res = extact_parenthesis(item)
            if res != item:
                # to_be_removed.add(item)
                to_be_added |= set(res)
            else:
                print(res)
        # apply modifications
        # remove items
        # for ddi in to_be_removed:
        # dataset_dict[k].remove(ddi)

        # add items
        for ddi in to_be_added:
            dataset_dict[k].add(ddi)

    with open("datasets_final.txt", "w") as f:
        for k, v in dataset_dict.items():
            dataset_representations = " $ ".join(v)
            f.write("{}:{}\n".format(k, dataset_representations))


#this function receives a string and extract the content of its paranthesis
def extact_parenthesis(x):
    print(x+"************")
    x = x.strip()
    index_1 = x.find("(")
    if index_1 != -1:
        index_2 = x.find(")")
        if index_1 != -1 and index_2 == len(x)-1:
            return [x[:index_1].strip(),x[index_1+1:index_2].strip()]
            print(x)
        print("{0}:{1}".format(index_1,index_2))
    print(x)
    return x

#This function reads the dataset id file into a dictionary
def read_dataset_ids_file():
    dic = {}
    with open(dataset_ids_file,"r") as f:
        for line in f.readlines():
            line = line.split(":")
            ds_strings = line[1]
            ds_strings = ds_strings.split("$")
            dic[line[0].strip()] = set([x.strip() for x in ds_strings])

    return dic

'''This function takes a dataset mention string and return its dataset id'''
def get_dataset_id(dataset_str,id_dict=None):
    if id_dict == None:
        id_dict = read_dataset_ids_file()
    for k,v in id_dict.items():
        if dataset_str in v:
            return k
    return None

''' This function takes dataset mention string, find its ID, then returns the set to which it belongs'''
def get_dataset_group_by_string(dataset_str,id_dict=None):
    if id_dict == None:
        id_dict = read_dataset_ids_file()
    ds_id = get_dataset_id(dataset_str)
    if ds_id != None:
        return id_dict[ds_id]

    return None

'''
This function receives a dataset id and its mentions strings and returns a list of dataset ids that contains (partially or fully) elements of the input set 
'''
def get_list_dataset_ids(id,mention_strings):
    dict = read_dataset_ids_file()
    output = []
    for k,v in dict.items():
        if k == id: #skip the id to wich the mention strings belongs
            continue
        for element in mention_strings:
            for element2 in v:
                if element in element2:
                    output.append((k,element,element2))

    return  output

#This functions try to find containment cases across dataset mention strings
def find_containent_cases():
    dict = read_dataset_ids_file()

    for k,v in dict.items():
        ds_ids = get_list_dataset_ids(k,v)
        print(f"{k}:{ds_ids}")

