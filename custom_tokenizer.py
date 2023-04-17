from datasets import  load_dataset
from transformers import AutoTokenizer
import general_util as gu

'''
This file contains code to train tokenizers on the postive contexts of the data that we have. 
'''

data_path = "data/exp_data/"
logger = gu.get_my_logger("retrain_tok")\

batch_size = 1000

# load data
dataset = load_dataset('json', data_files={data_path + '*.json'},split='train')
# filter out all contexts that does not contain dataset mentions
dataset = dataset.filter(lambda example: example['label'] == 1)
def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["context"]


def train_tokenizer(tokenizer_ckpt,output_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
    #print(tokenizer.is_fast)
    new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(),vocab_size=25000)
    new_tokenizer.save_pretrained(output_dir)
    print("tokenizer has been retrained")

def test_trained_tokenizer(origin_tokenizer_,custom_tokenizer_):
    trained_tokenizer = AutoTokenizer.from_pretrained(custom_tokenizer_)
    original_tokenizer = AutoTokenizer.from_pretrained(origin_tokenizer_)

    context = r"""In the BLSA PiB-PET study, T1-weighted volumetric magnetic resonance imaging scans were co-registered to the mean of the first 
        20-min dynamic PET images with the mutual information method in the Statistical Parametric Mapping software (SPM 2; Wellcome Department of Imaging Neuroscience,"""

    print(f"Original {origin_tokenizer_} Tokenization: ")
    print(original_tokenizer.tokenize(context))

    print(f"\nCustomized {custom_tokenizer_} Tokenization:")
    print(trained_tokenizer.tokenize(context))

    print('\n************************************************\n')

    context= "The statistics on FFRDC R&D presented in this report come from the FY 2012 NSF FFRDC Research and Development Survey . This annual survey is completed by FFRDC staff and administrators and collects data from FFRDCs on R&D expenditures by"

    print(f"Original {origin_tokenizer_} Tokenization: ")
    print(original_tokenizer.tokenize(context))

    print(f"\nCustomized {custom_tokenizer_} Tokenization:")
    print(trained_tokenizer.tokenize(context))

    print("\n*************************************************\n")

    context = r"""In this study, we showed that expression values of AD-related genes obtained from blood samples of ADNI , ANM1 and ANM2 could classify AD and CN. Additionally,
       we observed that AD-related genes from blood samples were enriched
      """

    print(f"Original {origin_tokenizer_} Tokenization: ")
    print(original_tokenizer.tokenize(context))

    print(f"\nCustomized {custom_tokenizer_} Tokenization:")
    print(trained_tokenizer.tokenize(context))




model_checkpoint =  "roberta-base"
#model_checkpoint = "microsoft/deberta-base"

output_dir = "tokenizers/my_roberta_tokenizer"
train_tokenizer(model_checkpoint,output_dir)

#test_trained_tokenizer(model_checkpoint,output_dir)






