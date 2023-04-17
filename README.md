# dataset_mention_extraction
This repo contains code for extracting datset mentions from scientific text.


Prepare Data:
1. Download kaggle dataset from  https://www.kaggle.com/competitions/coleridgeinitiative-show-us-the-data/data and save it in the data folder. You should have train folder and train.csv file directly under data folder.
2. From data_wrangling module use the "process_kaggle_data_with_id" function to extract the contexts and generate "all_samples.csv" file. 
This module contains also a function "" that counts the frequent words in positive and negative contexts. The frequent words are used to create a list of five questions.
3. Use create_folds module to create the five folds that will be used in the experiments.

Classification Experiments:
4. To get the classification result for a language model. run module cls_exp.py. It will generate a log file that contains the reuslts for each fold using the language model whose check point is given. It also stores the best performing model for each module.
5. BERT_MLP2_BFL.py uses MLP2 as classification head for Bert. 
6. mlp_exp.py module can be used to generate MLP-2 on Bert-mean or TF-IDF results. 
7. To use custome tokenization, cusotme_tokenizer.py module ca be used. It generates the trained tokenizer that you can used to replace or to modify the orignal tokenizer for a language model.
8. xgb_exp.py module can be used to classify contexts using XGBoosting on bert-mean and tfidf.
9. ensemble_meta_model_exp.py stacks an ensemble of three MLP-2 detectors each with different settings. First run mlp_exp module using the different settings to create the three models then use ensemble_meta_model_exp to combine them.

NER Experiments:
10. Moduel ner_exp.py is used to generate ner results using langaue models.
11. Module ner_space.py is used to generate ner results using spacy. For that you need to have the configs folder which is provided in the code and need to download spacy using the instruction
python -m spacy download en_core_web_md
then you can use the code. 

QA Experiments:
12. Moudle qa_exp.py can be used to generate the question answering experiments. 

Notebooks:
13. svm_cls_exp.ipynb: this notebook uses PCA, TSNe as input to SVM to do the detection task.
14. MLP2-analsis.ipynb: this notebook applies MLP-2 on the folds and store the results as a dictionay in a text file. It also used to analyze the detector resutls on fold 0.
15. pipe_and_qa_analysis.ipynb: this notebook is used to pipe the detector and extractor. It is also used to analyze the pipe performance when deberta with q3 is used on fold 0. 


