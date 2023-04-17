

import xgboost as xgb
from sklearn.metrics import classification_report
import general_util as gu

def bert_xgb():
        
    for fold in range(5):
        x_train,y_train = gu.get_train_xgbt_data(fold)

        # get the embedding
        x_train = gu.get_bert_embedding(x_train)


        # Fitting a simple xgboost on CountVec
        clf = xgb.XGBClassifier(max_depth=200, n_estimators=400, subsample=1, learning_rate=0.07, reg_lambda=0.1, reg_alpha=0.1,\
                           gamma=1)
        clf.fit(x_train, y_train)


        #get validation data
        x_val, y_val = gu.get_val_data(fold)

        # get the embedding
        x_val = gu.get_bert_embedding(x_val)

        #predict the labels for the validation set
        predictions = clf.predict(x_val)

        print(classification_report(y_val,predictions))


#tf_idf
def tf_xgb():
    for fold in range(5):
        x_train, y_train = gu.get_train_xgbt_data(fold)

        tfidf_engine = gu.get_tfidf_engine(x_train)

        # get tfidf embedding of train data
        train_data_embed = tfidf_engine.transform(x_train)

        # Fitting a simple xgboost on CountVec
        clf = xgb.XGBClassifier(max_depth=200, n_estimators=400, subsample=1, learning_rate=0.07, reg_lambda=0.1,
                                reg_alpha=0.1, \
                                gamma=1)
        clf.fit(train_data_embed, y_train)

        # get validation data
        x_val, y_val = gu.get_val_data(fold)

        # get tfidf embedding
        x_val = tfidf_engine.transform(x_val)

        # predict the labels for the validation set
        predictions = clf.predict(x_val)

        print(classification_report(y_val,predictions))


bert_xgb()
