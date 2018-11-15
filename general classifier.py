import sys
import jsonpickle
import os
import tweepy
import csv
import json
import re
import numpy as np
from string import punctuation
import sklearn
from preprocess_tweets import tweet_clean

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import TransformerMixin, BaseEstimator




FLAGS = re.MULTILINE | re.DOTALL


def preprocess(text):


    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "")
    text = re_sub(r"/","")
    text = re_sub(r"@\w+", "")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "")
    text = re_sub(r"{}{}p+".format(eyes, nose), "")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "")
    text = re_sub(r"<3","")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "")
    text = " ".join(word.strip() for word in re.split('#|_', text))
    text = re_sub(r"([!?.]){2,}", r"")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"")
    

    text=text.lower()
    text=tweet_clean(text)
    return text

def create_ngram_feat(docs_text, min_ngram=1,max_ngram=2):
    feat_vect = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(min_ngram,max_ngram))
    feat_vect.fit(train_set)
    ngram = feat_vect.transform(train_set)
    return feat_vect , ngram


# cntr will return you the count of the number of pos in that particular tweet. So to make the feature vector
# we can simply use this to fill in the columns and the rest as 0

def pos_count(tweet):
    pos=nltk.pos_tag(tweet) 
    pos_tags = []
    for i,j in pos:
        pos_tags.append(j)
    cntr=dict(collections.Counter(pos_tags))
    total = sum(cntr.values())
    cntr = {k:cntr[k]/total for k in cntr}
    return cntr

def k_fold_validation(train_set_features,labels,results,k=5,valid_size=0.2):
    #results will be the path to the file to store results of validation rounds
    sss_1 = StratifiedShuffleSplit(n_splits=k, test_size=valid_size, random_state=0)
    count = 1
    for [train_index, test_index] in sss.split(train_set_features, labels):
        print("Split No : ", count)
        trainData, testData = train_set_features[train_index], train_set_features_features[test_index]
        trainLabels, testLabels = labels[train_index], labels[test_index]
        print("[INFO] splited train and test data...")
        print("[INFO] train data  : {}".format(trainData.shape))
        print("[INFO] test data   : {}".format(testData.shape))
        print("[INFO] train labels: {}".format(trainLabels.shape))
        print("[INFO] test labels : {}".format(testLabels.shape))
        print("[INFO] creating model...")
        
        # Write ML Model here
        # model = #Fill in 
        model.fit(trainData, trainLabels)
        print("[INFO] evaluating model...")
        f = open(results, "a+")
        rank_1 = 0
        for (label, features) in zip(testLabels, testData):

            predictions = model.predict_proba(np.atleast_2d(features))[0]
            predictions = np.argsort(predictions)[::-1][:5]

        # rank-1 prediction increment
            if label == predictions[0]:
                rank_1 += 1

    # convert accuracies to percentages
        rank_1 = (rank_1 / float(len(testLabels))) * 100

    # write the accuracies to file
        f.write("\nSplit No :{} ".format(count))
        f.write("\nRank-1: {:.2f}%\n".format(rank_1))

    # evaluate the model of test data
        preds = model.predict(testData)

    # write the classification report to file
        f.write("{}\n".format(classification_report(testLabels, preds)))
        f.close()
        count = count + 1
        
def train_model(model,train_features,labels):
    if model =="logreg":
        model = LogisticRegression(random_state=seed)
    elif model=="svm":
    	model="svm"
        
    elif model=="nb":
        model = GaussianNB()
    elif model=="rf":
        model = RandomForestClassifier(n_estimators=500, max_depth=15, max_features="log2", random_state=seed)
        
    model.fit(train_features,labels)
    return model

def test_model(model,test_features,labels):
    prediction = model.predict(test_features)
    print("{}\n".format(classification_report(labels,prediction)))
    

fileName='C:/ml/blood_donation/EBRT/BDC_dataset.txt'
with open(fileName,'r') as f:
	dataset=json.load(f)
	
# print(dataset[0])
dataset_text=[]
dataset_contextual_feature=[]
dataset_user_metadata=[]
dataset_textual_metadata=[]
dataset_label=[]
for i in range(len(dataset)):
	temp=dataset[i]['text']
	temp=preprocess(temp)
	dataset_text.append(temp)
	temp2=[]
	if dataset[i]['hospital']=='NULL' or dataset[i]['hospital']=='0':
		temp2.append(0)
	else:
		temp2.append(1)
	if dataset[i]['blood group']=='NULL' or dataset[i]['blood group']=='0':
		temp2.append(0)
	else:
		temp2.append(1)
	try:
		if dataset[i]['contact number']=='NULL' or dataset[i]['contact number']=='0':
			temp2.append(0)
		else:
			temp2.append(1)
				
	except:
		if dataset[i]['Contact Number']=='NULL' or dataset[i]['Contact Number']=='0':
			temp2.append(0)
		else:
			temp2.append(1)

	
	if dataset[i]['blood quantity']=='NULL' or dataset[i]['blood quantity']=='0':
		temp2.append(0)
	else:
		temp2.append(1)
	if dataset[i]['patient disease']=='NULL' or dataset[i]['patient disease']=='0':
		temp2.append(0)
	else:
		temp2.append(1)
	if dataset[i]['contact name']=='NULL' or dataset[i]['contact name']=='0':
		temp2.append(0)
	else:
		temp2.append(1)
	if dataset[i]['place of donation']=='NULL' or dataset[i]['place of donation']=='0':
		temp2.append(0)
	else:
		temp2.append(1)
	
	dataset_contextual_feature.append(temp2)

	temp3=[]
	if dataset[i]['retweet_count']=='NULL' or dataset[i]['retweet_count']=='0':
		temp3.append(0)
	else:
		temp3.append(int(dataset[i]['retweet_count']))
	if dataset[i]['source_of_posting']=='NULL' or dataset[i]['source_of_posting']=='0':
		temp3.append(0)
	else:
		temp3.append(1)
	if dataset[i]['place_of_posting']=='NULL' or dataset[i]['place_of_posting']=='0':
		temp3.append(0)
	else:
		temp3.append(1)
	if dataset[i]['user_friends_count']=='NULL' or dataset[i]['user_friends_count']=='0':
		temp3.append(0)
	else:
		temp3.append(int(dataset[i]['user_friends_count']))
	if dataset[i]['user_followers_count']=='NULL' or dataset[i]['user_followers_count']=='0':
		temp3.append(0)
	else:
		temp3.append(int(dataset[i]['user_followers_count']))
	if dataset[i]['statuses_count']=='NULL' or dataset[i]['statuses_count']=='0':
		temp3.append(0)
	else:
		temp3.append(int(dataset[i]['statuses_count']))
	if dataset[i]['user_favourites_count']=='NULL' or dataset[i]['user_favourites_count']=='0':
		temp3.append(0)
	else:
		temp3.append(int(dataset[i]['user_favourites_count']))

	dataset_user_metadata.append(temp3)

	temp4=[]

	if dataset[i]['url_count']=='NULL' or dataset[i]['url_count']=='0':
		temp4.append(0)
	else:
		temp4.append(int(dataset[i]['url_count']))
	if dataset[i]['hashtag_count']=='NULL' or dataset[i]['hashtag_count']=='0':
		temp4.append(0)
	else:
		temp4.append(int(dataset[i]['hashtag_count']))
	if dataset[i]['user_mention_count']=='NULL' or dataset[i]['user_mention_count']=='0':
		temp4.append(0)
	else:
		temp4.append(int(dataset[i]['user_mention_count']))
	if dataset[i]['special_symbols_count']=='NULL' or dataset[i]['special_symbols_count']=='0':
		temp4.append(0)
	else:
		temp4.append(int(dataset[i]['special_symbols_count']))

	dataset_textual_metadata.append(temp4)
	
	if dataset[i]['blood required']=='NULL' or dataset[i]['blood required']=='0':
		dataset_label.append(0)
	else:
		dataset_label.append(1)

print(len(dataset_label))		
print(len(dataset_text))
print(len(dataset_textual_metadata))
print(len(dataset_user_metadata))
print(len(dataset_contextual_feature))
# print(dataset_text)
			
dataset_text, dataset_textual_metadata, dataset_user_metadata, dataset_contextual_feature, dataset_label = sklearn.utils.shuffle(dataset_text, dataset_textual_metadata, dataset_user_metadata, dataset_contextual_feature, dataset_label)



X_text_train=dataset_text[2700:]
X_textual_metadata_train=dataset_textual_metadata[2700:]
X_user_metadata_train=dataset_user_metadata[2700:]
X_contextual_feature_train=dataset_contextual_feature[2700:]
Y_label_train=dataset_label[2700:]

X_text_test=dataset_text[:2700]
X_textual_metadata_test=dataset_textual_metadata[:2700]
X_user_metadata_test=dataset_user_metadata[:2700]
X_contextual_feature_test=dataset_contextual_feature[:2700]
Y_label_test=dataset_label[:2700]


# 
# text_clf = Pipeline([
#     ('features', FeatureUnion([
#         ('text', Pipeline([
#             ('vectorizer', CountVectorizer(min_df=1,max_df=2,ngram_range=(1,2))),
#             ('tfidf', TfidfTransformer()),
#         ])),
#         ('other', Pipeline([
#             ('meta', FunctionTransformer(get_meta, validate=False)),
#         ]))

text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None)),
])




text_clf.fit(X_text_train, Y_label_train)
predicted = text_clf.predict(X_text_test)
print(np.mean(predicted == Y_label_test))

clf_SVC= LinearSVC()
clf_SVC.fit(X_contextual_feature_train,Y_label_train)
predicted=clf_SVC.predict(X_contextual_feature_test)
print(np.mean(predicted == Y_label_test))


clf_SVC.fit(X_user_metadata_train,Y_label_train)
predicted=clf_SVC.predict(X_user_metadata_test)
print(np.mean(predicted == Y_label_test))

clf_SVC.fit(X_textual_metadata_train,Y_label_train)
predicted=clf_SVC.predict(X_textual_metadata_test)
print(np.mean(predicted == Y_label_test))

# print(X_contextual_feature_train)
# q=input("wait")
# print(X_textual_metadata_train)
# q=input("wait")
# print(X_user_metadata_train)
# q=input("wait")
# print(X_textual_metadata_train)






X_textual_user_train=[]
X_textual_user_test=[]

for i in range(len(X_textual_metadata_train)):
	X_textual_user_train.append(X_textual_metadata_train[i]+X_user_metadata_train[i])
for i in range(len(X_textual_metadata_test)):	
	X_textual_user_test.append(X_textual_metadata_test[i]+X_user_metadata_test[i])

clf_SVC.fit(X_textual_user_train,Y_label_train)
predicted=clf_SVC.predict(X_textual_user_test)
print(np.mean(predicted == Y_label_test))


X_textual_contextual_train=[]
X_textual_contextual_test=[]

for i in range(len(X_textual_metadata_train)):
	X_textual_contextual_train.append(X_textual_metadata_train[i]+X_contextual_feature_train[i])
for i in range(len(X_textual_metadata_test)):
	X_textual_contextual_test.append(X_textual_metadata_test[i]+X_contextual_feature_test[i])

clf_SVC.fit(X_textual_contextual_train,Y_label_train)
predicted=clf_SVC.predict(X_textual_contextual_test)
print(np.mean(predicted == Y_label_test))


X_user_contextual_train=[]
X_user_contextual_test=[]

for i in range(len(X_contextual_feature_train)):
	X_user_contextual_train.append(X_user_metadata_train[i]+X_contextual_feature_train[i])
for i in range(len(X_contextual_feature_test)):
	X_user_contextual_test.append(X_user_metadata_test[i]+X_contextual_feature_test[i])

clf_SVC.fit(X_user_contextual_train,Y_label_train)
predicted=clf_SVC.predict(X_user_contextual_test)
print(np.mean(predicted == Y_label_test))



X_user_contextual_textual_train=[]
X_user_contextual_textual_test=[]

for i in range(len(X_contextual_feature_train)):
	X_user_contextual_textual_train.append(X_user_metadata_train[i]+X_contextual_feature_train[i]+X_textual_metadata_train[i])
for i in range(len(X_contextual_feature_test)):
	X_user_contextual_textual_test.append(X_user_metadata_test[i]+X_contextual_feature_test[i]+X_textual_metadata_test[i])

clf_SVC.fit(X_user_contextual_textual_train,Y_label_train)
predicted=clf_SVC.predict(X_user_contextual_textual_test)
print(np.mean(predicted == Y_label_test))


