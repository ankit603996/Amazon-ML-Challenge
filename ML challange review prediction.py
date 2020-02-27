############################# Brief Summary ############################
1.  Data Collection and Data Understanding
2.  Exploratory Data Analysis
3.  Feature Engineering
4.  Model Development and Parameter Tuning through Cross validation
5.  Model Performance Summary
6.  Final Prediction on test data file

########################################################################

############################# Library and Packages #####################
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd
import numpy as np
import scipy
import os
import pandas as pd
import numpy as np
import re
import nltk
import datetime
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import random
import spacy
import gensim 
from gensim.models import Word2Vec 
from nltk.tokenize import sent_tokenize, word_tokenize 
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",15)
pd.set_option("display.max_rows",300)
from nltk import FreqDist
import seaborn as sns

os.chdir('F:\LocalDriveD\Analytics\Interview\Amazon\Practice Test\ML challange Practice\ML chalange review prediction\Dataset')
#dataset = pd.read_csv('train.csv')

train = pd.read_csv('train.csv') # 5959,3
test = pd.read_csv('test.csv') # 2553,2
##### Data Snapshot ####### 
train.shape
test.shape
train.head(10)
train.info()

#---- columns are categorical variables
#----- columns are numerical variables


####################################### EDA of train data ###########################
# make entire text lowercase
train['Review Text'] = [r.lower() for r in train['Review Text']]
# Punctuation and numeric remove CONFIRM and TEST whether we need[^ws] or [^\w\s].
train['Review Text'] = [re.sub(r'[^\w\s]|[0-9]|[__]', "", d, flags=re.I) for d in train['Review Text']] # to include nueric user'[^ws]'
# Strip white space
train['Review Text'] = [re.sub(r"s+"," ", w, flags = re.I) for w in train['Review Text']]
# # function to remove stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new
train['Review Text'] = [remove_stopwords(r.split()) for r in train['Review Text']]
#lemmatization 
#!python -m spacy download en # one time run
nlp = spacy.load('en', disable=['parser', 'ner'])
def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output
tokenized_reviews = pd.Series(train['Review Text']).apply(lambda x: x.split())
reviews_2 = lemmatization(tokenized_reviews)
print(reviews_2[1]) # print lemmatized review
reviews_3 = []
for i in range(len(reviews_2)):
    reviews_3.append(' '.join(reviews_2[i]))
train['Review Text'] = reviews_3 #
# function to plot most frequent terms after removing stop words
def freq_words(x, terms = 30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    fdist = FreqDist(all_words)
    words_train = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
    words_train = words_train.sort_values(by='count',ascending=False)
# selecting top 20 most frequent words
    d = words_train.nlargest(columns="count", n = terms) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()
freq_words(train['Review Text'])
# Generate a word cloud image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text = " ".join(review for review in train['Review Text'])
wordcloud = WordCloud( background_color="white").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



####################################### Feature Engineering ###########################
# We will take description column of train and test data together for feature engineering
# to make sure similar feature availability in model developement and final prediction on test data

test['topic'] = 'dummy'
train['id'] = 'train'
test['id'] = 'test'
#train_data = train[['id','Review Text','Review Title']]
#test_data = test[['id','Review Text','Review Title']]
#dataset = pd.concat([train_data,test_data],axis=0)
dataset = pd.concat([train,test],axis=0)
dataset['Review Text'] = dataset['Review Text'].astype(str) ### You can club Review Title also
main= dataset.copy()
############ Feature 1: POS
from collections import Counter
from itertools import chain
from nltk import word_tokenize, pos_tag
# Punctuation and numeric remove
dataset['Review Text'] = [re.sub(r'[^\w\s]|[0-9]|[__]', "", d, flags=re.I) for d in dataset['Review Text']] # to include nueric user'[^ws]'
# Strip white space
dataset['Review Text'] = [re.sub(r"\s+"," ", w, flags = re.I) for w in dataset['Review Text']]
## If any of test recond shows str.len()==0 that means POS will throw an error SO ignore POS
'''
a = (dataset['Review Text'].str.len()==0) & (dataset['id']=='test')
a.sum()  # if this is one then leave POS
'''
## ELSE Go Ahead
dataset = dataset[dataset['Review Text'].str.len()!=0]
dataset.index = np.array(range(dataset.shape[0]))
tok_and_tag = lambda x: pos_tag(word_tokenize(x))
df = pd.DataFrame()
df['lower_sent'] = dataset['Review Text'].apply(str.lower)
df['tagged_sent'] = df['lower_sent'].apply(tok_and_tag)
possible_tags = sorted(set(list(zip(*chain(*df['tagged_sent'])))[1]))
def add_pos_with_zero_counts(counter, keys_to_add):
    for k in keys_to_add:
        counter[k] = counter.get(k, 0)
    return counter

# Detailed steps.
df['pos_counts'] = df['tagged_sent'].apply(lambda x: Counter(list(zip(*x))[1]))
df['pos_counts_with_zero'] = df['pos_counts'].apply(lambda x: add_pos_with_zero_counts(x, possible_tags))
df['sent_vector'] = df['pos_counts_with_zero'].apply(lambda x: [count for tag, count in sorted(x.most_common())])
df['sent_vector'] = df['tagged_sent'].apply(lambda x:
        [count for tag, count in sorted(
                add_pos_with_zero_counts(
                        Counter(list(zip(*x))[1]), 
                                possible_tags).most_common()
                    )
                        ]
                 )
POS = pd.DataFrame(df['sent_vector'].tolist())
POS.columns = possible_tags
############ Feature 2: total words/character in sentence
dataset['totalChar'] = dataset['Review Text'].str.len()
dataset['totalwords'] = dataset['Review Text'].str.split().str.len()

############ Feature 3: DTM Tf-idf   
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer(object): # this lemmatization function will be used as an argument in countvectorizer
  def __init__(self):
      self.wnl = WordNetLemmatizer()
  def __call__(self, articles):
      return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
vectorizer = TfidfVectorizer(lowercase=True,ngram_range=(1,3), max_features=4000,
stop_words='english', tokenizer=LemmaTokenizer()).fit(dataset['Review Text'])
TFIDF = vectorizer.fit_transform(dataset['Review Text']).toarray()
TFIDF = pd.DataFrame(TFIDF)
TFIDF.columns = vectorizer.get_feature_names()


############ Feature 4: NER ( For 8512 it took )
'''
from datetime import datetime,date,time
print(datetime.now())
from nltk import word_tokenize, ne_chunk, pos_tag
from nltk import tree2conlltags
def NERFUNCTION(text):
    NER = pd.DataFrame(columns=["B-PERSON","I-PERSON","B-ORGANIZATION","I-ORGANIZATION","B-LOCATION","I-LOCATION","B-GPE","I-GPE"])
    ne_tree = ne_chunk(pos_tag(word_tokenize(text)))
    iob_tagged = tree2conlltags(ne_tree)
    Df = pd.DataFrame(iob_tagged)
    Df.columns = ['words', 'POS', 'ENTITY']
    Df = Df[['words', 'ENTITY']]
    group = Df.groupby(['ENTITY'])
    NERForSent1 = (group.agg(np.count_nonzero))
    a = NER.columns[~(NER.columns.isin(NERForSent1.T.columns))]
    NER1 = NER[a]
    NER1.iloc[:,:] = np.repeat(0,len(a))
    NER1 = pd.DataFrame(np.repeat(0,len(a)),NER.columns[~(NER.columns.isin(NERForSent1.T.columns))])
    NER1 = NER1.T
    NERForSent1 = NERForSent1.T
    NERForSent1.index = NER1.index
    NER1 = pd.concat([NER1,NERForSent1],axis=1)
    #del NER1['O']
    NER1 = NER1[["B-PERSON","I-PERSON","B-ORGANIZATION","I-ORGANIZATION","B-LOCATION","I-LOCATION","B-GPE","I-GPE"]]
    return NER1.iloc[0,:]
a1 = dataset['Review Text'].apply(lambda x: NERFUNCTION(x))
NER = pd.DataFrame(columns=["PERSON","ORGANIZATION","LOCATION"])
NER['PERSON'] = a1[['B-PERSON',"I-PERSON"]].sum(axis=1)
NER['ORGANIZATION'] = a1[['B-ORGANIZATION',"I-ORGANIZATION"]].sum(axis=1)
NER['LOCATION'] = a1[['B-LOCATION',"I-LOCATION",'B-GPE',"I-GPE"]].sum(axis=1)
print(datetime.now())
'''

############ Feature 5: polarity score ()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
Polarity = []
corpus  = dataset.loc[:,'Review Text']
for sentence in corpus:
    ss = sid.polarity_scores(sentence)
    a = [ ]
    for k in sorted(ss):
        a.append(ss[k])
    Polarity.append(a)
Polarity = pd.DataFrame(Polarity)
Polarity.columns = ['compound','negative','neutral','postive']
    
########### Training and Test split
dataset.index = np.array(range(dataset.shape[0]))
TFIDF.index = np.array(range(dataset.shape[0]))
Polarity.index = np.array(range(dataset.shape[0]))

Newdata = pd.concat([dataset[['totalChar','totalwords']],TFIDF,Polarity],axis=1) #POS,NER
Newdata.index = np.array(range(Newdata.shape[0]))
Newdata = Newdata.loc[:,~Newdata.columns.duplicated()]

## Take Newdata_NUM seperate with all numerical variables: Newdata_NUM = Newdata.drop(['',''],axis=1)
## Take Newdata_CAT seperate with all numerical variables: Newdata_CAT = Newdata[['','']]
## Take dummy of cat = dummydata = pd.get_dummies(Newdata_CAT)
## merge both ie. Newdata = pc.concat([Newdata_NUM,Newdata_CAT],axis=1)

Newdata['id']  = dataset['id']
Newdata['topic']  = dataset['topic']
Newdata['Review Text'] = dataset['Review Text']
Newdata['Review Title']= dataset['Review Title']

x = Newdata[Newdata['id']=='train']
y = Newdata['topic'][Newdata['id']=='train']
y = pd.DataFrame(y)
del x['id']
del x['Review Text'];del x['Review Title']
a = random.sample(range(0, x.shape[0]), int(x.shape[0]*0.75))
X_train = x.iloc[a,] 
X_test = x.drop(x.index[a])
del X_train['topic']
del X_test['topic']
y_train = y.iloc[a,] 
y_test = y.drop(y.index[a])

#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
##labelencoder_y_1 = LabelEncoder()
#LE = labelencoder_y_1.fit(y_train)
#y_train = LE.transform(y_train)
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
scoring = ['precision_macro','recall_macro','accuracy']
from sklearn.tree import DecisionTreeClassifier
depth1 = pd.DataFrame(columns = ['test_accuracy','test_precision_macro','test_recall_macro','Max_depth'])
max_depth = [10,25,50]
for i in max_depth:
    clf = DecisionTreeClassifier(max_depth = i,random_state=0)
    scores = cross_validate(estimator = clf,X=X_train, y=y_train,cv=3,scoring=scoring)
    a = pd.DataFrame(scores)
    a = a[['test_accuracy','test_precision_macro','test_recall_macro']]
    a['Max_depth']=i
    depth1 = pd.concat([a,depth1],axis=0)
depth1['Max_depth'][depth1['test_accuracy'] ==depth1['test_accuracy'].max()] #50

from sklearn.ensemble import RandomForestClassifier
depth2 = pd.DataFrame(columns = ['test_accuracy','test_precision_macro','test_recall_macro',
                                 'Max_depth','n_estimators'])
max_depth = [10,50]
for j in range(1,2):
    for i in max_depth:
        clf = RandomForestClassifier(max_depth = i,n_estimators = j*10,random_state=0)
        scores = cross_validate(estimator = clf,X=X_train, y=y_train,cv=3,scoring=scoring)
        a = pd.DataFrame(scores)
        a = a[['test_accuracy','test_precision_macro','test_recall_macro']]
        a['Max_depth']=i
        a['n_estimators']=i
        depth2 = pd.concat([a,depth2],axis=0)
depth2[['Max_depth','n_estimators','test_accuracy']][depth2['test_accuracy'] ==depth2['test_accuracy'].max()] #50

from xgboost import XGBClassifier
depth3 = pd.DataFrame(columns = ['test_accuracy','test_precision_macro','test_recall_macro',
                                 'Max_depth','n_estimators','learning_rate'])
max_depth = [10]
for k in range(1,2):
    for j in range(1,2):
        for i in max_depth:
            clf = XGBClassifier(learning_rate =k/10, max_depth = i,n_estimators = j*100,random_state=0)
            scores = cross_validate(estimator = clf,X=X_train, y=y_train,cv=2,scoring=scoring)
            a = pd.DataFrame(scores)
            a = a[['test_accuracy','test_precision_macro','test_recall_macro']]
            a['Max_depth']=i
            a['n_estimators']=i
            a['learning_rate']=i
            depth3 = pd.concat([a,depth3],axis=0)
depth3[['Max_depth','n_estimators','learning_rate','test_accuracy']][depth3['test_accuracy'] ==depth3['test_accuracy'].max()] #50

# Validation of model results from best obtained model ( frm c.v.) on test data
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,roc_curve,auc
from sklearn.preprocessing import label_binarize
#from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt


dt = DecisionTreeClassifier(max_depth = 50,random_state=0)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

rf = RandomForestClassifier(max_depth = 50,n_estimators =50,random_state=0)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

xgb = RandomForestClassifier(max_depth = 50,n_estimators =50,random_state=0)
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

########## ROC Curve on best model
y_score = dt.predict_proba(X_test)
y_test_binary = label_binarize(y_test['topic'],classes = ['Not Effective', 'Bad Taste/Flavor', 'Quality/Contaminated',
       'Customer Service', 'Packaging', 'Pricing', 'Too big to swallow',
       'Allergic', 'False Advertisement', 'Color and texture', 'Texture',
       'Shipment and delivery', 'Ingredients', 'Smells Bad', 'Expiry',
       'Too Sweet', 'Wrong Product received', "Didn't Like",
       'Inferior to competitors', 'Hard to Chew', 'Customer Issues'])
n_classes = 21
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i],tpr[i],_= roc_curve(y_test_binary[:,i].ravel(),y_score[:,i].ravel())
    roc_auc[i] = auc(fpr[i],tpr[i])
# compute micro average ROC curve
fpr['micro'],tpr['micro'], _ = roc_curve(y_test_binary.ravel(),y_score.ravel())
roc_auc['micro'] = auc(fpr['micro'],tpr['micro'])
# aggregate all positive rates
lw =2
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# then concatenate all ROC_curve at this point
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr,fpr[i],tpr[i])
#finally average it and compute AUC
mean_tpr /= n_classes    
fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'],tpr['macro'])
# Plot All ROC curve
plt.figure()
plt.plot(fpr['micro'],tpr['micro'],
         Label = 'macro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc['micro']),
                   color = 'deeppink',Linestyle = ':',Linewidth = 4)
         
plt.plot(fpr['macro'],tpr['macro'],
         Label = 'macro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc['macro']),
                   color = 'navy',Linestyle = ':',Linewidth = 4)

color = cycle(['aqua','darkorange','cornflowerblue'])
for i,color in zip(range(n_classes),color):
    plt.plot(fpr[i],tpr[i],color= color,lw= lw,
             Label = 'ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i,roc_auc[i]))
plt.plot([0,1],[0,1],'k--',lw=lw)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc = 'Lower right')
plt.show()

#cohen kappa's
metrics.cohen_kappa_score(y_test.iloc[:,0],y_pred)


##################Final Prediction 
new = Newdata[Newdata['id']=='test']
new = new.drop(['id', 'topic'],axis=1)
new_corecolumns = pd.DataFrame(new[['Review Text', 'Review Title']])
new_corecolumns.index = np.array(range(new_corecolumns.shape[0]))

x_new = new.copy()
del x_new['Review Text'];del x_new['Review Title']    
New_pred = pd.DataFrame(rf.predict(x_new))
New_pred.columns = ['topic']
New_pred.index = np.array(range(New_pred.shape[0]))

Submit = pd.concat([new_corecolumns,New_pred],axis=1)
    