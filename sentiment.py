
import utils
import importlib
importlib.reload(utils)
from utils import *

import re, string, collections, bcolz, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import confusion_matrix


df = pd.read_csv('data/reviews.csv', lineterminator='\n')

df.shape



df.head()



df = df[['text', 'score']]

df.head()



print(f'reviews with score greater than 85: {len(df.loc[df["score"] > 85])}')
print(f'reviews with score less than 50: {len(df.loc[df["score"] < 50])}')

pos = df.loc[df['score'] > 85, 'text'].copy().reset_index(drop=True)
neg = df.loc[df['score'] < 50, 'text'].copy().reset_index(drop=True)

len(pos), len(neg)



for i in range(4):
    print(''.join(pos[np.random.randint(0, len(pos))]))
    print('\n')



for i in range(4):
    print(''.join(neg[np.random.randint(0, len(neg))]))
    print('\n')



neg = pd.concat([pd.DataFrame(neg), pd.DataFrame(np.zeros(neg.shape), columns=['class'])], 1)
pos = pd.concat([pd.DataFrame(pos), pd.DataFrame(np.ones(pos.shape), columns=['class'])], 1)



lens = neg['text'].str.len()
lens.mean(), lens.std(), lens.max()

lens.hist(figsize=(12, 6), bins=25);



long_reviews = neg.loc[neg['text'].str.len() > 5000].index
neg.drop(long_reviews, inplace=True)



lens = pos['text'].str.len()
lens.mean(), lens.std(), lens.max()

lens.hist(figsize=(12, 6), bins=25);

long_reviews = pos.loc[pos['text'].str.len() > 5000].index
pos.drop(long_reviews, inplace=True)



np.random.seed(42)
rand = np.random.permutation(pos.shape[0])
pos = pos.iloc[rand[:neg.shape[0]]].reset_index(drop=True)

neg.shape, pos.shape


df = pd.concat([pos, neg]).sample(frac=1).reset_index(drop=True)
df.head()

df.shape



X_train, X_test, y_train, y_test = train_test_split(df['text'].values, df['class'].values, test_size=0.2, random_state=42)

len(X_train), len(X_test), len(y_train), len(y_test)



re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()



vect = CountVectorizer(tokenizer=tokenize)

tf_train = vect.fit_transform(X_train)
tf_test = vect.transform(X_test)



tf_train



tf_train[0]



vocab = vect.get_feature_names()
len(vocab)

vocab[4000: 4005]

X_train[0]



w0 = set([o for o in X_train[0].split(' ')])

w0

vect.vocabulary_['unless']

tf_train[0, 41989]



svd = TruncatedSVD()
reduced_tf_train = svd.fit_transform(tf_train)

plot_embeddings(reduced_tf_train, y_train)



p = tf_train[y_train==1].sum(0) + 1
q = tf_train[y_train==0].sum(0) + 1
r = np.log((p/p.sum())/(q/q.sum()))
b = np.log(len(p)/len(q))

pre_preds = tf_test @ r.T + b
preds = pre_preds.T > 0
acc = (preds==y_test).mean()
print(f'Accuracy: {acc}')



model = LogisticRegression(C=0.2, dual=True)
model.fit(tf_train, y_train)
preds = model.predict(tf_test)
acc = (preds==y_test).mean()
print(f'Accuracy: {acc}')

plot_confusion_matrix(confusion_matrix(y_test, preds.T), classes=['Negative', 'Positive'], title='Confusion matrix')

"""Let's plot most relevant words that the algorithm uses to classify a text in positive or negative"""

coef_df = pd.DataFrame({'vocab': vocab, 'coef':model.coef_.reshape(-1)})
pos_top10 = coef_df.sort_values('coef', ascending=False).reset_index(drop=True)[:10]
neg_top10 = coef_df.sort_values('coef').reset_index(drop=True)[:10]

fig, axs = plt.subplots(1, 2, figsize=(8, 8))
fig.subplots_adjust(wspace=0.8)
pos_top10.sort_values('coef').plot.barh(legend=False, ax=axs[0])
axs[0].set_yticklabels(pos_top10['vocab'].values.tolist()[::-1])
axs[0].set_title('Positive');
neg_top10.sort_values('coef', ascending=False).plot.barh(legend=False, ax=axs[1])
axs[1].set_yticklabels(neg_top10['vocab'].values.tolist()[::-1])
axs[1].set_title('Negative');



vect.vocabulary_['ea']

tf_train = lil2(tf_train)
tf_train.removecol(12893)

tf_test = lil2(tf_test)
tf_test.removecol(12893)

model = LogisticRegression(C=0.2, dual=True)
model.fit(tf_train, y_train)
preds = model.predict(tf_test)
acc = (preds==y_test).mean()
print(f'Accuracy: {acc}')



vect = TfidfVectorizer(strip_accents='unicode', tokenizer=tokenize, ngram_range=(1, 2), max_df=0.9, min_df=3, sublinear_tf=True)

tfidf_train = vect.fit_transform(X_train)
tfidf_test = vect.transform(X_test)



svd = TruncatedSVD()
reduced_tfidf_train = svd.fit_transform(tfidf_train)

plot_embeddings(reduced_tfidf_train, y_train, 2000)

p = tfidf_train[y_train==1].sum(0) + 1
q = tfidf_train[y_train==0].sum(0) + 1
r = np.log((p/p.sum())/(q/q.sum()))
b = np.log(len(p)/len(q))

model = LogisticRegression(C=30, dual=True)
model.fit(tfidf_train, y_train)
preds = model.predict(tfidf_test)
acc = (preds==y_test).mean()
print(f'Accuracy: {acc}')

plot_confusion_matrix(confusion_matrix(y_test, preds.T), classes=['Negative', 'Positive'], title='Confusion matrix')



from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Flatten, Dense, Dropout, Convolution1D, MaxPooling1D, SpatialDropout1D, Input, concatenate
from keras.optimizers import Adam



df['tokenized'] = df['text'].apply(tokenize)



def update_vocab_counter(row):
    for word in row:
        vocab_counter[word] += 1

vocab_counter = collections.Counter()
df['tokenized'].apply(update_vocab_counter);
vocab = sorted(vocab_counter, key=vocab_counter.get, reverse=True)

len(vocab)



w2id = {w:i for i, w in enumerate(vocab)}



def transform_to_ids(row):
    return [w2id[w] for w in row]

df['tokenized'] = df['tokenized'].apply(lambda x: transform_to_ids(x))

X_train, X_test, y_train, y_test = train_test_split(df['tokenized'].values, df['class'].values, test_size=0.2, random_state=42)

maxlen = 1000



x_train = pad_sequences(X_train, maxlen=maxlen, value=-1)
x_test = pad_sequences(X_test, maxlen=maxlen, value=-1)

model = Sequential([Embedding(input_dim=len(vocab), output_dim=32, input_length=maxlen),
                    SpatialDropout1D(0.2),
                    Dropout(0.25),
                    Convolution1D(64, 5, padding='same', activation='relu'),
                    Dropout(0.25),
                    MaxPooling1D(),
                    Flatten(),
                    Dense(100, activation='relu'),
                    Dropout(0.85),
                    Dense(1, activation='sigmoid')])

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=4, batch_size=64)



