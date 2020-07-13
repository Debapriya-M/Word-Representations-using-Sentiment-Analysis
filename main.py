import json
import numpy as np
from sklearn.model_selection import train_test_split
from pos_function import Embed
from pos_function import model_build
from word_embeddings import load_data,prepare_data_for_word_vectors,building_word_vector_model,classification_model,padding_input,prepare_data_for_word_vectors_imdb
import warnings
warnings.filterwarnings('ignore')

# Modules for data manipulation
import numpy as np
import pandas as pd
import re

# Modules for visualization
import matplotlib.pyplot as plt

# Tools for preprocessing input data
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tools for creating ngrams and vectorizing input data
from gensim.models import Word2Vec, Phrases

# Tools for building a model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

# Tools for assessing the quality of model prediction
from sklearn.metrics import accuracy_score, confusion_matrix
import os


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIG_SIZE = 16
LARGE_SIZE = 20

params = {
    'figure.figsize': (16, 8),
    'font.size': SMALL_SIZE,
    'xtick.labelsize': MEDIUM_SIZE,
    'ytick.labelsize': MEDIUM_SIZE,
    'legend.fontsize': BIG_SIZE,
    'figure.titlesize': LARGE_SIZE,
    'axes.titlesize': MEDIUM_SIZE,
    'axes.labelsize': BIG_SIZE
}
plt.rcParams.update(params)


usecols = ['sentiment','review']
train_data = pd.read_csv(
    filepath_or_buffer='data/labeledTrainData.tsv',
    usecols=usecols, sep='\t')
unlabeled_data = pd.read_csv(
    filepath_or_buffer="data/unlabeledTrainData.tsv", 
    error_bad_lines=False,
    sep='\t')
submission_data = pd.read_csv(
    filepath_or_buffer="data/testData.tsv",
    sep='\t')

datasets = [train_data, submission_data, unlabeled_data]
titles = ['Train data', 'Unlabeled train data', 'Submission data']
for dataset, title in zip(datasets,titles):
    print(title)
    dataset.info()
    #display(dataset.head())


all_reviews = np.array([], dtype=str)
for dataset in datasets:
    all_reviews = np.concatenate((all_reviews, dataset.review), axis=0)
print('Total number of reviews:', len(all_reviews))


def clean_review(raw_review: str) -> str:
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "lxml").get_text()
    # 2. Remove non-letters
    letters_only = REPLACE_WITH_SPACE.sub(" ", review_text)
    # 3. Convert to lower case
    lowercase_letters = letters_only.lower()
    return lowercase_letters


def lemmatize(tokens: list) -> list:
    # 1. Lemmatize
    tokens = list(map(lemmatizer.lemmatize, tokens))
    lemmatized_tokens = list(map(lambda x: lemmatizer.lemmatize(x, "v"), tokens))
    # 2. Remove stop words
    meaningful_words = list(filter(lambda x: not x in stop_words, lemmatized_tokens))
    return meaningful_words


def preprocess(review: str, total: int, show_progress: bool = True) -> list:
    if show_progress:
        global counter
        counter += 1
        print('Processing... %6i/%6i'% (counter, total), end='\r')
    # 1. Clean text
    review = clean_review(review)
    # 2. Split into individual words
    tokens = word_tokenize(review)
    # 3. Lemmatize
    lemmas = lemmatize(tokens)
    # 4. Join the words back into one string separated by space,
    # and return the result.
    return lemmas

counter = 0
REPLACE_WITH_SPACE = re.compile(r'[^A-Za-z\s]')
stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

all_reviews = np.array(list(map(lambda x: preprocess(x, len(all_reviews)), all_reviews)))
counter = 0

X_train_data = all_reviews[:train_data.shape[0]]
Y_train_data = train_data.sentiment.values
X_submission = all_reviews[125000: 150000]

train_data['review_lenght'] = np.array(list(map(len, X_train_data)))
median = train_data['review_lenght'].median()
mean = train_data['review_lenght'].mean()
mode = train_data['review_lenght'].mode()[0]

fig, ax = plt.subplots()

ax.set_xlim(left=0, right=np.percentile(train_data['review_lenght'], 95))
ax.set_xlabel('Words in review')
ymax = 0.014
plt.ylim(0, ymax)
ax.plot([mode, mode], [0, ymax], '--', label=f'mode = {mode:.2f}', linewidth=4)
ax.plot([mean, mean], [0, ymax], '--', label=f'mean = {mean:.2f}', linewidth=4)
ax.plot([median, median], [0, ymax], '--',
        label=f'median = {median:.2f}', linewidth=4)
ax.set_title('Words per review distribution', fontsize=20)
plt.legend()
plt.show()

bigrams = Phrases(sentences=all_reviews)
trigrams = Phrases(sentences=bigrams[all_reviews])
print(bigrams['space station near the solar system'.split()])

embedding_vector_size = 256
trigrams_model = Word2Vec(
    sentences = trigrams[bigrams[all_reviews]],
    size = embedding_vector_size,
    min_count=3, window=5, workers=4)

print("Vocabulary size:", len(trigrams_model.wv.vocab))
trigrams_model.wv.most_similar('galaxy')


trigrams_model.wv.doesnt_match(['galaxy', 'starship', 'planet', 'dog'])

def vectorize_data(data, vocab: dict) -> list:
    print('Vectorize sentences...', end='\r')
    keys = list(vocab.keys())
    filter_unknown = lambda word: vocab.get(word, None) is not None
    encode = lambda review: list(map(keys.index, filter(filter_unknown, review)))
    vectorized = list(map(encode, data))
    print('Vectorize sentences... (done)')
    return vectorized

# fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

# summarize history for accuracy
# axis1.plot(history.history['acc'], label='Train', linewidth=3)
# axis1.plot(history.history['val_acc'], label='Validation', linewidth=3)
# axis1.set_title('Model accuracy', fontsize=16)
# axis1.set_ylabel('accuracy')
# axis1.set_xlabel('epoch')
# axis1.legend(loc='upper left')

# summarize history for loss
# axis2.plot(history.history['loss'], label='Train', linewidth=3)
# axis2.plot(history.history['val_loss'], label='Validation', linewidth=3)
# axis2.set_title('Model loss', fontsize=16)
# axis2.set_ylabel('loss')
# axis2.set_xlabel('epoch')
# axis2.legend(loc='upper right')
# plt.show()


print('Convert sentences to sentences with ngrams...', end='\r')
X_data = trigrams[bigrams[X_train_data]]
print('Convert sentences to sentences with ngrams... (done)')
input_length = 150
X_pad = pad_sequences(
    sequences=vectorize_data(X_data, vocab=trigrams_model.wv.vocab),
    maxlen=input_length,
    padding='post')
print('Transform sentences to sequences... (done)')

X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    Y_train_data,
    test_size=0.05,
    shuffle=True,
    random_state=42)

def build_model(embedding_matrix: np.ndarray, input_length: int):
    model = Sequential()
    model.add(Embedding(
        input_dim = embedding_matrix.shape[0],
        output_dim = embedding_matrix.shape[1], 
        input_length = input_length,
        weights = [embedding_matrix],
        trainable=False))
    model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1)))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model

model = build_model(
    embedding_matrix=trigrams_model.wv.vectors,
    input_length=input_length)

model.compile(
    loss="binary_crossentropy",
    optimizer='adam',
    metrics=['accuracy'])

history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    epochs=20)





y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)


print('Convert sentences to sentences with ngrams...', end='\r')
X_submit = trigrams[bigrams[X_submission]]
print('Convert sentences to sentences with ngrams... (done)')
X_sub = pad_sequences(
    sequences=vectorize_data(X_submit, vocab=trigrams_model.wv.vocab),
    maxlen=input_length,
    padding='post')
print('Transform sentences to sequences... (done)')


Y_sub_pred = model.predict_classes(X_sub)


def json_to_dict(json_set):
    for k,v in json_set.items():
        if v == "True":
            json_set[k]= True
        elif v == "False":
            json_set[k]=False
        else:
            json_set[k]=v
    return json_set

with open("config.json","r",encoding = "ISO-8859-1") as f:
    params_set = json.load(f)
params_set = json_to_dict(params_set)


with open("model_params.json", "r",encoding = "ISO-8859-1") as f:
    model_params = json.load(f)
model_params = json_to_dict(model_params)

'''
    load_data function works on imdb data. In order to load your data, comment line 27 and pass your data in the form of X,y
    X = text data column
    y = label column(0,1 etc)

'''
# for imdb data

if params_set["use_imdb"]==1:
    print("loading imdb data")
    x_train,x_test,y_train,y_test = load_data(params_set["vocab_size"],params_set["max_len"])
    X = np.concatenate([x_train,x_test])
    y = np.concatenate([y_train,y_test])
    sentences_as_words,word_ix = prepare_data_for_word_vectors_imdb(X)
    print(sentences_as_words[0])
    model_wv = building_word_vector_model(params_set["option"],sentences_as_words,params_set["embed_dim"],
                                       params_set["workers"],params_set["window"],y)
    print("word vector model built")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=params_set["split_ratio"], random_state=42)

    x_train_pad,x_test_pad = padding_input(x_train,x_test,params_set["max_len"])
    print("padded")
else:
# for other data:
# put your data in the form of X,y

    X = ["this is a sentence","this is another sentence by me","yet another sentence for training","one more again"]
    y=np.array([0,1,1,0])

    sentences_as_words,sentences,word_ix = prepare_data_for_word_vectors(X)
    print(sentences_as_words[0])
    print("sentences loaded")
    model_wv = building_word_vector_model(params_set["option"],sentences,params_set["embed_dim"],
                                       params_set["workers"],params_set["window"],y)
    print("word vectors model built")
    x_train, x_test, y_train, y_test = train_test_split(sentences, y, test_size=params_set["split_ratio"], random_state=42)
    print("splitting done")
    x_train_pad,x_test_pad = padding_input(x_train,x_test,params_set["max_len"])
    print("padded")


if params_set["use_imdb"]==1:
    print("")
    embed = Embed(params_set["vocab_size"],params_set["embed_dim"],params_set["pos_embed_dim"],params_set["max_len"],True)
    print("embed class")
    inp_seq,sent_emb = embed.embed_sentences(word_ix,model_wv,False,x_train_pad)
    print("sentence embedding done")
    pos_enc = embed.tag_pos1(sentences_as_words)
    print("POS encoded")
    x_train_pos, x_test_pos, _, _ = train_test_split(pos_enc, y, test_size=params_set["split_ratio"], random_state=42)
    x_train_pos_pad,x_test_pos_pad = padding_input(x_train_pos,x_test_pos,params_set["max_len"])
    print("POS padded")
    inp_pos,pos_embed = embed.embed_pos(x_train_pos_pad)
    print("building model")
    #model = model_build(inp_seq,inp_pos,sent_emb,pos_embed,x_train_pad,x_train_pos_pad,y_train,model_params["epochs"],model_params["batch_size"],x_test_pad,x_test_pos_pad,y_test)
    model = build_model(
    embedding_matrix=trigrams_model.wv.vectors,
    input_length=input_length)
    
    print("model built")
    print(model.summary())

else :

    embed = Embed(params_set["vocab_size"],params_set["embed_dim"],params_set["pos_embed_dim"],params_set["max_len"],True)
    inp_seq,sent_emb = embed.embed_sentences(word_ix,model_wv,False,x_train_pad)

    pos_enc = embed.tag_pos1(sentences_as_words)
    print("POS encoded")
    x_train_pos, x_test_pos, _, _ = train_test_split(pos_enc, y, test_size=params_set["split_ratio"], random_state=42)
    x_train_pos_pad,x_test_pos_pad = padding_input(x_train_pos,x_test_pos,params_set["max_len"])
    print("POS padded")
    inp_pos,pos_embed = embed.embed_pos(x_train_pos_pad)
    print("building model")
    model = model_build(inp_seq,inp_pos,sent_emb,pos_embed,x_train_pad,x_train_pos_pad,y_train,model_params["epochs"],model_params["batch_size"],x_test_pad,x_test_pos_pad,y_test)
    print("model built")
    print(model.summary())
