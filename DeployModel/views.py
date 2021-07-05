from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
import pandas as pd
#import matplotlib.pylot as plt
import matplotlib as pllt
from matplotlib import pyplot as plt
import seaborn as sns
import nltk 
import re
from wordcloud import WordCloud

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score




# Create your views here.
def home(request):
    return render(request, "home.html")

def result(request):

    fake = pd.read_csv('Fake.csv')
    text = ' '.join(fake['text'].tolist())
    ' '.join(['this','is','a','data'])
    real = pd.read_csv('True.csv')
    text = ' '.join(real['text'].tolist())
    unknown_publishers = []
    for index, row in enumerate(real.text.values):
        try:
            record = row.split('-', maxsplit=1)
            record[1]          
            assert(len(record[0])<120)
        except:
            unknown_publishers.append(index)
    real.iloc[unknown_publishers].text
    real.iloc[8970]
    real = real.drop(8970, axis=0)
    publisher = []
    tmp_text = []

    for index, row in enumerate(real.text.values):
        if index in unknown_publishers:
            tmp_text.append(row)
            publisher.append('Unknown')
            
        else:
            record = row.split('-', maxsplit=1)
            publisher.append(record[0].strip())
            tmp_text.append(record[1].strip())

    real['publisher']=publisher
    real['text'] = tmp_text
    empty_fake_index = [index for index,text in enumerate(fake.text.tolist()) if str(text).strip()==""]

    fake.iloc[empty_fake_index]

    real['text'] = real['title'] + " " + real['text']
    fake['text'] = fake['title'] + " " + fake['text']

    real['text'] = real['text'].apply(lambda x: str(x).lower())
    fake['text'] = fake['text'].apply(lambda x: str(x).lower())

    real['class'] = 1
    fake['class'] = 0

    real.columns

    real = real[['text', 'class']]

    fake = fake[['text', 'class']]

    data = real.append(fake, ignore_index=True)

    import gensim

    y = data['class'].values

    X = [d.split() for d in data['text'].tolist()]


    DIM = 100
    w2v_model = gensim.models.Word2Vec(sentences=X, window=10, min_count=1)

    nos = np.array([len(x) for x in X])

    maxlen = 1000

    tokenizer = Tokenizer()

    vocab_size = len(tokenizer.word_index) + 1
    vocab = tokenizer.word_index

    def get_weight_matrix(model):
        weight_matrix = np.zeros((vocab_size, DIM))
        
        for word, i in vocab.items():
            weight_matrix[i] = model.wv[word]
            
        return weight_matrix

    embedding_vectors = get_weight_matrix(w2v_model)

    model = Sequential()
    model.add(Embedding(vocab_size, output_dim=DIM, weights = [embedding_vectors], input_length=maxlen, trainable=False))
    model.add(LSTM(units=128))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    model.fit(X_train, y_train, validation_split=0.3, epochs=1)

    x = [(request.GET['Object'])]
    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=maxlen)
    ans = (model.predict(x) >=0.5).astype(int)

    import joblib

    filename = 'final777.sav'
    joblib.dump(joyp, filename)

    import pickle

    filename = 'finalized_model123.sav'
    pickle.dump(y_pred, open(filename, 'wb'))

    joblib.dump(joy7, 'filename.pkl')



    

    
        

        # name_of_object = (request.GET['Object'])

        # if search(a, name_of_object):
        #     ans = "is found"
        # else:
        #     ans = "is not found"




    # LA = joblib.load('finalized_model.sav')

    # s1 = pd.Series([(request.GET['Tenure']),(request.GET['Dependents']),(request.GET['MultipleLines']),(request.GET['InternetService']),(request.GET['PhoneService']),(request.GET['PaymentMethod']),(request.GET['TotalCharges']),(request.GET['Contract']),(request.GET['StreamingTV']),(request.GET['OnlineBackup'])])
    # cols =  ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    # df = pd.DataFrame([list(s1)],  columns =  cols)

    # lis = []

    # lis.append(request.GET['Tenure'])
    # lis.append(request.GET['Dependents'])
    # lis.append(request.GET['MultipleLines'])
    # lis.append(request.GET['InternetService'])
    # lis.append(request.GET['PhoneService'])
    # lis.append(request.GET['PaymentMethod'])
    # lis.append(request.GET['TotalCharges'])
    # lis.append(request.GET['Contract'])
    # lis.append(request.GET['StreamingTV'])
    # lis.append(request.GET['OnlineBackup'])

    ans = ans

    return render(request,"result.html",{'ans':ans})
