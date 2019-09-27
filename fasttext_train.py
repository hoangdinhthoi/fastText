# -*- coding: utf-8 -*-
import os
import pandas as pd
import string
import pickle
from tqdm import tqdm
from pyvi import ViTokenizer
from gensim.models.fasttext import FastText

# path data
pathdata = './vn_news'

def normalize_text(article):
    listpunctuation = string.punctuation
    for i in listpunctuation:
        article = article.replace(i, ' ')
    return article


def tokenize(article):
    text_token = ViTokenizer.tokenize(article)
    return text_token

def read_data(path=pathdata):
    traindata = []
    listfile = os.listdir(path)
    for namefile in tqdm(listfile):
        with open(path + '/' + namefile) as f:
            datafile = f.read()
        article = tokenize((normalize_text(datafile))).split()
        traindata.append(article)
    return traindata

if __name__ == '__main__':
    # size_list = [50, 100, 150, 200]
    size_list = [40]
    if os.path.exists("train_data.pkl"):
        with open("train_data.pkl", "rb") as f_r:
            train_data = pickle.load(f_r)
    else:
        train_data = read_data()
        with open("train_data.pkl", "wb") as f_w:
            pickle.dump(train_data, f_w)
    print("Read data complete!!!")
    for size in size_list:
        model_fasttext = FastText(size=size, window=10, min_count=2, workers=4, sg=1)
        print("begin build vocabulary!!!")
        model_fasttext.build_vocab(train_data)
        print("FastText training...")
        model_fasttext.train(train_data, total_examples=model_fasttext.corpus_count, epochs=model_fasttext.iter)
        print("Save model...")
        model_fasttext.wv.save("./trained_model/fasttext_gensim_" + str(size) + ".model")
