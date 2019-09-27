from gensim.models import KeyedVectors
from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from tqdm import tqdm

size_list = [50, 100, 150, 200]
path_visim = './validate_data/Visim-400.txt'
data = pd.read_csv(path_visim, sep="\t")
list_words = list(data["Word1"]) + list(data["Word2"])

def cosin_similarity(word1, word2, word_vector_reduced):
    vec_word1 = word_vector_reduced[word1]
    vec_word2 = word_vector_reduced[word2]
    return (2 - cosine(vec_word1, vec_word2))*5

for size in tqdm(size_list):
    model = KeyedVectors.load(
        "./trained_model/fasttext_gensim_" + str(size) + ".model")
    words_np = []
    words_label = []
    for word in list_words:
        words_np.append(model[word])
        words_label.append(word)
    word_vector_reduced = {}
    for index, vec in enumerate(words_np):
        word_vector_reduced[words_label[index]] = vec
    list_cosin_similarity = []
    for x, y in zip(data["Word1"], data["Word2"]):
        list_cosin_similarity.append(round(cosin_similarity(x, y, word_vector_reduced), 2))
    data["FastText_" + str(size)] = list_cosin_similarity
    if size == 200:
        for new_size in size_list[:-1]:
            svd = TruncatedSVD(n_components=new_size, n_iter=30)
            svd.fit(words_np)
            reduced = svd.transform(words_np)
            word_vector_reduced = {}
            for index, vec in enumerate(reduced):
                word_vector_reduced[words_label[index]] = vec
            list_cosin_similarity = []
            for x, y in zip(data["Word1"], data["Word2"]):
                list_cosin_similarity.append(round(cosin_similarity(x, y, word_vector_reduced), 2))
            data["FastText_SVD_" + str(new_size)] = list_cosin_similarity
# Ghi ket qua ra file csv
data.to_csv("./result/Visim_result.csv", sep="\t")
