import sys
import gensim
import sklearn
import numpy as np
import jieba
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from jieba import analyse
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
import json

f = open('data/test.json', 'r')

data = json.load(f)

tagged_data = [TaggedDocument(words=list(jieba.lcut(_d)), tags=[
                              str(i)]) for i, _d in enumerate(tqdm(data))]

print("Finished the Tagged Data!")

max_epochs = 100
vec_size = 100
alpha = 0.025

model = Doc2Vec(tagged_data,
                vector_size=vec_size,
                window = 5,
                alpha=alpha,
                min_alpha=0.001,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

# for i, epoch in enumerate(tqdm(range(max_epochs))):
#     # decrease the learning rate
#     model.alpha -= 0.0002
#     # fix the learning rate, no decay
#     model.min_alpha = model.alpha
#     if (i % 5 == 0):
#         model.save("model/d2v.model")

model.train(tagged_data,
                total_examples=model.corpus_count,
                start_alpha=0.002,
                end_alpha=-0.016,
                epochs=model.epochs)

model.save("model/d2v.model")
print("Model Saved")