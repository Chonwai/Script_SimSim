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

f = open('data/train.json', 'r', encoding='utf-8')

data = json.load(f)

tagged_data = [TaggedDocument(words=list(jieba.cut(_d)), tags=[
                              str(i)]) for i, _d in enumerate(tqdm(data))]

print("Finished the Tagged Data!")

max_epochs = 100
vec_size = 200
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for i, epoch in enumerate(tqdm(range(max_epochs))):
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
    if (i % 5 == 0):
        model.save("model/d2v.model")


model.save("model/d2v.model")
print("Model Saved")