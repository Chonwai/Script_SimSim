from gensim.models.doc2vec import Doc2Vec
import jieba
from nltk.tokenize import word_tokenize
import nltk
import json
import argparse
import re
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--sentence', default='一款好的鋼筆是不是可以讓你的鋼筆更出彩呢，這款鋼筆的筆尖是不是很舒服很順暢，不會卡頭。',
                    type=str, required=False, help='Please enter a sentence or paragraph.')
parser.add_argument('--data_path', default='../data/train_content.json',
                    type=str, required=False, help='Please enter a path of data.')

args = parser.parse_args()

f = open(args.data_path, 'r', encoding='utf-8')
data = json.load(f)

stop_words = pd.read_csv("../data/stopwords.txt", index_col=False, quoting=3,
                         names=['stopword'],
                         sep="\n",
                         encoding='utf-8')
stop_words = list(stop_words.stopword)

model = Doc2Vec.load("../model/d2v.model")
# to find the vector of a document which is not in training data

sentence = args.sentence

sentence = re.sub('[0-9]', '', sentence)
sentence = re.sub('[a-zA-Z]', '', sentence)
sentence = re.sub("[，、\:：“”‘’'|!！（）丨。？/*-+]", '', sentence)
sentence = re.sub('\s', '', sentence)
word_list = list(jieba.lcut(sentence))
segments = []
for word in word_list:
    if word not in stop_words:
        segments.append(word)

# to find most similar doc using tags
inferred_vector_dm = model.infer_vector(segments)
similar_doc = model.dv.most_similar([inferred_vector_dm], topn=5)
print('To find most similar doc using tags:')
print(similar_doc)

print("Test Sentence: " + args.sentence + '\n')

for item in similar_doc:
    print('Sentence: ' + data[int(item[0])] +
          ', Distance: ' + str(item[1]) + '\n')
