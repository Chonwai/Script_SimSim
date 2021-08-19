from gensim.models.doc2vec import Doc2Vec
import jieba
from nltk.tokenize import word_tokenize
import nltk
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sentence', default='一款好的鋼筆是不是可以讓你的鋼筆更出彩呢，這款鋼筆的筆尖是不是很舒服很順暢，不會卡頭。', type=str, required=False, help='Please enter a sentence or paragraph.')
parser.add_argument('--data_path', default='data/train_content.json', type=str, required=False, help='Please enter a path of data.')

args = parser.parse_args()

f = open(args.data_path, 'r')

data = json.load(f)

model = Doc2Vec.load("model/d2v.model")
# to find the vector of a document which is not in training data
test_data = list(jieba.lcut(args.sentence))

# to find most similar doc using tags
inferred_vector_dm = model.infer_vector(test_data)
similar_doc = model.dv.most_similar([inferred_vector_dm], topn=5)
print('To find most similar doc using tags:')
print(similar_doc)

print("Test Sentence: " + args.sentence + '\n')

for item in similar_doc:
    print('Sentence: ' + data[int(item[0])] + ', Distance: ' + str(item[1]) + '\n')
