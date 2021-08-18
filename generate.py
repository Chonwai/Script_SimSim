from gensim.models.doc2vec import Doc2Vec
import jieba
from nltk.tokenize import word_tokenize
import nltk
import json

f = open('data/train.json', 'r')

data = json.load(f)

test_sentence = "屌你老母。"

model = Doc2Vec.load("model/d2v.model")
# to find the vector of a document which is not in training data
# test_data = word_tokenize("English is a West Germanic language originally spoken by the inhabitants of early medieval England.[3][4][5] It is named after the Angles, one of the ancient Germanic peoples that migrated to the area of Great Britain that later took their name, England. Both names derive from Anglia, a peninsula on the Baltic Sea. English is most closely related to Frisian and Low Saxon, while its vocabulary has been significantly influenced by other Germanic languages, particularly Old Norse (a North Germanic language), as well as Latin and French.[6][7][8].")
# test_data = word_tokenize(test_sentence)
# test_data = word_tokenize("I go to school by bus.")
test_data = list(jieba.cut(test_sentence))

# to find most similar doc using tags
inferred_vector_dm = model.infer_vector(test_data)
similar_doc = model.dv.most_similar([inferred_vector_dm], topn=5)
print('To find most similar doc using tags:')
print(similar_doc)

print("Test Sentence: " + test_sentence + '\n')

for item in similar_doc:
    print('Sentence: ' + data[int(item[0])] + ', Distance: ' + str(item[1]) + '\n')
