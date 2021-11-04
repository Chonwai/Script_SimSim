from sentence_transformers import SentenceTransformer, util
import argparse
import pickle
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--sentence', default='我每天佩戴手錶坐巴士上學！',
                    type=str, required=False, help='Please enter a sentence.')
parser.add_argument('--embedding_path', default='../embeddings/embeddings.pkl',
                    type=str, required=False, help='Please enter a path of embeddings.')

start = time.time()

args = parser.parse_args()

query = [args.sentence]

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

#Load sentences & embeddings from disc
with open(args.embedding_path, "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['sentences']
    stored_embeddings = stored_data['embeddings']

query_embeddings = model.encode(query, convert_to_tensor=True)

#Compute cosine-similarities for each sentence with each other sentence
cosine_scores = util.pytorch_cos_sim(stored_embeddings, query_embeddings)

score_list = list(cosine_scores)

top_5_idx = np.argsort(score_list)[-5:]
score = 0

# Output the pairs with their score
for i in top_5_idx:
    print("{}......\n \t\t {} \t\t Score: {:.4f}".format(stored_sentences[i], query[0], cosine_scores[i][0]))
    score = score + cosine_scores[i][0]

end = time.time()

print("\n{}\tScore: {}".format(query[0], score/len(top_5_idx)))
print("Total time cost: " + str(end - start))