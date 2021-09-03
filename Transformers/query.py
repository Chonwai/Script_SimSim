from sentence_transformers import SentenceTransformer, util
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--sentence', default='我每天佩戴手錶坐巴士上學！',
                    type=str, required=False, help='Please enter a sentence.')
parser.add_argument('--embedding_path', default='../embeddings/embeddings.pkl',
                    type=str, required=False, help='Please enter a path of embeddings.')

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

top_10_idx = np.argsort(score_list)[-10:]
score = 0

# Output the pairs with their score
for i in top_10_idx:
    print("{}...... \t\t {} \t\t Score: {:.4f}".format(stored_sentences[i][0:15], query[0], cosine_scores[i][0]))
    score = score + cosine_scores[i][0]

print("\n{}\tScore: {}".format(query[0], score/len(top_10_idx)))