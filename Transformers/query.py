from sentence_transformers import SentenceTransformer, util
import torch
import argparse
import pickle

print(torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--sentence', default='我每天佩戴手錶坐巴士上學！',
                    type=str, required=False, help='Please enter a sentence.')
parser.add_argument('--embedding_path', default='../embeddings/embeddings.pkl',
                    type=str, required=False, help='Please enter a path of embeddings.')

args = parser.parse_args()

query = [args.sentence]

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

#Load sentences & embeddings from disc
with open('../embeddings/embeddings.pkl', "rb") as fIn:
    stored_data = torch.load(fIn, map_location=torch.device('cpu'))
    stored_sentences = stored_data['sentences']
    stored_embeddings = stored_data['embeddings']

query_embeddings = model.encode(query, convert_to_tensor=True)

#Compute cosine-similarities for each sentence with each other sentence
cosine_scores = util.pytorch_cos_sim(stored_embeddings, query_embeddings)

print(cosine_scores[0:20])

# Output the pairs with their score
for i in range(len(stored_sentences[0:20])):
    print("{}...... \t\t {} \t\t Score: {:.4f} \n".format(stored_sentences[i][0:15], query[0], cosine_scores[i][0]))
