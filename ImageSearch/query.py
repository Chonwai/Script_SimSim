from sentence_transformers import SentenceTransformer, util
import argparse
import pickle
import numpy as np
from PIL import Image
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--sentence', default='我每天佩戴手錶坐巴士上學！',
                    type=str, required=False, help='Please enter a sentence.')
parser.add_argument('--image_path', default='images/two_dogs_in_snow.jpg',
                    type=str, required=False, help='Please enter a path of image.')
# parser.add_argument('--embedding_path', default='../embeddings/embeddings.pkl',
#                     type=str, required=False, help='Please enter a path of embeddings.')

args = parser.parse_args()

# query = [args.sentence]

model = SentenceTransformer('clip-ViT-B-32')
multilingualModel = SentenceTransformer('clip-ViT-B-32-multilingual-v1')

# Load sentences & embeddings from disc
# with open(args.embedding_path, "rb") as fIn:
#     stored_data = pickle.load(fIn)
#     stored_sentences = stored_data['sentences']
#     stored_embeddings = stored_data['embeddings']

# query_embeddings = model.encode(query, convert_to_tensor=True)

# Compute cosine-similarities for each sentence with each other sentence
# cosine_scores = util.pytorch_cos_sim(stored_embeddings, query_embeddings)

# score_list = list(cosine_scores)

# top_10_idx = np.argsort(score_list)[-10:]
# score = 0

# # Output the pairs with their score
# for i in top_10_idx:
#     print("{}...... \t\t {} \t\t Score: {:.4f}".format(stored_sentences[i][0:15], query[0], cosine_scores[i][0]))
#     score = score + cosine_scores[i][0]

# print("\n{}\tScore: {}".format(query[0], score/len(top_10_idx)))

# Encode an image:
img_emb = model.encode(Image.open(args.image_path))

# Encode text descriptions
text_emb = multilingualModel.encode(
    ['Two dogs in the snow', 'A cat on a table', 'A picture of London at night', '兩隻狗在雪地上', '兩隻貓在草地上'])

# Compute cosine similarities
cos_scores = util.cos_sim(img_emb, text_emb)
print(cos_scores)
