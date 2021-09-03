from sentence_transformers import SentenceTransformer, util
import pickle
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='../data/data_watch.json',
                    type=str, required=False, help='Please enter a path of data.')
parser.add_argument('--save_embeddings_path', default='../embeddings/embeddings.pkl',
                    type=str, required=False, help='Please enter a path of saving embeddings result.')


args = parser.parse_args()

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

f = open(args.data_path, 'r', encoding='utf-8')
sentences = json.load(f)

#Compute embeddings
embeddings = model.encode(sentences)

#Store sentences & embeddings on disc
with open(args.save_embeddings_path, "wb") as fOut:
    pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)