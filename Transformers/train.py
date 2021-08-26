from sentence_transformers import SentenceTransformer, util
import pickle
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='../data/data_watch.json',
                    type=str, required=False, help='Please enter a path of data.')


args = parser.parse_args()

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

f = open(args.data_path, 'r', encoding='utf-8')
sentences = json.load(f)

# Single list of sentences
# sentences = ['男士佩戴手錶有哪些好處？本篇講給你講解下，是否都知道呢？',
#              '此款男士手錶是瑞士品牌天梭的魅時系列，在當下是熱銷產品。此款手錶除繼承以往的錶盤簡約設計以外，還在細節上下功夫，精緻的錶冠360度無死角設計，30米生活防水，不僅讓你佩戴時舒適，而且更加放心。',
#              '手錶是女人展示魅力的佩飾。一款優雅高貴的腕錶完全可以與女士們的各種珠寶媲美。尤其是追求時尚品位的現代女性，腕間的方寸不經意地透露出風情萬種，也讓社交場合的女人更加耀眼。',
#              '推薦這款calvinklein卡爾文克雷恩手錶，來自美國知名品牌。它採用精準完美的瑞士自動機械機芯，分毫不差。低調奢華的水晶玻璃表鏡耐磨實用，銀色的編織錶帶配著銀色的錶盤和銀色的指標和刻度，簡約時尚。精緻的蝴蝶雙扣表扣讓它牢牢戴在手腕上。不管是送家人還是女友男朋友都適宜，這是一款情侶表。贊',
#              '蘋果智慧運動手錶，從休閒步行到高強度騎行記錄各項運動都行，很多時候我們常常忘了手機放在哪裡，手機與手錶連線的狀態下，即可輕按3h鍵，通過鈴聲提示找到手機。',
#              '採用優質的面料柔軟透氣，防滑橡膠大底適合各種路況，鞋內柔軟的內裡設計，舒適透氣，讓你的雙腳可以自如的呼吸，鞋面設計，簡單大方，時尚潮流，搭配一頂紳士帽，增加一些熟男氣息，墨鏡、手錶、有質感的簡約素色褲子，這些小細節處理好，就很容易讓t恤穿出highfashion質感。',
#              '面料再經水洗工藝處理，使面料更加的舒適柔軟，充分考慮到身體的穿著舒適度，男士百搭的本性就足以讓所有男性們為之瘋狂了，時尚頸部，精美而富有質感，颳起一種簡單的街頭時尚風，可以嘗試搭配咖色手錶或者黑色手錶，來一場說走就走的旅行。',
#              '海鷗男士手錶，時尚的外觀結構，充分體現你的典雅風格，鏡面硬度高，耐磨性很好，讓你帶上經典不乏時尚美感。']

#Compute embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

#Store sentences & embeddings on disc
with open('../embeddings/embeddings.pkl', "wb") as fOut:
    pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)