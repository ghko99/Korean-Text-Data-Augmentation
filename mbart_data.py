from eunjeon import Mecab 
from dataset_utils import get_sents
import json
from tqdm import tqdm

# 어휘 형태소 셋을 출력
def Extract_Morpheme(sentence):
    mecab = Mecab()
    extract_morphs = ['NNG','NNP', 'NNB','NNBC','NR','NP',
                'VV','VA','VCP','VCN','VX','SN', 'MM']
    concept_set = []

    mecab_pos = mecab.pos(sentence)
    for pos in mecab_pos:
        split_pos = pos[1].split('+')
        if pos[1] in extract_morphs:
            concept_set.append(pos[0])
        elif split_pos[0][0] == 'V' and split_pos[1][0] == 'E':
            concept_set.append(pos[0])
    return '#'.join(concept_set)

# mbart train을 위한 dataset 저장
def save_data():
    train_sents,valid_sents = get_sents()
    with open('./data/train.json','w',encoding='utf-8-sig') as f:
        for sent in tqdm(train_sents):
            res = {"concept-set":Extract_Morpheme(sent),"scene":sent}
            json.dump(res,f,ensure_ascii=False)
            f.write('\n')
    f.close()
    with open('./data/valid.json','w',encoding='utf-8-sig') as f:
        for sent in tqdm(valid_sents):
            res = {"concept-set":Extract_Morpheme(sent),"scene":sent}
            json.dump(res,f,ensure_ascii=False)
            f.write('\n')
    f.close()

if __name__ == "__main__":
    save_data()
