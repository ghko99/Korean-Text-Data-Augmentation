from transformers import pipeline
import time
import numpy as np
from dataset_utils import get_sents
import json

def t5_augment(sent_list : list, pipe : pipeline, batch_size):
    sent_len_list = np.vectorize(len)(sent_list)
    #길이가 160 이하일 경우 drop
    under_160_sent_idx = np.where(np.array(sent_len_list) < 160)
    
    sent_list = np.array(sent_list)
    candidate_sentences = sent_list[under_160_sent_idx]
    
    tmp_ix = 0
    #t5 predict
    while(tmp_ix < len(candidate_sentences)):
        prev = time.time()
        batch = []
        if tmp_ix+batch_size > len(candidate_sentences):
            batch = candidate_sentences[tmp_ix:]
        else:
            batch = candidate_sentences[tmp_ix:tmp_ix+batch_size]
        outputs = pipe(batch.tolist(),max_length=160)
        new_sentences = [output['generated_text'] for output in outputs]
        candidate_sentences[tmp_ix:tmp_ix+batch_size] = new_sentences
        tmp_ix += len(batch)
        print(tmp_ix , '/' , len(candidate_sentences), ' : time elapsed: '  , time.time()-prev)
    #save augmentation result
    sent_list[under_160_sent_idx] = candidate_sentences
    return sent_list

if __name__ == "__main__":
    pipe = pipeline("text2text-generation", model="psyche/KoT5-paraphrase-generation", device=0 )
    batch_size = 64
    sents, _ = get_sents()
    aug_sents = t5_augment(sents,pipe,64)
    mapping = dict()
    for i in range(len(sents)):
        mapping[sents[i]] = aug_sents[i]
    
    with open('./data/mapping_t5.json','w',encoding='utf-8-sig') as f:
        json.dump(mapping,f,indent='\t',ensure_ascii=False)
    f.close()