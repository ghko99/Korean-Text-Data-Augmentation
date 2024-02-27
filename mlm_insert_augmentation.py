from transformers import AutoModelForMaskedLM, AutoTokenizer
from dataset_utils import get_sents
import random
import torch
import copy
import json
from tqdm import tqdm

#단어 사이에 mask 토큰 삽입
def insert_mask(sent,tokenizer,ratio):
    splitted_sent = sent.split()
    span = int(round(len(splitted_sent) * ratio))
    random_idx = random.sample(range(len(splitted_sent)), span)
    for idx in random_idx:
        splitted_sent.insert(idx,tokenizer.mask_token)
    return ' '.join(splitted_sent)

def mlm_insert_augment(sent,model,tokenizer,ratio):
    masked_sent = insert_mask(sent,tokenizer,ratio)
    inputs = tokenizer(masked_sent,
                       return_attention_mask = True,
                       return_tensors = "pt",
                       truncation = True,
                       padding = True)
    masked_input_id, attention_mask = inputs.input_ids, inputs.attention_mask
    with torch.no_grad():
        masked_input_id,attention_mask = masked_input_id.cuda(), attention_mask.cuda()
        output = model(masked_input_id,attention_mask)
        logits = output["logits"][0]
    copied = copy.deepcopy(masked_input_id.cpu().tolist()[0])
    for i in range(len(copied)):
        if copied[i] == tokenizer.mask_token_id:
            prob = logits[i].softmax(dim=0)
            probability, candidate = prob.topk(5)
            copied[i] = candidate[0]
    copied = tokenizer.decode(copied,skip_special_tokens=True)
    return copied

if __name__ == "__main__":
    model = AutoModelForMaskedLM.from_pretrained("ghko99/KR-ELECTRA-generator-for-aug").cuda()
    tokenizer = AutoTokenizer.from_pretrained("ghko99/KR-ELECTRA-generator-for-aug")
    sents,_ = get_sents()
    mapping = dict()
    for sent in tqdm(sents):
        aug_sent = mlm_insert_augment(sent,model,tokenizer,0.15)
        mapping[sent] = aug_sent
    
    with open('./data/mapping_insert.json','w',encoding='utf-8-sig') as f:
        json.dump(mapping,f,indent='\t',ensure_ascii=False)
    f.close()
