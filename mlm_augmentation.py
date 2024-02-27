import torch
import copy
from transformers import AutoModelForMaskedLM,AutoTokenizer
import json
from tqdm import tqdm
from mlm_train import mask_tokens
from dataset_utils import get_sents

#text가 한국어인지 판별
def is_korean(text):
    if text[0] == '#':
        text = text[2:]

    for char in text:
        if not '가' <= char <= '힣':
            return False
    return True

# 후보토큰 선정 (한국어이면서, 원본 토큰과 다른고 후보토큰 앞뒤로 반복이 없어야함) 
def candidate_filtering(tokenizer,
                        input_ids:list,
                        idx:int,
                        org:int,
                        candidates) -> int:
    
    org_token = tokenizer.convert_ids_to_tokens([org])[0]
    candidate_tokens = tokenizer.convert_ids_to_tokens(candidates.cpu().tolist())
    candidates = copy.deepcopy(candidates.cpu().tolist())
    for rank, token in enumerate(candidate_tokens):
        if org_token!=token and is_korean(org_token) and is_korean( token):
            if input_ids[idx-1]==candidates[rank] or input_ids[idx+1]==candidate_tokens[rank]:
                continue
            return candidates[rank]
    return org

def tokenize(tokenizer,sent):
    encoded_dict = tokenizer(sent,
                return_attention_mask = True,
                return_tensors="pt",
                truncation = True,
                padding = True
    )
    input_id, attention_mask = encoded_dict.input_ids, encoded_dict.attention_mask
    return input_id,attention_mask

def mlm_replace_augment(model,tokenizer,sent,mlm_prob):
    
    model.eval()

    threshold = 0.95

    input_id, attention_mask = tokenize(tokenizer=tokenizer,sent=sent)
    org_ids = copy.deepcopy(input_id[0])

    masked_input_id , _ = mask_tokens(tokenizer=tokenizer,input_ids=input_id,mlm_prob=mlm_prob,do_rep_random=False)

    while masked_input_id.cpu().tolist()[0].count(tokenizer.mask_token_id) < 1:
        masked_input_id, _ = mask_tokens(tokenizer=tokenizer,input_ids=input_id,mlm_prob=mlm_prob,do_rep_random=False)

    with torch.no_grad():
        masked_input_id,attention_mask = masked_input_id.cuda(), attention_mask.cuda()
        output = model(masked_input_id,attention_mask)
        logits = output["logits"][0]

    copied = copy.deepcopy(masked_input_id.cpu().tolist()[0])
    for i in range(len(copied)):
        if copied[i] == tokenizer.mask_token_id:
            org_token = org_ids[i]
            prob = logits[i].softmax(dim=0)
            probability, candidates = prob.topk(5)
            if probability[0]<threshold:
                #probability가 0.95미만일경우 candidate filtering을 통해 토큰 선정 
                res = candidate_filtering(tokenizer, copied, i, org_token, candidates)
            else: 
                #만약 probability가 0.95이상이면 무조건 선택
                res = candidates[0]
            copied[i] = res 
    copied = tokenizer.decode(copied,skip_special_tokens=True)
    return copied

if __name__ == "__main__":

    train_sents ,valid_sents = get_sents()
    tokenizer = AutoTokenizer.from_pretrained('./model/snu')
    model = AutoModelForMaskedLM.from_pretrained('./model/snu').cuda()

    mapping = dict()
    for sent in tqdm(train_sents):
        aug_sent = mlm_replace_augment(model, tokenizer,sent,0.15)
        mapping[sent] = aug_sent
    with open('./data/mapping_electra.json','w',encoding='utf-8-sig') as f:
        json.dump(mapping,f,indent='\t',ensure_ascii=False)
    f.close()
