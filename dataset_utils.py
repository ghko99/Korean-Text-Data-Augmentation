import json

def get_sents():
    with open('./data/klue_sts_train.json','r',encoding='utf-8-sig') as f:
        klue_sts_train = json.load(f)
    f.close()
    with open('./data/klue_sts_valid.json','r',encoding='utf-8-sig') as f:
        klue_sts_valid = json.load(f)
    f.close()
    train_sents = []
    valid_sents = []
    for key,val in klue_sts_train.items():
        train_sents.append(val['sentence1'])
        train_sents.append(val['sentence2'])
    for key,val in klue_sts_valid.items():
        valid_sents.append(val['sentence1'])
        valid_sents.append(val['sentence2'])

    return train_sents, valid_sents
