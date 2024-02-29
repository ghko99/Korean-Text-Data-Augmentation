from datasets import load_dataset
import json
from tqdm import tqdm

def dataset_to_json(dataset):
    new_dataset = {}
    for i in tqdm(range(len(dataset))):
        guid = dataset[i]['guid']
        sentence1 = dataset[i]['sentence1']
        sentence2 = dataset[i]['sentence2']
        labels = dataset[i]['labels']
        source = dataset[i]['source']
        new_dataset[guid] = {
            'sentence1':sentence1,
            'sentence2':sentence2,
            'labels':labels,
            'source':source           
        }
    return new_dataset

if __name__ == "__main__":
    with open('./data/klue-sts-v1.1_dev.json','r',encoding='utf-8-sig') as f:
        valid_dataset = json.load(f)
    f.close()


    print(valid_dataset)
    # train_json = dataset_to_json(train_dataset)
    valid_json = dataset_to_json(valid_dataset)

    # with open('./data/klue_sts_train.json','w',encoding='utf-8-sig') as f:
    #     json.dump(train_json,f,indent='\t',ensure_ascii=False)
    # f.close()

    with open('./data/klue_sts_valid.json','w',encoding='utf-8-sig') as f:
        json.dump(valid_json,f,indent='\t',ensure_ascii=False)
    f.close()
