# Making Augset
import pandas as pd
from tqdm import tqdm
# from chef import chef_augment
import joblib
import json
tqdm.pandas()

with open('./data/mapping_electra.json','r',encoding='utf-8-sig') as f:
    mlm_replacement_mapping = json.load(f)
f.close()

with open('./data/mapping_insert.json','r',encoding='utf-8-sig') as f:
    mlm_insert_mapping = json.load(f)
f.close()

with open('./data/mapping_mbart.json','r',encoding='utf-8-sig') as f:
    mbart_mapping = json.load(f)
f.close()

with open('./data/mapping_t5.json','r',encoding='utf-8-sig') as f:
    t5_mapping = json.load(f)
f.close()

orig_train = pd.read_json('sts/datasets/klue-sts-v1.1_train.json')

# # random insertion 
def apply_random_masking_insertion(x):
    return mlm_insert_mapping[x]

def apply_random_masking_replacement(x):
    return mlm_replacement_mapping[x]

def apply_t5(x):
    return t5_mapping[x]

def apply_mbart(x):
    return mbart_mapping[x][0]

random_masking_insertion_train = orig_train.copy()
pool = joblib.Parallel(n_jobs=8, prefer='threads')
mapper = joblib.delayed(apply_random_masking_insertion)
tasks = [mapper(row) for i, row in random_masking_insertion_train['sentence1'].items()]
random_masking_insertion_train['sentence1'] = pool(tqdm(tasks))

tasks = [mapper(row) for i, row in random_masking_insertion_train['sentence2'].items()]
random_masking_insertion_train['sentence2'] = pool(tqdm(tasks))

random_masking_insertion_augset = pd.concat([orig_train, random_masking_insertion_train])
random_masking_insertion_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
print(len(random_masking_insertion_augset))
random_masking_insertion_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_random_masking_insertion_augset.json')

random_masking_replace_train = orig_train.copy()
pool = joblib.Parallel(n_jobs=8, prefer='threads')
mapper = joblib.delayed(apply_random_masking_replacement)
tasks = [mapper(row) for i, row in random_masking_replace_train['sentence1'].items()]
random_masking_replace_train['sentence1'] = pool(tqdm(tasks))

tasks = [mapper(row) for i, row in random_masking_replace_train['sentence2'].items()]
random_masking_replace_train['sentence2'] = pool(tqdm(tasks))

random_masking_replace_augset = pd.concat([orig_train, random_masking_replace_train])
random_masking_replace_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
print(len(random_masking_replace_augset))
random_masking_replace_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_random_masking_replace_augset.json')


mbart_train = orig_train.copy()
pool = joblib.Parallel(n_jobs=8, prefer='threads')
mapper = joblib.delayed(apply_mbart)
tasks = [mapper(row) for i, row in mbart_train['sentence1'].items()]
mbart_train['sentence1'] = pool(tqdm(tasks))

tasks = [mapper(row) for i, row in mbart_train['sentence2'].items()]
mbart_train['sentence2'] = pool(tqdm(tasks))

mbart_augset = pd.concat([orig_train, mbart_train])
mbart_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
print(len(mbart_augset))
mbart_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_mbart_augset.json')


t5_train = orig_train.copy()
pool = joblib.Parallel(n_jobs=8, prefer='threads')
mapper = joblib.delayed(apply_t5)
tasks = [mapper(row) for i, row in t5_train['sentence1'].items()]
t5_train['sentence1'] = pool(tqdm(tasks))

tasks = [mapper(row) for i, row in t5_train['sentence2'].items()]
t5_train['sentence2'] = pool(tqdm(tasks))

t5_augset = pd.concat([orig_train, t5_train])
t5_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
print(len(t5_augset))
t5_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_t5_augset.json')


