from transformers import AutoModelForSeq2SeqLM, MBart50TokenizerFast
import json
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from mbart_data import Extract_Morpheme
import copy
from mbart_data import get_sents

class DataGenerateArguments:
    sent_max_len = 128
    model_name_or_path = './model/mbart'
    batch_size = 8
    num_beams = 10
    no_repeat_ngram_size=3
    num_return_sequences=1
    
class Dataset(Dataset):
    def __init__(self,concept_set,label,tokenizer,sent_max_len):
        self.tokenizer = tokenizer
        self.data = concept_set
        self.label = label
        input_ids, attention_masks = [], []
        for sent in tqdm(self.data):
            encoded_dict = self.tokenizer(
                                sent,
                                max_length = sent_max_len,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True
            )
            input_ids.append(encoded_dict.input_ids)
            attention_masks.append(encoded_dict.attention_mask)

        self.input_ids = torch.cat(input_ids,dim=0)
        self.attention_mask = torch.cat(attention_masks,dim=0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input_id = self.input_ids[index]
        attention_mask = self.attention_mask[index]
        return torch.tensor(input_id).cuda(),torch.tensor(attention_mask).cuda(), self.label[index]


def mbart_augment(sent_list):
    #load arguments, model ,tokenizer
    gen_args = DataGenerateArguments()
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_args.model_name_or_path).cuda()
    tokenizer = MBart50TokenizerFast.from_pretrained(gen_args.model_name_or_path, src_lang="ko_KR", tgt_lang="ko_KR")
    
    #save augmentation results
    res = dict()
    labels = copy.deepcopy(sent_list)
    concept_sets = []
    for i in tqdm(range(len(labels))):
        concept_sets.append(Extract_Morpheme(labels[i]))

    dataset = Dataset(concept_sets,labels,tokenizer,gen_args.sent_max_len)
    dataloader = DataLoader(dataset,batch_size=gen_args.batch_size,shuffle=False)

    for batch in tqdm(dataloader):
        b_input_ids, b_attn_mask,b_label = batch[0],batch[1],batch[2]
        with torch.no_grad():
            sequences = model.generate(
                                    input_ids=b_input_ids,
                                    attention_mask=b_attn_mask,
                                    num_beams = gen_args.num_beams,
                                    max_length = gen_args.sent_max_len,
                                    no_repeat_ngram_size=gen_args.no_repeat_ngram_size,
                                    num_return_sequences=gen_args.num_return_sequences,
                                    )
        sequences = sequences.view(b_input_ids.size(0), gen_args.num_return_sequences, -1)
        for beam_outputs, label in zip(sequences,b_label):
            output_sents = []
            for beam_output in beam_outputs:
                output = tokenizer.decode(beam_output,skip_special_tokens=True)
                output_sents.append(output)
            res[label]=output_sents

        # save mbart augmentation results
        with open('./data/mapping_mbart.json','w',encoding='utf-8-sig') as f:
            json.dump(res,f,ensure_ascii=False,indent='\t')
        f.close()

if __name__ == "__main__":
    sents, _ = get_sents()
    mbart_augment(sents)
