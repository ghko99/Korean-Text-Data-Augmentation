# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import torch
import json
import time
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer,AutoModelForMaskedLM,AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from datetime import datetime
from tqdm import tqdm


#dataset
class Dataset(Dataset):
    def __init__(self, data,tokenizer) :
        self.data = data
        input_ids, attention_masks = [], []

        for line in tqdm(self.data):
            sent_max_len = 128
            encoded_dict = tokenizer(line,
                                     max_length=sent_max_len,
                                     padding = "max_length",
                                     truncation = True,
                                     return_attention_mask = True,
                                     return_tensors="pt"
            )
            input_ids.append(encoded_dict.input_ids)
            attention_masks.append(encoded_dict.attention_mask)

        self.input_ids = torch.cat(input_ids,dim=0)
        self.attention_masks = torch.cat(attention_masks,dim=0)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx) :
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]

        return input_id,attention_mask

#masking
def mask_tokens(tokenizer, input_ids:torch.Tensor, mlm_prob:float=0.15, do_rep_random:bool=True):
    '''
        Copied from huggingface/transformers/data/data_collator - torch.mask_tokens()
        Prepare masked tokens inputs/labels for masked language modeling
        if do_rep_random is True:
            80% MASK, 10% random, 10% original
        else:
            100% MASK
    '''
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_prob)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value = 0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100 # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    mask_rep_prob = 0.8
    if not do_rep_random:
        mask_rep_prob = 1.0
    indices_replaced = torch.bernoulli(torch.full(labels.shape, mask_rep_prob)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    if do_rep_random:
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
    return input_ids, labels

#sts dataset load
def get_dataset(tokenizer,batch=64,shuffle=True):
    
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

    train_dataset = Dataset(train_sents,tokenizer)
    valid_dataset = Dataset(valid_sents,tokenizer)
    train_dataloader = DataLoader(train_dataset,batch_size=batch,shuffle=shuffle)
    valid_dataloader = DataLoader(valid_dataset,batch_size=batch,shuffle=shuffle)
    
    return train_dataloader,valid_dataloader

# augmentation을 위한 model mlm train
def MLM_train(train_dataloader,test_dataloader,epochs,model,tokenizer,optimizer,model_dir,mlm_prob=0.15):

    #log 파일 저장
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log = open(model_dir + "/log.txt", "w")

    loss_fn = nn.CrossEntropyLoss()

    #valid loss가 최솟값일때 모델 save
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []

    #loss 수렴시 train stop
    stop_count = 0

    for epoch in range(epochs):
        #경과 시간 저장용
        prev_time = time.time()

        model.train()
        train_loss = 0.0

        for step,batch in enumerate(train_dataloader):
            with autocast():
                b_input_ids,b_label = mask_tokens(tokenizer=tokenizer,input_ids=batch[0],mlm_prob=mlm_prob)
                b_input_ids,b_label = b_input_ids.cuda(), b_label.cuda()
                b_attn_mask = batch[1].cuda()
                model.zero_grad()
                outputs = model(input_ids=b_input_ids,attention_mask=b_attn_mask,labels=b_label)
                logits = outputs[1]

            loss_mx = b_label != -100
            logits = logits[loss_mx].view(-1,tokenizer.vocab_size)
            labels = b_label[loss_mx].view(-1)
            loss = loss_fn(logits,labels)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            #step 별로 진행 상황 출력
            if step % 100 == 0 and step != 0:
                log.write(f'Epoch [{epoch+1}/{epochs}], Batch [{step+1}/{len(train_dataloader)}], Train Loss: {train_loss/(step+1):.4f}\n')
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{step+1}/{len(train_dataloader)}], Train Loss: {train_loss/(step+1):.4f}')


        avg_train_loss = train_loss/len(train_dataloader)
        train_losses.append(avg_train_loss)


        model.eval()

        #evaluation
        valid_loss = 0.0
        for batch in test_dataloader:
            b_input_ids,b_label = mask_tokens(tokenizer=tokenizer,input_ids=batch[0] , mlm_prob=mlm_prob)
            b_input_ids,b_label = b_input_ids.cuda(), b_label.cuda()
            b_attn_mask = batch[1].cuda()

            with torch.no_grad():
                outputs = model(input_ids=b_input_ids,attention_mask=b_attn_mask,labels=b_label)
            
            logits = outputs[1]
            loss_mx = b_label != -100
            logits = logits[loss_mx].view(-1,tokenizer.vocab_size)
            labels = b_label[loss_mx].view(-1)
            loss = loss_fn(logits,labels)
            valid_loss += loss.item()


        avg_valid_loss = valid_loss/len(test_dataloader)
        valid_losses.append(avg_valid_loss)

        #valid loss가 최소일때 모델 저장
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            model_to_save = model.module if hasattr(model,'module') else model
            model_to_save.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            stop_count = 0
        else:
            stop_count += 1        
        
        # epoch마다 진행상황 출력
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Time Elapsed: {(time.time()-prev_time):.4f}')
        log.write(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Time Elapsed: {(time.time()-prev_time):.4f}\n')
        
        #30 epoch동안 개선이없으면 학습종료
        if stop_count > 30:
            break

    return train_losses,valid_losses

# train loss, valid loss 그래프 저장
def show_hist(train_losses, valid_losses, model_dir):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'loss_results.png'))

if __name__ == '__main__':

    model_name = "snunlp/KR-ELECTRA-generator"
    model_dir = './model/snu'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).cuda()

    optimizer = AdamW(model.parameters(), 
                      lr=1e-6, 
                      eps=1e-6, 
                      weight_decay = 0.01
                      )
    epochs  = 1000

    train_dataloader ,test_dataloader = get_dataset(tokenizer)
    train_losses, valid_losses = MLM_train(train_dataloader,test_dataloader,epochs,
                                          model,tokenizer,optimizer,model_dir)

    show_hist(train_losses,valid_losses,model_dir)