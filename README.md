
# pre-trained language model을 활용한 한국어 데이터 증강
BART, BERT, T5 기반 언어 모델을 활용한 한국어 데이터 증강법 <br>
Environments
```
python: 3.8.18
OS: Windows 11
```
## 1. BART-based Korean Text data Augmentation
[Korean-CommonGen](https://github.com/J-Seo/Korean-CommonGen) 과 
[CHEF in the Language Kitchen: A Generative Data Augmentation Leveraging Korean Morpheme Ingredients](https://aclanthology.org/2023.emnlp-main.367/) 
논문을 참고한 mbart-50-large 기반 한국어 텍스트 데이터 증강법.<br>

![형태소](https://github.com/ghko99/korean-augmentation/assets/115913818/f9591f3e-dfdf-4b7f-ac0f-a541ba7f6314)

한국어는 실질형태소와 형식형태소로 나뉘며, 형식형태소를 바꿔가며 의미는 동일한 다양한 문장을 생성할 수 있음.<br>
해당 특성을 이용해 BART기반 한국어 데이터 증강을 구현할 수있음.

### 1.1 Data Augmentation algorithm
![chef](https://github.com/ghko99/korean-augmentation/assets/115913818/b4638235-996a-4403-943c-7b0e54e56be5)

mecab 형태소 분석기 활용해 어휘 형태소 set을 생성후, mbart-large 모델로 학습.
### 1.2 How to train mbart model
```
python mbart_train.py
```
학습 dataset은 [KLUE](https://huggingface.co/datasets/klue)의 Semantic Text Similarity(STS) 데이터셋을 사용함.
### 1.3 Arguments
```python
class ModelArguments:
    model_name_or_path = "facebook/mbart-large-50"
    setproc_model = "mbart_train"
    use_fast_tokenizer = True
    use_auth_token = False
    cache_dir = None
    model_revision = "main"

class DataTrainingArguments:
    train_file = './data/train.json'
    validation_file = './data/valid.json'
    test_file = './data/valid.json'
    preprocessing_num_workers=None
    overwrite_cache = False
    max_source_length = 1024  
    max_target_length = 128  
    pad_to_max_length = False
    num_beams = 10
    num_return_sequences = 10  
    ignore_pad_token_for_loss = True
```

### 1.4 Trained Model
🤗 [ghko99/mbart50-large-for-aug](https://huggingface.co/ghko99/mbart50-large-for-aug)
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ghko99/mbart50-large-for-aug")
model = AutoModelForSeq2SeqLM.from_pretrained("ghko99/mbart50-large-for-aug")
```
![스크린샷 2024-03-02 122036](https://github.com/ghko99/Korean-Text-Data-Augmentation/assets/115913818/aa2376ef-7439-403b-890d-c43b5b5b00fd)

### 1.5 Usage 
```python
from mbart_aug import mbart_augment

sent_list = ["무엇보다도 호스트분들이 너무 친절하셨습니다.",
            "주요 관광지 모두 걸어서 이동가능합니다.",
            "가족 모임 일정은 바꾸지 말도록 하십시오."]

aug_sents = mbart_augment(sent_list)
print(aug_sents)
```

### 1.6 Example Output
```
>>> [['무엇보다 호스트가 매우 친절하고 세심합니다.'],['주요 관광지를 걸어서 이동 가능합니다.'],['가족모임 일정을 바꾸지 말도록 하세요.']]
```


## 2. BERT-based Korean Text data Augmentation

### 2.1 BERT의 Masked Language Model을 활용한 데이터 증강 (Random Masking Replacement).
BERT의 MLM(Masked Language Model)은 문맥을 고려한 유의어 교체의 구현을 가능하게함.
![bert](https://github.com/ghko99/korean-augmentation/assets/115913818/909c993b-c5f2-4327-a439-fbf119c4452f)

#### 2.1.1 활용 사례
[Data Augmentation using Pre-trained Transformer Models](https://arxiv.org/abs/2003.02245)에서는 영어 데이터에서 BERT의 Masked Language Model을 활용해 Synonym Replacement(Random Masking Replacement)를 구현함.<br>
[한국어 상호참조해결을 위한 End-to-end 상호참조해결 모델과 데이터 증강 방법](https://www.dbpia.co.kr/journal/detail?nodeId=T15773139)에서는 한국어 데이터에서 BERT의 Random Masking Replacement를 활용.<br>
코드의 상당부분은 [MLM-data-augmentation](https://github.com/seoyeon9646/MLM-data-augmentation)을 참고함.

### 2.2 BERT의 Masked Language Model을 활용한 데이터 증강 (Random Masking Insertion).
[K-TACC(Korean Text Augmentation Considering Context)](https://github.com/kyle-bong/K-TACC)에서는 BERT의 Masked Language Model을 활용해 Random Insertion(Random Masking Insertion)을 구현함. <br>
기존 Replacement의 경우 반의어로 교체하는 경우가 있기 때문에 의미보존이 힘들고, 한국어 토큰화 기법의 특성으로 인해 어색한 문장이 여전히 존재함. <br>
하지만 한국어는 영어에 비해 어순이 유연하기 때문에 문맥을 알 수 있다면 단어 사이에 들어갈만한 토큰을 찾기가 쉬움. 따라서 Masked Language Model을 활용할 경우 특정 토큰을 [MASK]토큰으로 대체하는 것 보다 기존 토큰들은 유지한채 삽입하는것이 더욱 의미보존이 가능하며 자연스러운 문장이 나오기 쉬움.<br>
### 2.3 Data Augmentation algorithm (Random Masking Replacement)
#### 2.3.1 Pre-training
![mlm_pretrained](https://github.com/ghko99/korean-augmentation/assets/115913818/cb40497d-6efe-43f7-a305-a7b1f7eb0f07)
#### 2.3.2 How to pretrain model
```
python mlm_train.py
```
#### 2.3.3 Pre-trained Model
🤗 [ghko99/KR-ELECTRA-generator-for-aug](https://huggingface.co/ghko99/KR-ELECTRA-generator-for-aug)
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("ghko99/KR-ELECTRA-generator-for-aug")
model = AutoModelForMaskedLM.from_pretrained("ghko99/KR-ELECTRA-generator-for-aug")
```
#### 2.3.4 Random Masking Replacement

![mlm](https://github.com/ghko99/korean-augmentation/assets/115913818/a14b309d-7b24-4c19-b562-01c9ceeeebe6)
#### 2.3.5 Usage
```python
from mlm_augmentation import mlm_replace_augment
from transformers import AutoModelForMaskedLM,AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ghko99/KR-ELECTRA-generator-for-aug")
model = AutoModelForMaskedLM.from_pretrained("ghko99/KR-ELECTRA-generator-for-aug").cuda()

sent = "무엇보다도 호스트분들이 너무 친절하셨습니다."
aug_sent = mlm_replace_augment(model=model,tokenizer=tokenizer,sent=sent,mlm_prob=0.15)
print(aug_sent)
```
#### 2.3.6 Example Output
```
>>> 무엇보다도 호스트분이 매우 친절하셨습니다.
```
### 2.4 Data Augmentation algorithm (Random Masking Insertion)
#### 2.4.1 Random Masking Insertion

![rmi](https://github.com/ghko99/korean-augmentation/assets/115913818/a132f064-a49e-44b5-9d86-88eb676873b8)
앞서 pretrained된 모델을 사용해 Random Masking Insertion을 수행
#### 2.4.2 Usage
```python
from mlm_insert_augmentation import mlm_insert_augment
from transformers import AutoModelForMaskedLM,AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('ghko99/KR-ELECTRA-generator-for-aug')
model = AutoModelForMaskedLM.from_pretrained('ghko99/KR-ELECTRA-generator-for-aug').cuda()

sent = "무엇보다도 호스트분들이 너무 친절하셨습니다."
aug_sent = mlm_insert_augment(model=model,tokenizer=tokenizer,sent=sent,ratio=0.15)
print(aug_sent)
```
#### 2.4.3 Example Output
```
>>> 무엇보다도 호스트분들이 정말 너무 친절하셨습니다.
```
## 3. T5-based Korean Text data Augmentation
T5의 Paraphrase task로 입력문장과 의미는 같지만 다른 문장을 생성

![T5](https://github.com/ghko99/korean-augmentation/assets/115913818/7cf18e7e-7044-4540-a5e2-da9f8a37ba5d) <br>
Huggingface의 [psyche/KoT5-paraphrase-generation](https://huggingface.co/psyche/KoT5-paraphrase-generation/discussions)을 활용함
### 3.1 Usage
```python
from t5_augmentation import t5_augment
from transformers import pipeline

pipe = pipeline("text2text-generation", model="psyche/KoT5-paraphrase-generation", device=0)
sent_list = ["무엇보다도 호스트분들이 너무 친절하셨습니다.",
            "주요 관광지 모두 걸어서 이동가능합니다.",
            "가족 모임 일정은 바꾸지 말도록 하십시오."]
aug_sents = t5_augment(sent_list=sent_list,pipe=pipe,batch_size=64)
print(aug_sents)
```
### 3.2 Example Output
```
>>> ['무엇보다도 호스트분들이 너무 친절하셨습니다.','주요 관광지를 걸어서 모두 갈 수 있습니다.','가족 모임 일정은 변경하지 마십시오.']
```

## 4. Augmentation sample files
[KLUE](https://huggingface.co/datasets/klue)의 Semantic Text Similarity(STS) 데이터셋을 4가지 증강방식으로 1:1 비율(2배)로 증강함. <br><br>
Random Masking Replacement 증강 sample 파일: 
[mapping_electra.json](https://github.com/ghko99/Korean-Text-Data-Augmentation/blob/master/data/mapping_electra.json)<br>
Random Masking Insertion 증강 sample 파일: 
[mapping_insert.json](https://github.com/ghko99/Korean-Text-Data-Augmentation/blob/master/data/mapping_insert.json)<br>
mbart50-large 증강 sample 파일: 
[mapping_mbart.json](https://github.com/ghko99/Korean-Text-Data-Augmentation/blob/master/data/mapping_mbart.json)<br>
t5 증강 sample 파일: 
[mapping_t5.json](https://github.com/ghko99/Korean-Text-Data-Augmentation/blob/master/data/mapping_t5.json)<br>
## 5. Augmentation 성능 측정
[K-TACC](https://github.com/kyle-bong/K-TACC)의 STS 성능 측정 코드를 활용해 데이터 증강의 성능 측정을 진행함. <br>
baseline model은 [monologg/koelectra-base-v3-discriminator](https://huggingface.co/monologg/koelectra-base-v3-discriminator)을 활용했음.
### 5.1 training arguments:
```
data:
    shuffle: True
    augmentation: True
    max_length : 128

train:
  seed: 42
  batch_size: 32
  max_epoch: 4
  learning_rate: 5e-5
  logging_step: 100
  drop_out: 0.1
  warmup_ratio: 0.1
  weight_decay: 0.01

runner:
  options:
    num_workers: 8
```
### 5.2 KLUE STS 증강셋 생성
```
python augmentation.py
```
### 5.3 STS 성능 측정
```
sh train.sh
```
### 5.4 STS 성능 측정 결과
|           Augmentation           | Pearson's correlation |
|:--------------------------------:|:---------------------:|
|             Baseline             |   0.9241138696670532  |
| (BERT) Random Masking Replacemnt |   0.8990367650985718  |
|  (BERT) Random Masking Insertion |   0.9256289601325989  |
|       (BART) mbart50-large       |   0.9271315336227417  |
|  (T5) KoT5-paraphrase-generation |   0.915034294128418   |

BART와 BERT의 Random Masking Insertion 증강법이 STS에서 가장 높은 성능을 보여주는것으로 보임.
## 6. Reference
* [K-TACC](https://github.com/kyle-bong/K-TACC)
* [KLUE](https://huggingface.co/datasets/klue)
* [psyche/KoT5-paraphrase-generation](https://huggingface.co/psyche/KoT5-paraphrase-generation/discussions)
* [Data Augmentation using Pre-trained Transformer Models](https://arxiv.org/abs/2003.02245)
* [한국어 상호참조해결을 위한 End-to-end 상호참조해결 모델과 데이터 증강 방법](https://www.dbpia.co.kr/journal/detail?nodeId=T15773139)
* [MLM-data-augmentation](https://github.com/seoyeon9646/MLM-data-augmentation)
* [monologg/koelectra-base-v3-discriminator](https://huggingface.co/monologg/koelectra-base-v3-discriminator)
