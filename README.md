[Korean](README.md) | [English](README_ENG.md)
# Text-Classification-Transformers
**Easy text classification for everyone**

텍스트 분류 문제는 자연어 처리 분야에서 가장 쉽게 접할 수 있으며 다양한 곳에 활용될 수 있습니다.     

다만, 주어진 데이터를 전처리를 해야하고 해당 전처리에 맞게 모델의 데이터 파이프라인을 만들어야 합니다.

본 Repository는 텍스트 분류 데이터가 특정한 구조로 전처리가 되어있다면, 손쉽게 Transformers (BERT) 류의 모델로 텍스트 분류를 할 수 있도록 하는 것이 목적입니다.

빠르고 편리한 구현을 위해 [Huggingface transformers](https://github.com/huggingface/transformers)의 [AutoModels](https://huggingface.co/transformers/master/model_doc/auto.html)을 바탕으로 구현하였습니다.


## Data Preprocessing
텍스트 분류 작업에 해당하는 데이터는 `data_in` 폴더 내에 `train.csv`, `dev.csv`, `test.csv` 로 존재해야합니다.

또한 각 데이터들은 `label,text` 형식으로 구성되어합니다.

여기서 `label`이 총 2개일 경우 `0`, `1`로 표현이 되고, `N`개일 경우 `0` ~ `N-1`로 표현해야합니다.

현재 한국어 데이터 `nsmc`, `kornli`, 영어 데이터 `sst2`, `sst5`가 기본적으로 포함되어 있습니다.

기본적으로 제공되는 데이터는 `utils/$dataset{nsmc, kornli, sst}_preprocess.py`를 통해 전처리를 할 수 있습니다.

`nsmc`를 이용하여 `label,text` 구성의 데이터셋을 아래와 같은 방법으로 만들 수 있습니다.

```shell script
$ python utils/nsmc_preprocess.py

Number of dataset : 150000
Number of train dataset : 120000
Number of dev dataset : 30000
```

결과는 `data_in` 폴더에서 `train.csv`, `dev.csv`, `test.csv`로 확인할 수 있습니다.

```shell script
$ head data_in/train.csv

0,아 더빙.. 진짜 짜증나네요 목소리
1,흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나
0,너무재밓었다그래서보는것을추천한다
0,교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정
1,사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다
...
```

여기서 `0`은 Negative를, `1`은 Positive 레이블을 표현합니다.

`kornli`와 `sst2`, `sst5`도 이와 같은 방식으로 전처리 및 데이터를 구성 할 수 있습니다.

`kornli`의 경우 텍스트 분류 문제로 해결하기 위해 2개의 문장을 `[SEP]` 토큰을 이용하여 합쳐서 사용하게 됩니다.

```text
1,오두막집 문을 부수고 땅바닥에 쓰러졌어 [SEP] 나는 문을 박차고 들어가 쓰러졌다.
0,어른과 아이들을 위한 재미. [SEP] 외동아이들을 위한 재미.
2,"그래, 넌 학생이 맞아 [SEP] 넌 기계공 학생이지?"
```

`sst2`와 `sst5`의 경우 하나의 `sst_preprocess.py` 파일을 사용하여 아래와 같은 방식으로 동작합니다

```shell script
$ python utils/sst_preprocess.py --task sst2

Number of data in data_in/sst2/stsa_binary_train.txt : 6920
Number of data in data_in/sst2/stsa_binary_dev.txt : 872
Number of data in data_in/sst2/stsa_binary_test.txt : 1821

$ python utils/sst_preprocess.py --task sst5

Number of data in data_in/sst5/stsa_fine_train.txt : 8544
Number of data in data_in/sst5/stsa_fine_dev.txt : 1101
Number of data in data_in/sst5/stsa_fine_test.txt : 2210

```

**기본적으로 제공하지 않는 데이터의 경우 `label,text` 형태로 만들어 `data_in` 폴더 내에 `train.csv`, `dev.csv`, `test.csv`로 존재해야합니다.**

## Model

[Huggingface models](https://huggingface.co/models) 에서 지원하는 모델 중 대부분의 모델을 지원합니다.

하지만 `AutoModelForSequenceClassification`을 지원하지 않는 모델일 경우 지원을 하지 않습니다.

[Huggingface models](https://huggingface.co/models) 지원하는 모델의 경우 Huggingface transformers와 같은 방식으로 `model_name_or_path` argument를 통하여 사용할 수 있습니다.

`model_name_or_path` argument를 사용하지 않고 조금 더 간편하게 `model` argument를 사용하여 불러올 수 있는 모델이 존재합니다.

해당 모델들은 아래와 같습니다.

```text
MODEL = {
    "bert": "bert-base-multilingual-cased",
    "albert": "albert-base-v2",
    "bart": "facebook/bart-base",
    "camembert": "camembert-base",
    "distilbert": "distilbert-base-uncased",
    "electra": "google/electra-base-discriminator",
    "flaubert": "flaubert/flaubert_base_cased",
    "longformer": "allenai/longformer-base-4096",
    "mobilebert": "google/mobilebert-uncased",
    "roberta": "roberta-base",
    "kobert": "monologg/kobert",
    "koelectra": "monologg/koelectra-base-v2-discriminator",
    "distilkobert": "monologg/distilkobert",
    "kcbert": "beomi/kcbert-base"
}
```

`model` argument를 사용한 방법은 [shell-script](sh/nsmc/run_bert_base_multilingual_cased_nsmc.sh#L10) 파일을 참조하세요.

**Text classification task 에서는 `token_type_embedding`이 필요 없고, 여러 모델에서 지원하지 않으므로 `token_type_embedding`은 모든 모델에서 지원하지 않도록 하였습니다.**

`token_type_embedding`이 없더라도 `[SEP]` 토큰을 이용하여 두 문장을 붙여서 텍스트 분류를 하였을 때도 어느정도 [성능](#Result) 을 얻을 수 있었습니다.

학습된 모델을 로드해서 테스트할 경우 `model_name_or_path`에 실제 모델 및 파일들이 저장된 폴더를 지정해주어야 합니다.

**kobert와 distilkobert의 경우 학습된 모델을 로드할 때 폴더명에 `kobert`가 포함되도록 하여야합니다**

## Requirements
```text
torch==1.6.0
torchvision==0.7.0
tensorboard==2.3.0
transformers==3.0.2
```

## Usage

```shell script
python main.py \
        --do_train \
        --do_eval \
        --do_predict \
        --evaluate_during_training \
        --output_dir <save_path> \
        --data_dir <data_path> \
        --cache_dir <cache_save_path> \
        --overwrite_output_dir \
        --model <model_name> \
        --model_name_or_path <model_name_or_path> \
        --seed <seed> \
        --save_total_limit <num> \
        --learning_rate <learning_rate> \
        --per_device_train_batch_size <train_batch_size> \
        --per_device_eval_batch_size <eval_batch_size> \
        --num_train_epochs <epoch> \
        --max_seq_length <max_length> \
        --task_name <task_name> \
        --num_labels <num_labels> \
        --eval_steps <eval_steps> \
        --logging_steps <logging_steps> \
        --save_steps <save_steps> \
        --warmup_steps <warmup_steps> \
        --gradient_accumulation_steps <gradient_accumulation_steps>
```

**`model` 과 `model_name_or_path` 중 하나는 필수적으로 입력하여야 합니다.**
**Dataset의 label 수에 맞게 `num_labels`를 조절해주어야 합니다.**

Argument에 대한 설명은 [Huggingface transformers doc](https://huggingface.co/transformers/main_classes/trainer.html?highlight=arguments#trainingarguments) 또는 아래와 같은 방법으로 확인할 수 있습니다.
```shell script
python main.py -h
```
또한 기본적으로 제공된 [shell-script](sh) 파일들을 통해 예제를 확인할 수 있습니다. 

**실행예제는 [google colab example](https://colab.research.google.com/drive/1_54nFGE-t0rJYt-kgkbkbluWN74dC8jr?usp=sharing) 에서 확인할 수 있습니다.**

## Result

본 실험의 결과는 [shell-script](sh)를 이용하여 실험하였으며, Hyper-parameter tuning을 하지 않은 테스트입니다.

다양한 Hyper-parameter tuning을 통해 더 좋은 성능을 얻을 수 있으며, 본 성능은 참고용으로만 사용해주세요.

**KoNLI 에서는 `token_type_embedding`을 사용하지 않고  `[SEP]` 토큰으로 두 문장을 연결 시키는 방법을 사용하였습니다.**


**Korean**

|Model|NSMC|KoNLI|
|:---|:---:|:---:|
|bert-base-multilingual-cased|0.8748|0.7666|
|kobert|0.903|0.7992|
|koelectra-base-v2-discriminator|0.8976|0.8193|
|distilkobert |0.8860|0.6886|
|kcbert-base |0.901|0.7572|


**English**

|Model|SST-2|SST-5|
|:---|:---:|:---:|
|bert-base-multilingual-cased|0.8775|0.4945|
|bert-base-uncased|0.9231|0.5533|
|albert-base-v2|0.9192|0.5565|
|distilbert-base-uncased|0.9115|0.5298|
|mobilebert-uncased |0.9071|0.5416|
|roberta-base |0.9450|0.5701|
|longformer-base-4096|0.9511|0.5760|
|bart-base |0.9261|0.5606|
|electra-base-discriminator |0.9533|0.5868|


## Reference
[Huggingface Transformers](https://github.com/huggingface/transformers)  
[Huggingface Models](https://huggingface.co/models)
[KoBERT](https://github.com/SKTBrain/KoBERT)  
[KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)  
[DistilKoBERT](https://github.com/monologg/DistilKoBERT)  
[KoELECTRA](https://github.com/monologg/KoELECTRA)  
[KcBERT](https://github.com/Beomi/KcBERT)  
[NSMC](https://github.com/e9t/nsmc)  
[KorNLI](https://github.com/kakaobrain/KorNLUDatasets)  
[SST2, SST5](https://nlp.stanford.edu/sentiment/)  
