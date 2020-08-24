[Korean](README.md) | [English](README_ENG.md)
# Text-Classification-Transformers
Easy text classification for everyone

텍스트 분류 문제는 자연어 처리 분야에서 가장 쉽게 접할 수 있으며 다양한 곳에 활용될 수 있습니다.     

다만, 주어진 데이터를 전처리를 해야하고 해당 전처리에 맞게 모델의 데이터 파이프라인을 만들어야 합니다.

본 Repository는 텍스트 분류 데이터가 특정한 구조로 전처리가 되어있다면, 손쉽게 Transformers (BERT) 류의 모델로 텍스트 분류를 할 수 있도록 하는 것이 목적입니다.


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
2,그들이 물어본 질문들이 흥미롭거나 합법적이지 않았다는 것은 아니다(대부분은 이미 묻고 대답한 범주에 속했지만). [SEP] 이 주제에 대해 상담한 포커스 그룹에 따르면 모든 질문이 흥미로웠다.
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

기본적으로 제공하지 않는 데이터의 경우 `label,text` 형태로 `data_in` 폴더 내에 `train.csv`, `dev.csv`, `test.csv`로 존재해야합니다.

## Model

token type embedding 지원 X

## Usage


## Result

|Model|NSMC|KoNLI|
|:---|:---:|:---:|
|bert-base-multilingual-cased|0.8748|0.7666|
|kobert|0.903|0.7992|
|koelectra-base-v2-discriminator|0.8976|0.8193|
|distilkobert |0.8860|0.6886|
|kcbert-base |0.901|0.7572|

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
[KoBERT](https://github.com/SKTBrain/KoBERT)
[KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)
[DistilKoBERT](https://github.com/monologg/DistilKoBERT)
[KoELECTRA](https://github.com/monologg/KoELECTRA)
[KcBERT](https://github.com/Beomi/KcBERT)
[NSMC](https://github.com/e9t/nsmc)
[KorNLI](https://github.com/kakaobrain/KorNLUDatasets)
[SST2, SST5](https://nlp.stanford.edu/sentiment/)
