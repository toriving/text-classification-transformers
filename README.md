[Korean](README_KR.md) | [English](README.md)
# text-classification-transformers
**Easy text classification for everyone**

Text classification tasks are most easily encountered in the area of natural language processing and can be used in various ways.

However, the given data needs to be preprocessed and the model's data pipeline must be created according to the preprocessing.

The purpose of this Repository is to allow text classification to be easily performed with Transformers (BERT)-like models if text classification data has been preprocessed into a specific structure.

Implemented based on Huggingfcae transformers for quick and convenient implementation.

## Data Preprocessing
Data must exist as `train.csv`, `dev.csv`, and `test.csv` in the `data_in` folder.

Also, each data is composed of `label,text` format.

If there are a total of 2 `labels`, it is expressed as `0` and `1`, and if there are `N`, it should be expressed as `0` to `N-1`.

Korean dataset `nsmc`, `kornli`, English dataset `sst2`, and `sst5` are included by default.

Basically provided dataset can be preprocessed through `utils/$dataset{nsmc, kornli, sst}_preprocess.py`.

Using `nsmc`, a dataset composed of `label,text` can be created in the following way.

```shell script
$ python utils/nsmc_preprocess.py

Number of dataset : 150000
Number of train dataset : 120000
Number of dev dataset : 30000
```

The results can be checked with `train.csv`, `dev.csv`, and `test.csv` in the `data_in` folder.

```shell script
$ head data_in/train.csv

0,아 더빙.. 진짜 짜증나네요 목소리
1,흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나
0,너무재밓었다그래서보는것을추천한다
0,교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정
1,사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다
...
```

Here, `0` represents `Negative` and `1` represents `Positive` label.

`kornli`, `sst2`, and `sst5` can also preprocess data in the same way.

In the case of `kornli`, two sentences are combined and preprocessed using the `[SEP]` token.

```text
1,오두막집 문을 부수고 땅바닥에 쓰러졌어 [SEP] 나는 문을 박차고 들어가 쓰러졌다.
0,어른과 아이들을 위한 재미. [SEP] 외동아이들을 위한 재미.
2,"그래, 넌 학생이 맞아 [SEP] 넌 기계공 학생이지?"
```

For 'sst2' and 'sst5', the 'sst_preprocess.py' file is used to operate in the following manner.

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

**Data that is not provided by default should be created in the form of `label, text` and exist as `train.csv`, `dev.csv`, and `test.csv` in the `data_in` folder.**

## Model

Most of the models supported by [Huggingface models](https://huggingface.co/models) are supported.

However, if the model does not support `AutoModelForSequenceClassification` on huggingface transformers model, this repository's models are not supported.

[Huggingface models](https://huggingface.co/models) Supported models can be used through the `model_name_or_path` argument in the same way as Huggingface transformers.

There are models that do not use the `model_name_or_path` argument and can be loaded more easily by using the `model` argument.

The models are as follows.


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

For how to use the `model` argument, refer to the [shell-script](sh/nsmc/run_bert_base_multilingual_cased_nsmc.sh#L10) file.

**Text classification task does not require `token_type_embedding` and is not supported by many models, so `token_type_embedding` is not supported by our models.**

Even if there is no `token_type_embedding`, I was able to get some [Performance](#Result) even when the text was classified by concatenating two sentences using the `[SEP]` token.

When reusing and testing the trained model, you must specify the folder where the actual model and files are stored in `model_name_or_path`.

**For kobert and distilkobert, make sure to include `kobert` in the folder name when loading the trained model**

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

**One of `model` and `model_name_or_path` must be entered.**
**Need to adjust `num_labels` according to the number of labels in the dataset.**

Argument description can be found in [Huggingface transformers doc](https://huggingface.co/transformers/main_classes/trainer.html?highlight=arguments#trainingarguments) or in the following way.
```shell script
python main.py -h
```
Also, you can check the example through the [shell-script](sh) files provided by default.

**Execution examples can be found in [google colab example](https://colab.research.google.com/drive/1_54nFGE-t0rJYt-kgkbkbluWN74dC8jr?usp=sharing).**



## Result

The result of this experiment was tested using [shell-script](sh), and Hyper-parameter tuning was not performed.

Better performance can be obtained through various hyper-parameter tuning, and this performance is for reference only.

**In KoNLI, we used a method of linking two sentences with a token of `[SEP]` instead of using `token_type_embedding`.**


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
