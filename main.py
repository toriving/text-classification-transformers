from transformers import AutoConfig, AutoModelForSequenceClassification, Trainer, HfArgumentParser, set_seed
from modeling import MODEL, AutoTokenizer
from datasets import ClassificationDataset
from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from trainer import metrics_fn
from utils.utils import set_logger, path_checker
from runner import Runner


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    path_checker(training_args)
    set_logger(training_args)

    # Get model name
    model_name = model_args.model_name_or_path \
        if model_args.model_name_or_path is not None \
        else MODEL[model_args.model.lower()] \
        if model_args.model.lower() in MODEL \
        else model_args.model

    # Set seed
    set_seed(training_args.seed)

    # Set model
    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_name, cache_dir=model_args.cache_dir, num_labels=data_args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_name, cache_dir=model_args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, cache_dir=model_args.cache_dir)

    # Set dataset
    train = ClassificationDataset(data_args.data_dir, tokenizer, data_args.task_name, data_args.max_seq_length,
                                  data_args.overwrite_cache, "train") if training_args.do_train else None
    dev = ClassificationDataset(data_args.data_dir, tokenizer, data_args.task_name, data_args.max_seq_length,
                                data_args.overwrite_cache, "dev") if training_args.do_eval else None
    test = ClassificationDataset(data_args.data_dir, tokenizer, data_args.task_name, data_args.max_seq_length,
                                 data_args.overwrite_cache, "test") if training_args.do_predict else None

    # Set trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        compute_metrics=metrics_fn,
    )

    runner = Runner(model_name, trainer, tokenizer, training_args, test)
    runner()


if __name__ == "__main__":
    main()
