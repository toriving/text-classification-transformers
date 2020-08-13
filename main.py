import logging
import os
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    HfArgumentParser,
    set_seed
)
from modeling import MODEL, AutoTokenizer
from datasets import ClassificationDataset
from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from trainer import metrics_fn, prediction


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.FileHandler(training_args.logging_dir + "/logging.log", 'w', encoding='utf-8'), logging.StreamHandler()]
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Get model name
    model_name = model_args.model_name_or_path \
        if model_args.model_name_or_path is not None \
        else MODEL[model_args.model.lower()] \
        if model_args.model.lower() in MODEL \
        else model_args.model

    # Set seed
    set_seed(training_args.seed)

    # Set model
    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_name,
                                        cache_dir=model_args.cache_dir, num_labels=data_args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_name,
                                              cache_dir=model_args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config,
                                                               cache_dir=model_args.cache_dir)

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

    # Training
    if training_args.do_train:
        trainer.train(model_path=model_name if os.path.isdir(model_name) else None)
        trainer.save_model()
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Validation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        result = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        logger.info("Validation set result : {}".format(result))

    # Test prediction
    if training_args.do_predict:
        logger.info("*** Test ***")
        predictions = trainer.predict(test_dataset=test)
        output_test_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_master():
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results *****")
                logger.info("{}".format(predictions))
                writer.write("prediction : \n{}\n\n".format(prediction(predictions.predictions).tolist()))
                if predictions.label_ids is not None:
                    writer.write("ground truth : \n{}\n\n".format(predictions.label_ids.tolist()))
                    writer.write("metrics : \n{}\n\n".format(predictions.metrics))


if __name__ == "__main__":
    main()
