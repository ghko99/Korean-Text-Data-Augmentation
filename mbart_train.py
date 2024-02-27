from transformers import (
    AutoModelForSeq2SeqLM,
    MBart50TokenizerFast,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    AutoConfig
)
import logging
from datasets import load_dataset


logger = logging.getLogger(__name__)

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

summarization_name_mapping = {"commongen": ("concept-set", 'scene')}

def main():
    
    model_args, data_args = ModelArguments(), DataTrainingArguments()
    
    # set training arguments
    training_args = Seq2SeqTrainingArguments(
        do_eval=True,
        do_train=True,
        output_dir = "./model/mbart",
        run_name = "model_for_mbart_augmentation",
        save_steps=10000,
        save_total_limit=1   
    )
    
    # get dataset file
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    
    #get congig, tokenizer, model
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = MBart50TokenizerFast.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    

    max_target_length = data_args.max_target_length
    padding =  False

    #text to input_ids (preprocessing)
    def preprocess_function(examples):
        inputs = examples["concept-set"]
        targets = examples["scene"]
        inputs = [inp for inp in inputs]

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
    
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    #get train dataset
    train_dataset = datasets["train"]
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=['concept-set', 'scene'],
        load_from_cache_file=not data_args.overwrite_cache,
    )

    #get evaluate dataset
    eval_dataset = datasets["validation"]
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=['concept-set', 'scene'],
        load_from_cache_file=not data_args.overwrite_cache,
    )

    #get data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    #get trainer    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )
    
    #mbart train
    train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    #mbart evaluate
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(
        max_length=max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
    )
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":  
    main()