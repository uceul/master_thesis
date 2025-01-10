
    ## Dataset
    print("Loading dataset...")
    dataset = ConceptDataset(pd.read_csv(DATA_PATH))

    dataset_size = len(dataset)
    train_size = int(size_train_dataset * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    ## Setup LORA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # attention heads
        lora_alpha=32,  # alpha scaling
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        inference_mode=False,
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    ## Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_PATH,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        gradient_checkpointing=True,
        logging_dir="./logs",
        logging_steps=1,
        logging_strategy="epoch",
        optim="adamw_torch",
        learning_rate=lr,
        evaluation_strategy="epoch" if size_train_dataset < 1 else "no",
        fp16=True,
        save_strategy="steps",
        save_steps=400,
    )

    ## Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Training...")
    model.config.use_cache = False
    model.train() # put model in train mode
    trainer.train() # actually do the training

    model.save_pretrained(OUTPUT_MODEL_PATH)
