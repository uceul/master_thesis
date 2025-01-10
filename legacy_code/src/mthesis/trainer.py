import os

from pl_bolts.callbacks import ORTCallback
from pytorch_lightning import Trainer, seed_everything
from lightning_transformers.plugins.checkpoint import HFSaveCheckpoint
from torch.optim import AdamW

from mthesis.models import JsonformerModel

# seed_everything(42, workers=True)


if settings is None:
    settings = "settings.yml"
if stats is None:
    stats_path = "stats.yml"
else:
    stats_path = stats
if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log.info("Loading settings and stats")

settings = load_yaml(settings)
stats = load_yaml(stats_path)


model_settings = settings["models"][0]

model_path = model_settings["model_path"]
model_name = model_settings["model_name"]

model = JsonformerModel(model_path)
# optimizer = AdamW(get_grouped_params(model), lr=5e-4)
# Epochs: 3-5
# Training examples: 100, 300, 500
trainer = Trainer(
    # devices="auto", accelerator="auto", deterministic=False,  # all default
    # precision="16-mixed",
    callbacks=ORTCallback(),
    max_epochs=1,
    # accumulate_grad_batches=4,
    fast_dev_run=True,
    # plugins=HFSaveCheckpoint(model=model),
)  # set deterministic=True for reproducible results later on

# automatically restores model, epoch, step, LR schedulers, etc from checkpoint
# trainer.fit(model, ckpt_path=f"/home/kit/iti/lz5921/llms/checkpoints/{model_name}.ckpt")
trainer.fit(model, ckpt_path=os.path.join(settings["checkpoint_path"], f"{model_path.split('/')[-1]}/{model_name}.ckpt")


# ----------------------------------------------t
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
