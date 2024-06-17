import os
import sys
import typer
import torch
import logging
from typing_extensions import Annotated
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline,
)
from pathlib import Path

from mthesis.models import JsonformerModel
from mthesis.utils import load_yaml, save_yaml, count_occurences
from mthesis.conversion import ans2tempcelsius, ans2hours, txt2cid
from mthesis.confusion import Confusion
from mthesis.dataloader import (
    MOFDataset,
    LabeledMOFDataset,
    InstructionMOFDataset,
    LabeledMOFDatasetTokens,
)

logging.basicConfig(
    filename='error/error_more.log',
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.WARNING,
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def evaluate(
    settings: Annotated[
        str,
        typer.Option(
            help="Path to `settings.yml` file, used to read most configuration"
        ),
    ] = "settings.yml",
    stats_path: Annotated[
        str,
        typer.Option(
            help="Path to `stats.yml` file, used to save progress and results"
        ),
    ] = "stats.yml",
    device: Annotated[
        str,
        typer.Option(
            help="Manually specify device if torch autodetection is not working."
        ),
    ] = None,
    only_model: Annotated[
        str,
        typer.Option(
            help="Specify only one specific model to evaluate, skip all others. Requires precise name."
        ),
    ] = None,
):
    """Based on a provided file of SETTINGS, and a path to write STATS,
    evaluate a list of LLMs on provided tasks, and record their results
    in STATS.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Loading settings and stats")

    settings = load_yaml(settings)
    stats = load_yaml(stats_path)
    # run evaluation of models

    evaluated = frozenset(map(lambda s: (s["paragraph_id"], s["model_name"]), stats))

    log.info(f"Loading Dataset")
    dataset = MOFDataset(settings["dataset_path"])

    for model_settings in settings["models"]:
        model_path = model_settings["model_path"]
        model_name = model_settings["model_name"]
        model = None

        if only_model and only_model != model_name:
            log.info(f"Skipping model [{model_name}]")
            continue

        progress_bar = tqdm(dataset, file=open(os.devnull, "w"))
        progress_bar.set_postfix_str(model_name)

        diff = -(len(evaluated) % len(dataset))
        first = True
        count = 0

        for item in progress_bar:
            log.info(str(progress_bar))
            print(str(progress_bar))
            paragraph_id = item["paragraph_id"]
            if (paragraph_id, model_name) in evaluated:
                log.debug(f"Skipping {paragraph_id}, as it has been processed before.")
                progress_bar.update()
                continue

            if first:
                progress_bar.update(diff)
                first = False
                log.info(f"Loading Model [{model_name}]")
                model = JsonformerModel(**model_settings)
                model.eval()  # set model to eval mode

            count += 1

            entry = {
                "paragraph_id": paragraph_id,
                "model_name": model_name,
            }

            entry["answer"] = model(item["text"])  # forward the dataset text
            stats.append(entry)
            if count >= 20:
                log.info(f"Saving progress to `{stats_path}`")
                save_yaml(stats, stats_path)
                save_yaml(stats, "backup/" + stats_path)
                count = 0
        log.info(f"Saving progress to `{stats_path}`")
        save_yaml(stats, stats_path)
        save_yaml(stats, "backup/" + stats_path)


@app.command()
def test(
    settings: Annotated[
        str,
        typer.Option(
            help="Path to `settings.yml` file, used to read most configuration"
        ),
    ] = "settings.yml",
    stats_path: Annotated[
        str,
        typer.Option(
            help="Path to `stats.yml` file, used to save progress and results"
        ),
    ] = "stats.yml",
    device: Annotated[
        str,
        typer.Option(
            help="Manually specify device if torch autodetection is not working."
        ),
    ] = None,
):
    """Test information."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Loading settings and stats")

    settings = load_yaml(settings)
    stats = load_yaml(stats_path)

    for model_settings in settings["models"]:
        model_name = model_settings["model_name"]
        model_path = model_settings["model_path"]
        try:
            log.info(f"Loading Tokenizer [{model_name}]")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            log.info(f"Loading Model [{model_name}] for Training")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
            )
            log.info("Finished loading. Begin Training for one Sample")
            OUTPUT_MODEL_PATH = f"/home/kit/iti/lz5921/checkpoints/{model_name}/"
            Path(OUTPUT_MODEL_PATH).mkdir(parents=True, exist_ok=True)

            # DataCollator:
            # https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling
            # mlm (bool, optional, defaults to True) — Whether or not to use
            # masked language modeling. If set to False, the labels are the
            # same as the inputs with the padding tokens ignored (by setting
            # them to -100). Otherwise, the labels are -100 for non-masked
            # tokens and the value to predict for the masked token.
            data_collator = DataCollatorForLanguageModeling(tokenizer)  # mlm=False

            dataset = TokenizeDataSet(SingleSampleDataset(), tokenizer)

            ## Arguments
            training_args = TrainingArguments(
                output_dir=OUTPUT_MODEL_PATH,
                overwrite_output_dir=True,
                num_train_epochs=1,
                weight_decay=0.005,
                per_device_train_batch_size=1,
                gradient_checkpointing=True,
                logging_dir="./logs",
                logging_steps=1,
                logging_strategy="epoch",
                optim="adamw_torch",
                learning_rate=5e-4,
                evaluation_strategy="epoch",  # alternatively: "no",
                # fp16=True,
                save_strategy="steps",
                save_steps=50,
            )

            # torch.cuda.set_device(0) # -1 for cpu, 0, 1, ... for gpu

            ## Training
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                eval_dataset=dataset,
                data_collator=data_collator,
            )

            # automatically restores model, epoch, step, LR schedulers, etc from checkpoint
            # model.config.use_cache = False
            model.train()  # put model in training mode
            trainer.train()
            model.save_pretrained(OUTPUT_MODEL_PATH)

        except Exception as e:
            log.error(f"{e} during loading of {model_name}.")


@app.command()
def train(
    settings: Annotated[
        str,
        typer.Option(
            help="Path to `settings.yml` file, used to read most configuration"
        ),
    ] = "settings.yml",
    stats_path: Annotated[
        str,
        typer.Option(
            help="Path to `stats.yml` file, used to save progress and results"
        ),
    ] = "stats.yml",
    device: Annotated[
        str,
        typer.Option(
            help="Manually specify device if torch autodetection is not working."
        ),
    ] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Loading settings and stats")

    settings = load_yaml(settings)
    stats = load_yaml(stats_path)

    try:
        label_cols = {
            param: settings["extract_config"][param]["dataset_cols"][0]
            for param in settings["extract_config"]
        }
    except KeyError as e:
        log.critical(
            f"KeyError: Invalid SETTINGSFILE. Missing global key `extract_config` or param-key `dataset_cols`."
        )
        sys.exit(1)

    # TODO: FOR loop
    model_settings = settings["models"][0]

    model_path = model_settings["model_path"]
    model_name = model_settings["model_name"]

    log.info(f"Loading Tokenizer [{model_name}]")
    tokenizer = AutoTokenizer.from_pretrained(model_path, return_tensors="pt")
    tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Loading Dataset '{settings['dataset_path']}'")
    # dataset = InstructionMOFDataset(settings["dataset_path"], label_cols, from_csv="mof_dataset_labeled.csv")
    # dataset = LabeledMOFDatasetTokens(tokenizer, True, settings["dataset_path"], label_cols, from_csv="mof_dataset_labeled.csv")
    # dataset = TokenizeDataSet(InstructionMOFDataset(settings["dataset_path"], label_cols, from_csv="mof_dataset_labeled.csv"), tokenizer)
    dataset = LabeledMOFDatasetTokens(
        tokenizer,
        dataset_path=settings["dataset_path"],
        label_cols=label_cols,
        from_csv="mof_dataset_labeled.csv",
    )

    # dataset.to_csv("mof_dataset_labeled.csv")

    generator = torch.Generator().manual_seed(42)
    train_ds, eval_ds = torch.utils.data.random_split(
        dataset, [0.05, 0.95], generator=generator
    )

    log.info(f"Loading Model [{model_name}]")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    # model = JsonformerHFModel(model_path)

    OUTPUT_MODEL_PATH = f"/home/kit/iti/lz5921/llms/checkpoints/{model_name}/"
    Path(OUTPUT_MODEL_PATH).mkdir(parents=True, exist_ok=True)

    # DataCollator: https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling
    # mlm (bool, optional, defaults to True) — Whether or not to use masked
    # language modeling. If set to False, the labels are the same as the inputs
    # with the padding tokens ignored (by setting them to -100). Otherwise, the
    # labels are -100 for non-masked tokens and the value to predict for the
    # masked token.
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    ## Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_PATH,
        overwrite_output_dir=True,
        num_train_epochs=1,
        weight_decay=0.005,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        logging_dir="./logs",
        logging_steps=1,
        logging_strategy="epoch",
        optim="adamw_torch",
        learning_rate=5e-4,
        evaluation_strategy="epoch",  # alternatively: "no",
        # fp16=True,
        save_strategy="steps",
        save_steps=50,
    )

    # torch.cuda.set_device(0) # -1 for cpu, 0, 1, ... for gpu

    ## Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    log.info("Set up training. Starting ...")

    # automatically restores model, epoch, step, LR schedulers, etc from checkpoint
    model.config.use_cache = False
    model.train()  # put model in training mode
    trainer.train()
    log.info("Concluded training. Saving ...")
    model.save_pretrained(OUTPUT_MODEL_PATH)


@app.command()
def analyse(
    settings: Annotated[
        str,
        typer.Option(
            help="Path to `settings.yml` file, used to read most configuration"
        ),
    ] = "settings.yml",
    stats_path: Annotated[
        str,
        typer.Option(
            help="Path to `stats.yml` file, used to load progress and results"
        ),
    ] = "stats.yml",
    output_file: Annotated[
        str,
        typer.Option(
            help="Path to `stats.csv` file, used to save the analysis of results",
        ),
    ] = "stats.csv",
    count_dataset: Annotated[
        bool,
        typer.Option(
            help="Count various occurences in the dataset, and print the result. Will not modify `stats.csv`",
        ),
    ] = False,
    count_stats: Annotated[
        bool,
        typer.Option(
            help="Count various occurences in STATS, and print the result. Will not modify `stats.csv`",
        ),
    ] = False,
    check_resolving: Annotated[
        bool,
        typer.Option(
            help="After counting stats or dataset, check if elements resolve",
        ),
    ] = False,
    dataset_cache_path: Annotated[
        str,
        typer.Option(
            help="Path to `mof_dataset_labeled.csv` file, used to cache dataset intermediaries.",
        ),
    ] = "mof_dataset_labeled_M.csv",
    reload_dataset: Annotated[
        bool,
        typer.Option(
            help="Fully re-calculate the dataset cache intermediaries.",
        ),
    ] = False,
):
    """Analyse the results of previuos evaluation runs."""
    # TODO:
    # - [x] read in settings -> dataset path
    # - [x] read in stats
    # - [x] read dataset (with labels)
    # - [x] iterate through stats values (fetch dataset entries)
    # - [x] figure out accuracy of entries (probably done?)
    # - [ ] build confusion matrix for each parameter, for each model
    # - [ ] for each model, for each entry, compare label with predicted, convert units when necessary
    settings = load_yaml(settings)
    stats = load_yaml(stats_path)  # :: [{str -> str}]

    try:
        label_cols = {
            param: settings["extract_config"][param]["dataset_cols"][0]
            for param in settings["extract_config"]
        }
    except KeyError as e:
        log.critical(
            f"KeyError: Invalid SETTINGSFILE. Missing global key `extract_config` or param-key `dataset_cols`."
        )
        sys.exit(1)

    if reload_dataset:
        ds = LabeledMOFDataset(
            settings["dataset_path"],
            label_cols,
        ) 
        ds.to_csv(dataset_cache_path)
        return

    ds = LabeledMOFDataset(
        settings["dataset_path"],
        label_cols,
        from_csv=dataset_cache_path,
    )

    # do some counting, if so requested
    cs, cd = None, None
    if count_stats:
        cs = count_occurences(map(lambda e: e["answer"], stats))
    if count_dataset:
        cd = count_occurences(map(lambda e: e["label"], ds))

    if count_stats or count_dataset:
        # if check_resolving:
        #     if cs:
        #         for k, v in (cs["additive"] | cs["solvent"]).items():
        #             if txt2cid(k):
        #                 ...
        #     if ds:
        #         ...
        return

    # build confusion 'matrix'
    confusion = Confusion()

    for evaluation in stats:
        pid = evaluation["paragraph_id"]
        model_name = evaluation["model_name"]
        try:
            item = ds[pid]
        except KeyError:
            continue
        text, label = item["text"], item["label"]
        answer = evaluation["answer"]

        # time or temp: convert units, unify number type (float/int)
        a_temp = ans2tempcelsius(a_full := f'{answer["temperature"]} {answer["temperature_unit"]}')
        l_temp = ans2tempcelsius(l_full := f'{label["temperature"]} {label["temperature_unit"]}')
        if a_temp != l_temp:
            if ans2tempcelsius(f'{answer["temperature"]} C') == l_temp:
                confusion.wrong_unit(model_name, "temperature")
            else:
                log.debug(f"temperature [{pid}] {a_temp} != {l_temp} | {a_full} != {l_full}")
            confusion.wrong(model_name, "temperature")
        else:
            confusion.correct(model_name, "temperature")

        a_time = ans2hours(a_full := f'{answer["time"]} {answer["time_unit"]}')
        l_time = ans2hours(l_full := f'{label["time"]} {label["time_unit"]}')
        if a_time != l_time:
            if ans2hours(f'{answer["time"]} d') == l_time:
                confusion.wrong_unit(model_name, "time")
            elif ans2hours(f'{answer["time"]} s') == l_time:
                confusion.wrong_unit(model_name, "time")
            elif ans2hours(f'{answer["time"]} h') == l_time:
                confusion.wrong_unit(model_name, "time")
            else:
                log.debug(f"duration [{pid}] {a_time} != {l_time} | {a_full} != {l_full}")
            confusion.wrong(model_name, "time")
        else:
            confusion.correct(model_name, "time")

        # all models answered _something_, even if there was
        # no additive. So when the label is empty,
        # whatever the model says is just wrong.
        if label["additive"] == "":
            confusion.wrong(model_name, "additive")
        elif txt2cid(answer["additive"]) == []:
            confusion.resolve_answer(model_name, "additive")
            confusion.wrong(model_name, "additive")
        elif set(txt2cid(answer["additive"])).isdisjoint(set(label["additive_cid"])):
            log.debug(f"adddiff [{pid}] {answer['additive']} != {label['additive']}")
            confusion.wrong(model_name, "additive")
        else:
            confusion.correct(model_name, "additive")


        if txt2cid(answer["solvent"]) == []:
            confusion.resolve_answer(model_name, "solvent")
            confusion.wrong(model_name, "solvent")
        elif set(txt2cid(answer["solvent"])).isdisjoint(set(label["solvent_cid"])):
            log.debug(f"soldiff [{pid}] {answer['solvent']} != {label['solvent']}")
            confusion.wrong(model_name, "solvent")
        else:
            confusion.correct(model_name, "solvent")

    # log.info(f"{correct}/{uwrong}/{wrong}/{total}  correct/unit/wrong/total")
    # log.info(f"prop {correct / total:.2f}/{wrong / total:.2f} correct/wrong")
    # log.info(f"prop {uwrong / total:.2f} with unit wrong")

    # goal of table format:
    # paragraph_id, model_name, additive, solvent, temp, tempdelta_C, time, timedelta_h
    # WONRUM02_clean, LLaMa 7B, True    , False  , True, 0          , False, 8
    confusion.print_stats()
    confusion.print_prop_stats()

def main():
    app()


if __name__ == "__main__":
    main()
