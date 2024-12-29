import os
import pandas as pd
import re
import sys
import typer
import torch
import logging
from datetime import datetime
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
from mthesis.confusion import Confusion
from mthesis.dataloader import (
    MOFDataset,
    LabeledMOFDataset,
    InstructionMOFDataset,
    LabeledMOFDatasetTokens,
)

# Get log level from environment variable, default to INFO
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
numeric_level = getattr(logging, log_level, None)
if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {log_level}')

# Configure logging with just basic settings - no handlers yet
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=numeric_level,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler()  # Only console output in basicConfig
    ]
)

# Create custom log levels
WRONG_TEMP = 25 
WRONG_TIME = 26
WRONG_SOLVENT = 27
WRONG_ADDITIVE = 28
UNRESOLVABLE_CHEMICAL = 29

# Create logs directory if it doesn't exist
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("logs", timestamp)
os.makedirs(log_dir, exist_ok=True)

# Add level names
for level_name, level_num in [
    ('WRONG_TEMP', WRONG_TEMP),
    ('WRONG_TIME', WRONG_TIME),
    ('WRONG_SOLVENT', WRONG_SOLVENT),
    ('WRONG_ADDITIVE', WRONG_ADDITIVE),
    ('UNRESOLVABLE_CHEMICAL', UNRESOLVABLE_CHEMICAL)
]:
    logging.addLevelName(level_num, level_name)

# Extend the logger class with custom methods
class CustomLogger(logging.Logger):
    def wrong_temperature(self, msg, *args, **kwargs):
        if self.isEnabledFor(WRONG_TEMP):
            self._log(WRONG_TEMP, msg, args, **kwargs)
            
    def wrong_solvent(self, msg, *args, **kwargs):
        if self.isEnabledFor(WRONG_SOLVENT):
            self._log(WRONG_SOLVENT, msg, args, **kwargs)
            
    def wrong_time(self, msg, *args, **kwargs):
        if self.isEnabledFor(WRONG_TIME):
            self._log(WRONG_TIME, msg, args, **kwargs)
            
    def wrong_additive(self, msg, *args, **kwargs):
        if self.isEnabledFor(WRONG_ADDITIVE):
            self._log(WRONG_ADDITIVE, msg, args, **kwargs)

    def unresolvable_chemical(self, msg, *args, **kwargs):
        if self.isEnabledFor(UNRESOLVABLE_CHEMICAL):
            self._log(UNRESOLVABLE_CHEMICAL, msg, args, **kwargs)

class GeneralFilter(logging.Filter):
    def filter(self, record):
        # Get the numeric level
        level_num = record.levelno
        
        # Allow through all standard logging levels
        if level_num in [logging.DEBUG, logging.INFO, logging.WARNING, 
                        logging.ERROR, logging.CRITICAL]:
            return True
            
        # Block our custom levels
        return level_num not in [WRONG_TEMP, WRONG_TIME, WRONG_SOLVENT, 
                               WRONG_ADDITIVE, UNRESOLVABLE_CHEMICAL]

# Create filter classes for each custom level
class WrongTempFilter(logging.Filter):
    def filter(self, record):
        return record.levelname == 'WRONG_TEMP'

class WrongTimeFilter(logging.Filter):
    def filter(self, record):
        return record.levelname == 'WRONG_TIME'

class WrongSolventFilter(logging.Filter):
    def filter(self, record):
        return record.levelname == 'WRONG_SOLVENT'

class WrongAdditiveFilter(logging.Filter):
    def filter(self, record):
        return record.levelname == 'WRONG_ADDITIVE'

class UnresolvableChemicalFilter(logging.Filter):
    def filter(self, record):
        return record.levelname == 'UNRESOLVABLE_CHEMICAL'

# Set the custom logger class
logging.setLoggerClass(CustomLogger)

# Get the root logger
root_logger = logging.getLogger()

# Remove any existing handlers (including those from basicConfig)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create our module's logger
log = logging.getLogger(__name__)

# Create and configure handlers for each type
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))

general_handler = logging.FileHandler(os.path.join(log_dir, 'general.log'))
general_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
general_handler.addFilter(GeneralFilter())

temp_handler = logging.FileHandler(os.path.join(log_dir, 'temperature_errors.log'))
temp_handler.addFilter(WrongTempFilter())
temp_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

time_handler = logging.FileHandler(os.path.join(log_dir, 'time_errors.log'))
time_handler.addFilter(WrongTimeFilter())
time_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

solvent_handler = logging.FileHandler(os.path.join(log_dir, 'solvent_errors.log'))
solvent_handler.addFilter(WrongSolventFilter())
solvent_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

additive_handler = logging.FileHandler(os.path.join(log_dir, 'additive_errors.log'))
additive_handler.addFilter(WrongAdditiveFilter())
additive_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

chemical_handler = logging.FileHandler(os.path.join(log_dir, 'unresolvable_chemical_errors.log'))
chemical_handler.addFilter(UnresolvableChemicalFilter())
chemical_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

# Add all handlers to root logger
root_logger.addHandler(console_handler)
root_logger.addHandler(general_handler)
root_logger.addHandler(temp_handler)
root_logger.addHandler(time_handler)
root_logger.addHandler(solvent_handler)
root_logger.addHandler(additive_handler)
root_logger.addHandler(chemical_handler)

app = typer.Typer()

# late import because of logging setup
from mthesis import conversion
from mthesis.conversion import ans2tempcelsius, ans2hours, txt2cid

conversion.setup_logger(log)

@app.command()
def evaluate(
    settings: Annotated[
        str,
        typer.Option(help="Path to `settings.yml` file, used to read most configuration"),
    ] = "settings.yml",
    stats_path: Annotated[
        str,
        typer.Option(help="Path to `stats.yml` file, used to save progress and results"),
    ] = "stats.yml",
    device: Annotated[
        str,
        typer.Option(help="Manually specify device if torch autodetection is not working."),
    ] = None,
    only_model: Annotated[
        str,
        typer.Option(help="Specify only one specific model to evaluate, skip all others. Requires precise name."),
    ] = None,
    prompt: Annotated[
        str,
        typer.Option(help="Prompt with general task information for the model."),
    ] = "",
    temperature: Annotated[
        float,
        typer.Option(help="Temperature used to query the model."),
    ] = 0.1,
    description: Annotated[
        str,
        typer.Option(help="Description of task to be saved in log folder for later reference."),
    ] = "",
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save description to log folder
    if description:
        with open(os.path.join(log_dir, "description.txt"), "w") as f:
            f.write(description)

    log.info("Loading settings and stats")

    log.info(f"Using prompt: {prompt}, temperature: {temperature}")

    settings = load_yaml(settings)
    log.debug(f"Settings loaded. CSV path: {settings.get('csv_path')}")
    
    try:
        stats = load_yaml(stats_path)
        if stats is None:
            stats = []
        elif isinstance(stats, dict):
            stats = []  # Reset if it's an empty dict
        # If it's already a list, keep it as is
    except Exception as e:
        log.error(f"Error loading stats from {stats_path}: {e}")
        stats = []

    # Load the CSV file containing labels
    log.info(f"Loading labels from CSV file: {settings.get('csv_path')}")  # Debug print
    try:
        labels_df = pd.read_csv(settings['csv_path'], sep=';')  # Add sep=';' parameter
        valid_paragraph_ids = set(labels_df['filename'].values)
        log.info(f"Found {len(valid_paragraph_ids)} paragraphs with labels")  # Debug print
        log.debug(f"First few valid IDs: {list(valid_paragraph_ids)[:5]}")  # Debug print
    except Exception as e:
        log.error(f"Error loading CSV file: {e}")
        return

    evaluated = frozenset(map(lambda s: (s["paragraph_id"], s["model_name"]), stats))
    log.debug(f"Already evaluated: {len(evaluated)} items")  # Debug print

    log.info(f"Loading Dataset from {settings['dataset_path']}")  # Debug print
    dataset = MOFDataset(settings["dataset_path"])
    log.info(f"Dataset loaded with {len(dataset)} items")  # Debug print
    
    # Debug: Check overlap between dataset and valid IDs
    dataset_ids = set(item["paragraph_id"] for item in dataset)
    overlap = dataset_ids.intersection(valid_paragraph_ids)
    log.info(f"Found {len(overlap)} paragraphs that have labels")  # Debug print
    log.debug(f"First few overlapping IDs: {list(overlap)[:5]}")  # Debug print

    for model_settings in settings["models"]:
        model_path = model_settings["model_path"]
        model_name = model_settings["model_name"]
        log.info(f"Processing model: {model_name}")  # Debug print
        
        if only_model and only_model != model_name:
            log.info(f"Skipping model [{model_name}]")  # Debug print
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

            # Skip if no label exists for this paragraph
            if paragraph_id not in valid_paragraph_ids:
                log.debug(f"Skipping {paragraph_id}, no label found in CSV")
                progress_bar.update()
                continue

            if (paragraph_id, model_name) in evaluated:
                log.debug(f"Skipping {paragraph_id}, as it has been processed before.")
                progress_bar.update()
                continue

            if first:
                progress_bar.update(diff)
                first = False
                log.info(f"Loading Model [{model_name}]")
                model = JsonformerModel(prompt=prompt, temperature=temperature, **model_settings)
                model.eval()  # set model to eval mode

            count += 1

            entry = {
                "paragraph_id": paragraph_id,
                "model_name": model_name,
            }

            entry["answer"] = model(item["text"])  # forward the dataset text
            stats.append(entry)
            if count >= 20:
                try:
                    log.info(f"Saving progress to `{stats_path}`")
                    save_yaml(stats, stats_path)
                    save_yaml(stats, "backup/" + stats_path)
                    count = 0
                except:
                    log.warning("Failed to backup results. Continuing...")   
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
    multiple_chemicals_in_answers: Annotated[
        bool,
        typer.Option(
            help="Allow for solvent/additive answers to contains multiple chemicals.",
        ),
    ] = False,
    description: Annotated[
        str,
        typer.Option(help="Description of task to be saved in log folder for later reference."),
    ] = "",
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
    # Save description to log folder
    if description:
        with open(os.path.join(log_dir, "description.txt"), "w") as f:
            f.write(description)
    log.info("analysing!")
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
        try:
            pid = evaluation["paragraph_id"]
            model_name = evaluation["model_name"]
            try:
                item = ds[pid]
            except KeyError:
                log.error(f"KeyError for pid: {pid}")
                continue
            text, label = item["text"], item["label"]
            answer = evaluation["answer"]
            
            # time or temp: convert units, unify number type (float/int)
            a_temp = ans2tempcelsius(a_full := f'{answer["temperature"]} {answer["temperature_unit"]}')
            l_temp = ans2tempcelsius(l_full := f'{label["temperature"]} {label["temperature_unit"]}')
            if a_temp != l_temp:
                if ans2tempcelsius(f'{answer["temperature"]} C') == l_temp:
                    confusion.wrong_unit(model_name, "temperature")
                    log.wrong_temperature(
                    f"\nParagraph: {pid}\n"
                    f"Model: {model_name}\n"
                    f"Expected: {l_temp}°C ({l_full})\n"
                    f"Got: {a_temp}°C ({a_full}) - Wrong unit\n"
                    f"Original text: {text[:200]}..."
                    )
                else:
                    log.wrong_temperature(
                        f"\nParagraph: {pid}\n"
                        f"Model: {model_name}\n"
                        f"Expected: {l_temp}°C ({l_full})\n"
                        f"Got: {a_temp}°C ({a_full})\n"
                        f"Original text: {text[:200]}..."
                    )
                confusion.wrong(model_name, "temperature")
            else:
                confusion.correct(model_name, "temperature")

            a_time = ans2hours(a_full := f'{answer["time"]} {answer["time_unit"]}')
            l_time = ans2hours(l_full := f'{label["time"]} {label["time_unit"]}')
            if a_time != l_time:
                wrong_unit = False
                unit_tests = [
                    ('d', 'days'), 
                    ('s', 'seconds'), 
                    ('h', 'hours')
                ]
                
                for unit_short, unit_name in unit_tests:
                    if ans2hours(f'{answer["time"]} {unit_short}') == l_time:
                        wrong_unit = True
                        confusion.wrong_unit(model_name, "time")
                        log.wrong_time(
                            f"\nParagraph: {pid}\n"
                            f"Model: {model_name}\n"
                            f"Expected: {l_time}h ({l_full})\n"
                            f"Got: {a_time}h ({a_full}) - Wrong unit ({unit_name})\n"
                            f"Original text: {text[:200]}..."
                        )
                        break
                
                if not wrong_unit:
                    log.wrong_time(
                        f"\nParagraph: {pid}\n"
                        f"Model: {model_name}\n"
                        f"Expected: {l_time}h ({l_full})\n"
                        f"Got: {a_time}h ({a_full})\n"
                        f"Original text: {text[:200]}..."
                    )
                confusion.wrong(model_name, "time")
            else:
                confusion.correct(model_name, "time")

            # For additives
            if "additive" not in label or not isinstance(label["additive"], list):
                raw_additive = answer["additive"].strip()
                if raw_additive.lower() in ["none", "no additive", ""]:
                    confusion.correct(model_name, "additive")
                else:
                    log.wrong_additive(
                        f"\nParagraph: {pid}\n"
                        f"Model: {model_name}\n"
                        f"Expected: none\n"
                        f"Got: {raw_additive}\n"
                        f"Original text: {text[:200]}..."
                    )
                    confusion.wrong(model_name, "additive")
                return

            raw_additive = answer["additive"].strip()
            if not raw_additive:
                confusion.wrong(model_name, "additive")
                log.wrong_additive(
                    f"\nParagraph: {pid}\n"
                    f"Model: {model_name}\n"
                    f"Expected one of: {label['additive']}\n"
                    f"Got empty answer\n"
                    f"Original text: {text[:200]}..."
                )
                return

            if multiple_chemicals_in_answers:
                delimiters = [',', '/', ' and ', ':', ';', '&']
                pattern = '|'.join(map(re.escape, delimiters))
                parts = [p.strip() for p in re.split(pattern, raw_additive) if p.strip()]
                
                if not parts:
                    confusion.wrong(model_name, "additive")
                    return
                    
                if any(p.lower() in [syn.lower() for syn in label["additive"]] for p in parts):
                    confusion.correct(model_name, "additive")
                elif all(txt2cid(p) == [] for p in parts):
                    confusion.resolve_answer(model_name, "additive")
                    confusion.wrong(model_name, "additive")
                    log.wrong_additive(
                        f"\nParagraph: {pid}\n"
                        f"Model: {model_name}\n"
                        f"Expected one of: {label['additive']}\n"
                        f"Got unresolvable: {raw_additive}\n"
                        f"Original text: {text[:200]}..."
                    )
                else:
                    confusion.wrong(model_name, "additive")
                    log.wrong_additive(
                        f"\nParagraph: {pid}\n"
                        f"Model: {model_name}\n"
                        f"Expected one of: {label['additive']}\n"
                        f"Got: {raw_additive}\n"
                        f"Original text: {text[:200]}..."
                    )
            else:
                if raw_additive.lower() in [syn.lower() for syn in label["additive"]]:
                    confusion.correct(model_name, "additive")
                elif txt2cid(raw_additive) == []:
                    confusion.resolve_answer(model_name, "additive")
                    confusion.wrong(model_name, "additive")
                    log.wrong_additive(
                        f"\nParagraph: {pid}\n"
                        f"Model: {model_name}\n"
                        f"Expected one of: {label['additive']}\n"
                        f"Got unresolvable: {raw_additive}\n"
                        f"Original text: {text[:200]}..."
                    )
                else:
                    confusion.wrong(model_name, "additive")
                    log.wrong_additive(
                        f"\nParagraph: {pid}\n"
                        f"Model: {model_name}\n"
                        f"Expected one of: {label['additive']}\n"
                        f"Got: {raw_additive}\n"
                        f"Original text: {text[:200]}..."
                    )

            # For solvents (similar logic) but raise error if no solvent in labels
            if "solvent" not in label or not isinstance(label["solvent"], list):
                log.error(f"Missing required solvent data for paragraph {pid}")
                raise ValueError(f"Paragraph {pid} is missing required solvent data")
            
            raw_solvent = answer["solvent"].strip()
            if not raw_solvent:
                confusion.wrong(model_name, "solvent")
                log.wrong_solvent(
                    f"\nParagraph: {pid}\n"
                    f"Model: {model_name}\n"
                    f"Expected one of: {label['solvent']}\n"
                    f"Got empty answer\n"
                    f"Original text: {text[:200]}..."
                )
                return

            if multiple_chemicals_in_answers:
                delimiters = [',', '/', ' and ', ':', ';', '&']
                pattern = '|'.join(map(re.escape, delimiters))
                parts = [p.strip() for p in re.split(pattern, raw_solvent) if p.strip()]
                
                if not parts:
                    confusion.wrong(model_name, "solvent")
                    return
                    
                if any(p.lower() in [syn.lower() for syn in label["solvent"]] for p in parts):
                    confusion.correct(model_name, "solvent")
                elif all(txt2cid(p) == [] for p in parts):
                    confusion.resolve_answer(model_name, "solvent")
                    confusion.wrong(model_name, "solvent")
                    log.wrong_solvent(
                        f"\nParagraph: {pid}\n"
                        f"Model: {model_name}\n"
                        f"Expected one of: {label['solvent']}\n"
                        f"Got unresolvable: {raw_solvent}\n"
                        f"Original text: {text[:200]}..."
                    )
                else:
                    confusion.wrong(model_name, "solvent")
                    log.wrong_solvent(
                        f"\nParagraph: {pid}\n"
                        f"Model: {model_name}\n"
                        f"Expected one of: {label['solvent']}\n"
                        f"Got: {raw_solvent}\n"
                        f"Original text: {text[:200]}..."
                    )
            else:
                if raw_solvent.lower() in [syn.lower() for syn in label["solvent"]]:
                    confusion.correct(model_name, "solvent")
                elif txt2cid(raw_solvent) == []:
                    confusion.resolve_answer(model_name, "solvent")
                    confusion.wrong(model_name, "solvent")
                    log.wrong_solvent(
                        f"\nParagraph: {pid}\n"
                        f"Model: {model_name}\n"
                        f"Expected one of: {label['solvent']}\n"
                        f"Got unresolvable: {raw_solvent}\n"
                        f"Original text: {text[:200]}..."
                    )
                else:
                    confusion.wrong(model_name, "solvent")
                    log.wrong_solvent(
                        f"\nParagraph: {pid}\n"
                        f"Model: {model_name}\n"
                        f"Expected one of: {label['solvent']}\n"
                        f"Got: {raw_solvent}\n"
                        f"Original text: {text[:200]}..."
                    )
        except Exception as e:
            log.error(f"Error processing evaluation for {evaluation['paragraph_id']}, {evaluation['model_name']}: {str(e)}")
            continue
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
