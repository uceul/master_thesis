import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

import os
import csv
import json
import logging

from mthesis.utils import read_paragraph
from mthesis.conversion import cid2syns

log = logging.getLogger(__name__)


class ConceptDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = (
            ">>ABSTRACT<<"  # = Token_id of 1
            + row["abstract"]
            + ">>SUMMARY<<"  # = Token_id of 3
            + str(row["tags"].split(","))
            + ">>INTRODUCTION<<"  # = Token_id of 2
        )

        # Tokenize text and tags separately
        text_encodings = tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer_max_length,
        )

        return {
            "input_ids": text_encodings["input_ids"].flatten().to(device),
            "attention_mask": text_encodings["attention_mask"].flatten().to(device),
            "labels": text_encodings["input_ids"].flatten().to(device),
        }


class MOFDataset(torch.utils.data.Dataset):
    """Dataset of MOF synthesis paragraphs, without labels."""

    def __init__(self, dataset_path: str):
        log.info(f"Loading MOFDataset from '{dataset_path}'")
        self.paragraph_list = os.listdir(dataset_path)
        self.paragraphs = dict()
        for p_id in self.paragraph_list:
            try:
                self.paragraphs[p_id] = read_paragraph(p_id, dataset_path)
            except Exception as e:
                log.warning(e)
                continue

        self.paragraphs = OrderedDict(self.paragraphs)
        self.paragraph_list = list(self.paragraphs.items())

    def __len__(self):
        # return length of full dataset
        return len(self.paragraphs)

    def __getitem__(self, idx: int | str) -> dict:
        # return item at position idx.
        paragraph = ""
        try:
            paragraph = self.paragraphs[idx]
        except KeyError:
            idx, paragraph = self.paragraph_list[idx]

        # since Jsonformer is doing tokenization later on,
        # we're not doing that yet
        return {
            "paragraph_id": idx,
            "text": paragraph,
        }


class LabeledMOFDataset(MOFDataset):
    """Dataset of MOF synthesis paragraphs, with labels."""

    def __init__(self, dataset_path: str, label_cols: dict, from_csv: str = None):
        """
        dataset_path: path from which to load synthesis paragraphs
        label_cols: mapping from parameters to columns in label files (where the label for the parameter can be read from).
        from_csv: load dataset from csv filepath instead (if path is provided)
        """
        self.labels = dict()
        self.label_cols = label_cols
        if from_csv:
            try:
                self.paragraphs = dict()
                self.paragraph_list = list()
                self._from_csv(from_csv)
                return
            except Exception:
                log.error(
                    f"Failed loading dataset from CSV '{from_csv}'. Will reconstruct instead."
                )

        # load paragraphs first
        super().__init__(dataset_path)

        # now load label data
        labels_path_A = os.path.join(dataset_path, "../results", "SynMOF_A_out.csv")
        labels_path_M = os.path.join(dataset_path, "../results", "SynMOF_M_out.csv")

        labels_A = pd.read_csv(labels_path_A, sep=";")
        labels_M = pd.read_csv(labels_path_M, sep=";")

        progress_bar = tqdm(self.paragraphs.keys(), file=open(os.devnull, "w"))
        progress_bar.set_postfix_str("Configuring Dataset Labels")

        # count = 5

        for paragraph_id in progress_bar:
            # if count <= 0:
            #     self.paragraphs = { k: self.paragraphs[k] for k in self.labels.keys() }
            #     break
            # count -= 1
            print(str(progress_bar))
            self.labels[paragraph_id] = dict()
            for parameter, column in label_cols.items():
                answer_a, answer_m = None, None
                try:
                    answer_a = labels_A.loc[labels_A["filename"] == paragraph_id][
                        column
                    ].values[0]
                    answer_m = (
                        None
                        if paragraph_id not in labels_M["filename"].values
                        else labels_M.loc[labels_M["filename"] == paragraph_id][
                            column
                        ].values[0]
                    )
                except KeyError as e:
                    log.critical(
                        f"KeyError: Invalid SETTINGSFILE. Requested column `dataset_cols` '{column}' for parameter {parameter} does not exist in file. Failed when loading labels."
                    )
                    sys.exit(1)
                answer_a = None if np.isnan(answer_a) else int(answer_a)
                if answer_m is not None:
                    answer_m = None if np.isnan(answer_m) else int(answer_m)

                if answer_a and answer_m and answer_a != answer_m:
                    log.warning(
                        f"Dataset: answer cids differ. a: {answer_a} m: {answer_m}. Continuing with answer [m]."
                    )
                # if answer_a is None:
                #     self.labels[paragraph_id][parameter] = None
                #     continue

                if answer_m is None:
                    self.labels[paragraph_id][parameter] = None
                    continue

                if parameter in ["additive", "solvent"]:
                    synonyms = cid2syns(answer_m)
                    found = False

                    self.labels[paragraph_id][parameter + "_cid"] = answer_m

                    for syn in synonyms:
                        if syn.lower() in self.paragraphs[paragraph_id].lower():
                            self.labels[paragraph_id][parameter] = syn
                            found = True
                            break
                    if not found:
                        log.error(f"Didn't find {parameter} synonym for {synonyms[0]} in text. Selecting it as default. Available: {len(synonyms)}")
                        self.labels[paragraph_id][parameter] = synonyms[0]
                elif parameter in ["time", "temperature"]:
                    self.labels[paragraph_id][parameter] = answer_m
            self.labels[paragraph_id]["temperature_unit"] = "C"
            self.labels[paragraph_id]["time_unit"] = "h"

        # resulting output from Jsonformer:
        # {'additive': 'water', 'solvent': 'water', 'temperature': 90.0, 'temperature_unit': 'C', 'time': 40.0, 'time_unit': 'h'}

    def to_csv(self, filename):
        """
        Header:
        paragraph_id, <parameters>, context
        idx, <labels>, <context>
        """
        keys = ["paragraph_id"] + list(self.label_cols.keys()) + ["context", "temperature_unit", "time_unit", "additive_cid", "solvent_cid"]
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, delimiter=";")

            # write header
            writer.writeheader()

            for idx in self.paragraphs.keys():
                writer.writerow(
                    {"paragraph_id": idx, "context": self.paragraphs[idx]}
                    | {p: v for p, v in self.labels[idx].items()}
                )
        log.info(f"[dataset] Saved to [{filename}]")

    def _from_csv(self, from_csv: str):
        log.info(f"[dataset]: Loading LabeledMofDataset from csv '{from_csv}'")
        # need to reconstruct:
        # - self.paragraphs: dict[paragraph_id -> context]
        # - self.paragraph_list: list[(paragraph_id, context)]
        # - self.labels: dict[paragraph_id, parameter -> str | float | None]
        with open(from_csv, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                idx = row["paragraph_id"]
                self.paragraphs[idx] = row["context"]
                self.labels[idx] = {k: row[k] for k in self.label_cols.keys()}
                if not self.labels[idx].get("temperature_unit"):
                    self.labels[idx]["temperature_unit"] = "C"
                if not self.labels[idx].get("time_unit"):
                    self.labels[idx]["time_unit"] = "h"
        self.paragraph_list = list(self.paragraphs.items())
        log.info(f"[dataset]: Finished loading from csv. #Items: {len(self)}")

    def __getitem__(self, idx: int | str) -> dict:
        # return item at position idx.
        paragraph = ""
        label = ""
        try:
            paragraph = self.paragraphs[idx]
        except KeyError:
            try:
                idx, paragraph = self.paragraph_list[idx]
            except TypeError:
                raise KeyError

        label = self.labels[idx]
        # {'additive': 'H2O', 'solvent': 'H2O', 'temperature': 150.0, 'temperature_unit': 'C', 'time': 96.0, 'time_unit': 'h'}

        # since Jsonformer is doing tokenization later on,
        # we're not doing that yet
        return {
            "paragraph_id": idx,
            "text": paragraph,
            "label": label,
        }


class LabeledMOFDatasetTokens(LabeledMOFDataset):
    def __init__(self, tokenizer, load_untokenized: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs["from_csv"] and load_untokenized:
            return

        self.tokenized = {
            k: tokenizer.encode(self.paragraphs[k] + "\n" + str(self.labels[k]))
            for k in self.paragraphs.keys()
        }

    def __getitem__(self, idx: int | str) -> dict:
        try:
            return self.tokenized[idx]
        except KeyError:
            idx, _ = self.paragraph_list[idx]
            return self.tokenized[idx]


class InstructionMOFDataset(LabeledMOFDataset):
    """Basically, a LabeledMOFDataset, but we're returning instructions here."""

    def __init__(self, *args, **kwargs):
        log.info(str(args))
        log.info(str(kwargs))
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int | str) -> dict:
        paragraph = ""
        label = ""
        try:
            paragraph = self.paragraphs[idx]
        except KeyError:
            idx, paragraph = self.paragraph_list[idx]

        label = self.labels[idx]
        label_text = f"### Answer: \n {label}"

        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "additive": {"type": "string"},
                    "solvent": {"type": "string"},
                    "temperature": {"type": "number"},
                    "temperature_unit": {"type": "string"},
                    "time": {"type": "number"},
                    "time_unit": {"type": "string"},
                },
            }
        )

        instruction = f"### Task: Extract the required synthesis parameters from the paragraph above in the following JSON schema format:\n{schema}"
        # "instruction": "Detect the sentiment of the tweet.",
        # "input": row_dict["tweet"],
        # "output": sentiment_score_to_name(row_dict["sentiment"])
        combined = f"{paragraph}\n\n{instruction}\n\n{label_text}"

        # use wrapper TokenizeDataSet
        return {
            "paragraph_id": idx,
            "instruction": instruction,
            "input": paragraph,
            "output": label,
            "text": combined,
        }


class TokenizeField(torch.utils.data.Dataset):
    """Tokenize a specified field upon request."""

    def __init__(self, dataset: torch.utils.data.Dataset, tokenizer, field: str):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.field = field

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int | str) -> dict:
        item = self.dataset[idx]
        enc = self.tokenizer(
            item[self.field],
            add_special_tokens=True,
            return_tensors="pt",
            # padding=True,
            # truncation=True,
            # max_length=tokenizer_max_length,
        )
        item["input_ids"] = enc["input_ids"]
        item["attention_mask"] = enc["attention_mask"]
        return item


class MOFDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        # download, tokenize, save data to disk.
        # called from main process only. local (object) variables won't be available
        # when doing distributed training. Last step should always be to save to disk.
        pass

    def setup(self, stage: str):
        # how to split, define dataset, etc
        # `stage` is one of "fit" / "train"
        # first step is to load from disk. this is executed for each
        # distributed shard.
        pass

    def predict_dataloader(self):
        pass
