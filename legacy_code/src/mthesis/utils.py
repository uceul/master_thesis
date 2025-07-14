import yaml
import os
import sys
import logging
from rich.pretty import pprint

log = logging.getLogger(__name__)


def md5(fpath: str):
    """Return md5 hash of file at location `fpath`."""
    import hashlib

    hash_md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_yaml(filename: str, root_dir: str = None) -> dict:
    """Load `filename` as yaml-file, and try with both endings `.yml` and
    `.yaml`, regardless of what was passed.
    If the file is not found, it is created (empty), and `None` is returned.
    You need to handle errors when the directory does not exist yourself.

    Params:
        filename: name of the yaml file to load
        root_dir: root-directory of file (default: ./)
    """
    if root_dir is None:
        root_dir = ""
    name = ".".join(filename.split(".")[:-1])
    endings = [".yml", ".yaml"]
    for end in endings:
        try:
            with open(root_dir + name + end, "r") as stream:
                return yaml.safe_load(stream)
        except FileNotFoundError as e:
            log.warn(e)
    log.warn("Could not load '{}', creating it (empty)".format(filename))
    save_yaml({}, root_dir + filename)


def save_yaml(data: dict | list | str, filename: str):
    """Save `data` to `filename` in yaml."""
    with open(filename, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def read_paragraph(paragraph_id: str, dataset_path: str = None):
    """Read file `dataset_path/synthesis_paragraphs/paragraph_id/content.txt`.

    Will encode with utf8, and replace e.g. '\n', '\u2009', '\u2005' and '\xa0'
    with a normal space.
    """
    # TODO: two hardcoded locations: synthesis_paragraphs and content.txt
    if not dataset_path:
        dataset_path = "~/mof_synthesis/synthesis_paragraphs"
    paragraph_file = os.path.join(dataset_path, paragraph_id, "content.txt")
    paragraph_text = ""
    with open(paragraph_file, encoding="utf8") as f:
        paragraph_text = " ".join(f.readlines())
        for char in ["\n", "\u2009", "\u2005", "\xa0"]:
            paragraph_text = paragraph_text.replace(char, " ")
    return paragraph_text


def count_occurences(iterable: iter, counters: dict = None):
    if counters is None:
        counters = {
            "additive": {},
            "solvent": {},
            "temperature": {},
            "time": {},
            "temperature_unit": {},
            "time_unit": {},
        }
    for item in iterable:
        for k, v in item.items():
            counters[k][v] = counters[k].get(v, 0) + 1
    pprint(counters)
    return counters
