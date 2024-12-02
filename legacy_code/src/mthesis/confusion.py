from rich.pretty import pprint
import csv


def confusion(stats, ds):
    """ build a confusion matrix for each parameter, for each model
    """
    confusion = {} # model_name -> parameter -> accurate/unit/wrong

def flatten_empty(d: dict) -> dict:
    """ recursively remove empty or zero-values in a dictionary.
    """
    if not isinstance(d, dict):
        return d
    return {k: flatten_empty(v) for k, v in d.items() if v}

def relative_proportion(d: dict) -> dict:
    def rel(w: dict) -> dict:
        correct = w.get('correct', 0)
        wrong = w.get('wrong', 0)
        total = correct + wrong
        if total == 0:
            return {k: 0 for k in w.keys()}
        return {k: v / total if total > 0 else 0 for k, v in w.items()}

    return {m: {p: rel(v) for p, v in e.items()} for m, e in d.items()}
class Confusion:
    confusion = {
        "Total": {
        }
    }
    def __init__(self):
        ...

    def _ensure_dict(self, model_name, parameter):
        if self.confusion.get(model_name) is None:
            self.confusion[model_name] = {}

        if self.confusion[model_name].get(parameter) is None:
            self.confusion[model_name][parameter] = {
                "correct": 0, "wrong": 0, "unit": 0,
                "resolve_label": 0,
                "resolve_answer": 0,
            }

        if self.confusion["Total"].get(parameter) is None:
            self.confusion["Total"][parameter] = {
                "correct": 0, "wrong": 0, "unit": 0,
                "resolve_label": 0,
                "resolve_answer": 0,
            }

    def wrong_unit(self, model_name, parameter):
        self._ensure_dict(model_name, parameter)
        self.confusion[model_name][parameter]["unit"] += 1
        self.confusion["Total"][parameter]["unit"] += 1

    def wrong(self, model_name, parameter):
        self._ensure_dict(model_name, parameter)
        self.confusion[model_name][parameter]["wrong"] += 1
        self.confusion["Total"][parameter]["wrong"] += 1

    def correct(self, model_name, parameter):
        self._ensure_dict(model_name, parameter)
        self.confusion[model_name][parameter]["correct"] += 1
        self.confusion["Total"][parameter]["correct"] += 1

    def resolve_label(self, model_name, parameter):
        self._ensure_dict(model_name, parameter)
        self.confusion[model_name][parameter]["resolve_label"] += 1
        self.confusion["Total"][parameter]["resolve_label"] += 1

    def resolve_answer(self, model_name, parameter):
        self._ensure_dict(model_name, parameter)
        self.confusion[model_name][parameter]["resolve_answer"] += 1
        self.confusion["Total"][parameter]["resolve_answer"] += 1

    def print_stats(self):
        flat = flatten_empty(self.confusion)
        pprint(flat)
        print(flat)

    def print_prop_stats(self):
        flat = relative_proportion(flatten_empty(self.confusion))
        pprint(flat)
        print(flat)

    def save_csv(self, filename):
        keys = ["model_name", "parameter", "correct", "unit", "wrong"]
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, delimiter=";")

            writer.writeheader()

            for model in self.confusion.keys():
                for parameter in self.confusion[model].keys():
                    writer.writerow(
                        { "model_name": model, "parameter": parameter }
                        | self.confusion[model][parameter] )
        log.info(f"[confusion] Saved to [{filename}]")
