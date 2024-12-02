import re
import time
import pubchempy as pcp
import logging
import numpy as np
from functools import cache

p = logging.getLogger("pubchempy").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def ans2cid(answer: str, paragraph_id: str = None) -> int:
    cids = []
    for word in re.split(" |/", answer):
        for char in [".", ","]:
            if word.endswith(char):
                word = word[:-1]  # remove punctuation marks at the end of words
        if word.lower() not in [
            "and",
            "at",
            "as",
            "in",
            "is",
            "of",
            "out",
            "the",
            "was",
            "acts",
        ]:  # common words which are also in the database as molecules
            for char in ["⋅", "·"]:  # crytsla water not recognzed by pubchempy
                if char in word:
                    word = word.split(char)[0]
                    break
            try:
                cid = pcp.get_cids(word.strip())
                if len(cid) > 0:
                    cids.append([cid[0], word])
            except pcp.PubChemHTTPError as e:
                log.error(e)
    if len(cids) == 0:
        return None
    if len(cids) > 1:
        log.warn(f'Found more than one cid: {cids} in "{answer}" for {paragraph_id}')
    return cids[0][0]

# TODO: improve so that multiple cids can be returned or include multiple hardcoded answers like DMA/DMF
@cache
def txt2cid(txt: str) -> list[int]:
    if not txt:
        return -1
    if "DMA" in txt:
        return [31374]
    try:
        r = pcp.get_cids(txt.strip())
        if not r:
            log.error(f"pcp: Could not find `cid` for [{txt}]")
        return r
    except pcp.PubChemHTTPError as e:
        log.warning(e)
        time.sleep(2)
        return txt2cid(txt)
    except URLError as e:
        log.warning(e)
        time.sleep(2)
        return txt2cid(txt)


@cache
def cid2syns(cid: int) -> list[str]:
    try:
        return pcp.Compound.from_cid(cid).synonyms
    except pcp.PubChemHTTPError as e:
        log.warning(e)
        time.sleep(2)
        return cid2syns(txt)


def ans2tempcelsius(answer: str, default_celsius: bool = True) -> int | None:
    """ Convert a string, potentially containing a number and a
    temperature unit, to a temperature in celsius
    """
    answer = answer.replace(".0", "")
    answer = answer.strip() + " "

    degC, K = None, None

    degC_search = re.search(
        "(\d)+.C|(\d)+.(\xB0C?)|(\d)+.(°C?)|(\d)+.(oC?)|(\d)+.h\W", answer + " "
    )
    if degC_search != None:
        degC = int(re.search("(\d)+", degC_search.group()).group())

    K_search = re.search("(\d)+.(K?)|(\d)+.h\W", answer + " ")
    if K_search != None:
        K = int(re.search("(\d)+", K_search.group()).group())

    if degC is not None and default_celsius:
        return degC
    elif K is not None:
        return K - 273
    else:
        return None


def ans2hours(answer: str) -> int:
    """Convert a string, potentially containing a number and
    time unit, to a duration in hours.
    """
    answer = answer.replace(".0", "")
    answer = answer.strip() + " "
    numbers_dir = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    for key in numbers_dir:
        answer = answer.replace(key, str(numbers_dir[key]))

    seconds, minutes, hours, days, weeks = 0, 0, 0, 0, 0

    sec_search = re.search("(\d)+.([sS]ec(onds)?)|(\d)+.s\W", answer + " ")
    if sec_search != None:
        seconds = int(re.search("(\d)+", sec_search.group()).group())

    min_search = re.search("(\d)+.([mM]in(utes)?)|(\d)+.m\W", answer + " ")
    if min_search != None:
        minutes = int(re.search("(\d)+", min_search.group()).group())

    hours_search = re.search("(\d)+.([hH]ours?)|(\d)+.h\W", answer + " ")
    if hours_search != None:
        hours = int(re.search("(\d)+", hours_search.group()).group())

    days_search = re.search("(\d)+.([dD]ays?)|(\d)+.d\W", answer + " ")
    if days_search != None:
        days = int(re.search("(\d)+", days_search.group()).group())

    weeks_search = re.search("(\d)+.([wW]eeks?)|(\d)+.w\W", answer + " ")
    if weeks_search != None:
        weeks = int(re.search("(\d)+", weeks_search.group()).group())

    if seconds > 0:
        minutes += seconds // 60

    if minutes > 0:
        hours += minutes // 60

    return 24 * (7 * weeks + days) + hours


def _outer_levenshtein(x: np.array, y: np.array):
    levenshtein_vec = np.vectorize(distance, signature="(),()->()")
    x = x[:, None]
    y = y[:, None].T
    return levenshtein_vec(x, y)


def convert_string_to_cid(model_answer, table, name_column, cid_column):
    all_substance_names = np.asarray(table[name_column].values)

    distances = np.empty_like(all_substance_names)
    answers = np.asarray([model_answer]).flatten()

    for array_index, names_for_solvent in enumerate(all_substance_names):
        names = np.asarray(names_for_solvent.split(";;"))
        dist_matrix = _outer_levenshtein(x=answers, y=names)
        distances[array_index] = np.min(dist_matrix)

    best_match_index = np.argmin(distances)

    best_matching_cids = table.iloc[best_match_index][cid_column]

    return best_matching_cids
