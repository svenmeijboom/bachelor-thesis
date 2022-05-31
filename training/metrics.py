import collections
import re
import string


def normalize_answer(s):
    """Lower text and remove punctuation, articles, accents and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        pattern = re.compile(r'[{}]+'.format(re.escape(string.punctuation)))

        return " ".join(pattern.split(text))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_with_mapping(s):
    """
    Lower text and remove punctuation, articles, accents and extra whitespace.
    Keeps track of mapping between characters in the normalized string and their positions in the original string.
    """

    def replace_with_mapping(pattern, text, mapping):
        regex = re.compile(pattern, re.UNICODE)

        new_mapping = []
        new_text = ''

        start = 0
        for match in regex.finditer(text):
            new_mapping.extend(mapping[start:match.start()])
            new_mapping.append(-1)
            new_text += text[start:match.start()] + ' '
            start = match.end()

        new_text += text[start:]
        new_mapping.extend(mapping[start:])

        return new_text, new_mapping

    def remove_articles(text, mapping):
        return replace_with_mapping(r"\b(a|an|the)\b", text, mapping)

    def white_space_fix(text, mapping):
        return replace_with_mapping(r"\s+", text, mapping)

    def remove_punc(text, mapping):
        return replace_with_mapping(r'[{}]+'.format(re.escape(string.punctuation)), text, mapping)

    def lower(text, mapping):
        return text.lower(), mapping

    char_mapping = list(range(len(s)))

    s, char_mapping = lower(s, char_mapping)
    s, char_mapping = remove_punc(s, char_mapping)
    s, char_mapping = remove_articles(s, char_mapping)
    s, char_mapping = white_space_fix(s, char_mapping)

    return s, char_mapping


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
