from typing import Generator, List
from collections import namedtuple
import argparse
import pathlib

import flair
from train_grid import UD_UKRAINIAN


# As per: https://github.com/lang-uk/lang.org.ua/blob/master/languk/corpus/ud_converter.py#L10
_UPOS_MAPPING = [
    ("NOUN", "N"),  # 4130481
    ("VERB", "V"),  # 3345193
    ("ADP", "a"),  # 1851693
    ("ADV", "A"),  # 1651200
    ("PRON", "P"),  # 1525969
    ("ADJ", "J"),  # 1427357
    ("PART", "p"),  # 1147072
    ("CCONJ", "C"),  # 1101499
    ("DET", "D"),  # 873070
    ("PROPN", "O"),  # 684675
    ("SCONJ", "S"),  # 484188
    ("X", "X"),  # 175188
    ("NUM", "n"),  # 96248
    ("PUNCT", "t"),  # 88265
    ("INTJ", "I"),  # 61924
    ("SYM", "s"),  # 415
    ("AUX", "x"),  # 275
]

COMPRESS_UPOS_MAPPING = dict(_UPOS_MAPPING)


class AlignedToken(namedtuple("AlignedToken", ("token", "orig_pos", "new_pos"))):
    """
    As we do changing the whitespaces in tokenized text according to the punctuation rules,
    we need a class to maintain the positions of whitespace tokenized vs. normalized one
    """

    __slots__: tuple = ()

    def __str__(self) -> str:
        return str(self.token)


# Borrowed from https://github.com/lang-uk/vulyk-ner/blob/master/bin/convert2vulyk.py#L136
def reconstruct_tokenized(tokenized_text: List[List[str]]) -> Generator[AlignedToken, None, None]:
    """
    Accepts tokenized text [["sent1_word1", "sent1_word2"], ["sent2_word2"]]
    and normalizes spaces in the text according to the punctuation.
    Returns an iterator over AlignedToken, where each token has the information
    on the original position and updated position
    """
    SPACES_BEFORE: str = "([“«"
    NO_SPACE_BEFORE: str = ".,:!?)]”»"

    orig_pos: int = 0
    adj_pos: int = 0

    for s_idx, s in enumerate(tokenized_text):
        if s_idx > 0:
            yield AlignedToken("\n", (orig_pos, orig_pos + 1), (adj_pos, adj_pos + 1))
            orig_pos += 1
            adj_pos += 1

        prev_token: str = ""
        for w_idx, w in enumerate(s):
            w_stripped = w.strip()

            if not w_stripped:
                # If original text contained a space(-es), let's adjust original position for it
                # + one space after
                orig_pos += len(w)
                if w_idx > 0:
                    orig_pos += 1

                continue

            if w_idx > 0:
                if w_stripped not in NO_SPACE_BEFORE and not prev_token in SPACES_BEFORE:
                    yield AlignedToken(" ", (orig_pos, orig_pos + 1), (adj_pos, adj_pos + 1))
                    orig_pos += 1
                    adj_pos += 1
                else:
                    # If we are omitting the space (for example, before comma), we
                    # adjusting original position as if it's there
                    orig_pos += 1

            yield AlignedToken(w_stripped, (orig_pos, orig_pos + len(w)), (adj_pos, adj_pos + len(w_stripped)))

            orig_pos += len(w)
            adj_pos += len(w_stripped)

            prev_token = w_stripped


def convert_sentence(sentence: flair.data.Sentence, prefix_text: str = "речення: ", label="upos") -> str:
    words: List[str] = []
    tagged: List[str] = []

    for w in sentence:
        words.append(w.text)
        tagged.append(f"{w.get_label(label).value}: {w.text}")

    final_sentence: str = "".join(map(str, reconstruct_tokenized([words])))

    return prefix_text + final_sentence + "\n" + "\n".join(tagged)


def convert_sentence_inline(
    sentence: flair.data.Sentence, prefix_text: str = "речення: ", annotation: str = "анотація: ", label="upos"
) -> str:
    words: List[str] = []
    tagged: List[str] = []

    for w in sentence:
        words.append(w.text)

        tag: str = COMPRESS_UPOS_MAPPING[w.get_label(label).value]

        tagged.append(w.text)
        tagged.append(f"/{tag}")

    final_sentence: str = "".join(map(str, reconstruct_tokenized([words])))
    final_tagged_sentence: str = " ".join(tagged)
    return prefix_text + final_sentence + "\n" + annotation + final_tagged_sentence


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert UD dataset for Ukrainian to "
        "the prompt format, suitable for the GPT2 eval. Output goes to an output directory"
    )

    parser.add_argument("outdir", type=pathlib.Path, help="Output directory to store dev/train/test")
    parser.add_argument("--format", default="inline", choices=["inline", "post"])

    corpus = UD_UKRAINIAN()
    args: argparse.Namespace = parser.parse_args()

    for split in ["dev", "train", "test"]:
        outfile = args.outdir / (split + f".{args.format}.gpt2.txt")
        with outfile.open("w") as fp_out:
            for sentence in getattr(corpus, split):
                if args.format == "inline":
                    fp_out.write(convert_sentence_inline(sentence) + "\n\n")
                else:
                    fp_out.write(convert_sentence(sentence) + "\n\n")
