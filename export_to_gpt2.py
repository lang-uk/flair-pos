from typing import Generator, List
from collections import namedtuple
import argparse
import pathlib

import flair
from train_grid import UD_UKRAINIAN


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


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert UD dataset for Ukrainian to "
        "the prompt format, suitable for the GPT2 eval. Output goes to an output directory"
    )

    parser.add_argument("outdir", type=pathlib.Path, help="Output directory to store dev/train/test")

    corpus = UD_UKRAINIAN()
    args: argparse.Namespace = parser.parse_args()

    for split in ["dev", "train", "test"]:
        outfile = args.outdir / (split + ".gpt2.txt")
        with outfile.open("w") as fp_out:
            for sentence in getattr(corpus, split):
                fp_out.write(convert_sentence(sentence) + "\n\n")
