# Adapted for Romanian from BertWordPieceTokenizer, of
# tokenizers.implementations.bert_wordpiece

import sys
import os
from typing import Dict, List, Optional, Union
from tokenizers import AddedToken, Tokenizer, decoders, trainers
from tokenizers.models import WordPiece
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer, WhitespaceSplit
from ro_normalizer import RomanianNormalizer
from ro_pretokenizer import RomanianPreTokenizer
from tokenizers.implementations import BaseTokenizer


class RoBertWordPieceTokenizer(BaseTokenizer):
    """Romanian-specific Bert WordPiece Tokenizer"""

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        unk_token: Union[str, AddedToken] = "[UNK]",
        sep_token: Union[str, AddedToken] = "[SEP]",
        cls_token: Union[str, AddedToken] = "[CLS]",
        pad_token: Union[str, AddedToken] = "[PAD]",
        mask_token: Union[str, AddedToken] = "[MASK]",
        wordpieces_prefix: str = "##"
    ):
        ro_pretokenizer = RomanianPreTokenizer()

        if vocab is not None:
            tokenizer = Tokenizer(WordPiece(vocab,
                                            unk_token=str(unk_token),
                                            max_input_chars_per_word=ro_pretokenizer.maxwordlen))
        else:
            tokenizer = Tokenizer(WordPiece(unk_token=str(unk_token),
                                            max_input_chars_per_word=ro_pretokenizer.maxwordlen))

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        # end if
        
        if tokenizer.token_to_id(str(sep_token)) is not None:
            tokenizer.add_special_tokens([str(sep_token)])
        # end if

        if tokenizer.token_to_id(str(cls_token)) is not None:
            tokenizer.add_special_tokens([str(cls_token)])
        # end if

        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])
        # end if

        if tokenizer.token_to_id(str(mask_token)) is not None:
            tokenizer.add_special_tokens([str(mask_token)])
        # end if

        tokenizer.normalizer = Normalizer.custom(RomanianNormalizer())
        tokenizer.pre_tokenizer = PreTokenizer.custom(ro_pretokenizer)

        if vocab is not None:
            sep_token_id = tokenizer.token_to_id(str(sep_token))

            if sep_token_id is None:
                raise TypeError("sep_token not found in the vocabulary")
            # end if

            cls_token_id = tokenizer.token_to_id(str(cls_token))
            
            if cls_token_id is None:
                raise TypeError("cls_token not found in the vocabulary")
            # end if
        # end if

        tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)

        parameters = {
            "model": "RoBertWordPiece",
            "unk_token": unk_token,
            "sep_token": sep_token,
            "cls_token": cls_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "wordpieces_prefix": wordpieces_prefix
        }

        super().__init__(tokenizer, parameters)

    @staticmethod
    def from_file(vocab: str, **kwargs):
        vocab = WordPiece.read_file(vocab)
        return RoBertWordPieceTokenizer(vocab, **kwargs)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 150000,
        min_frequency: int = 2,
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        special_tokens: List[Union[str, AddedToken]] = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
        ],
        show_progress: bool = True,
        wordpieces_prefix: str = "##",
    ):
        """Train the model using the given files"""

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            special_tokens=special_tokens,
            show_progress=show_progress,
            continuing_subword_prefix=wordpieces_prefix,
        )

        if isinstance(files, str):
            files = [files]
        # end if

        self._tokenizer.train(files, trainer=trainer)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 ro_wordpiece.py <folder with .txt files>', file=sys.stderr, flush=True)
        exit(1)
    # end if

    corola_folder = sys.argv[1]
    corola_files = []

    for txt in os.listdir(path=corola_folder):
        if txt.endswith('.txt'):
            corola_files.append(os.path.join(corola_folder, txt))
        # end if
    # end for

    tokenizer = RoBertWordPieceTokenizer()
    # After inspecting the CoRoLa vocabulary, these are the best values.
    tokenizer.train(files=corola_files, vocab_size=500_000, min_frequency=5)
    tokenizer.save_model(directory='model')
