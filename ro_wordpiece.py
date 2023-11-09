# Adapted for Romanian from BertWordPieceTokenizer, of
# tokenizers.implementations.bert_wordpiece

import sys
import os
from typing import Dict, List, Optional, Union
from tokenizers import AddedToken, Tokenizer, decoders, trainers
from tokenizers.models import WordPiece
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from ro_normalizer import RomanianNormalizer
from ro_pretokenizer import RomanianPreTokenizer, TrainingPreTokenizer
from ro_decoder import RomanianDecoder
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizer


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
        wordpieces_prefix: str = "##",
        train_mode: bool = False
    ):
        if train_mode:
            ro_pretokenizer = TrainingPreTokenizer()
            max_token_len = RomanianPreTokenizer().maxwordlen
        else:
            ro_pretokenizer = RomanianPreTokenizer()
            max_token_len = ro_pretokenizer.maxwordlen
        # end if

        if vocab is not None:
            tokenizer = Tokenizer(WordPiece(vocab,
                                            unk_token=str(unk_token),
                                            max_input_chars_per_word=max_token_len))
        else:
            tokenizer = Tokenizer(WordPiece(unk_token=str(unk_token),
                                            max_input_chars_per_word=max_token_len))
        # end if

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

        if not train_mode:
            tokenizer.normalizer = Normalizer.custom(RomanianNormalizer())
        # end if

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

        tokenizer.decoder = decoders.Sequence([
            decoders.WordPiece(prefix=wordpieces_prefix, cleanup=True),
            # TODO: Decoder.custom() is not yet implemented in tokenizers.
            # When updating this module, try and uncomment the next decoder
            # and update the test_pretrained.py/test_one() and test_two() methods
            # to remove spaces in front of clitics (e.g. '-o')
            #
            # decoders.Decoder.custom(RomanianDecoder())
        ])
        
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


class RoBertPreTrainedTokenizer(PreTrainedTokenizer):
    """Use this class with the `transformers` library.

    The following should be enforced:
    - the `RoBertWordPieceTokenizer`, which is the underlying tokenizer,
    is case sensitive, so the use of `do_lower_case` is not tested 
    - `save_pretrained` does not work with custom/Python tokenizers
    - this tokenizer cannot be pushed to the HuggingFace hub."""

    vocab_files_names = {
        'RoBertWordPieceTokenizer': os.path.join(os.path.dirname(__file__), 'model', 'vocab.txt')
    }

    def __init__(self, *init_inputs, **kwargs):
        if 'name_or_path' in kwargs:
            # When called from RoBertPreTrainedTokenizer.from_pretrained(vocab.txt)
            self._ro_wordpiece_tokenizer = \
                RoBertWordPieceTokenizer.from_file(
                    vocab=kwargs['name_or_path'])
        else:
            self._ro_wordpiece_tokenizer = RoBertWordPieceTokenizer.from_file(
                vocab=RoBertPreTrainedTokenizer.vocab_files_names['RoBertWordPieceTokenizer'])
        
        super().__init__(
            unk_token=self._ro_wordpiece_tokenizer._parameters['unk_token'],
            sep_token=self._ro_wordpiece_tokenizer._parameters['sep_token'],
            pad_token=self._ro_wordpiece_tokenizer._parameters['pad_token'],
            cls_token=self._ro_wordpiece_tokenizer._parameters['cls_token'],
            mask_token=self._ro_wordpiece_tokenizer._parameters['mask_token'],
            **kwargs)

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without externally added new tokens).
        """
        return self._ro_wordpiece_tokenizer.get_vocab_size(with_added_tokens=True)
    
    def get_vocab(self) -> Dict[str, int]:
        return self._ro_wordpiece_tokenizer.get_vocab(with_added_tokens=True)

    def _tokenize(self, text, **kwargs):
        return self._ro_wordpiece_tokenizer.encode(sequence=text,
                                                   is_pretokenized=False,
                                                   add_special_tokens=False).tokens

    def _convert_token_to_id(self, token):
        return self._ro_wordpiece_tokenizer.token_to_id(token=token)
    
    def _convert_id_to_token(self, index: int) -> str:
        return self._ro_wordpiece_tokenizer.id_to_token(id=index)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 ro_wordpiece.py <folder with .txt files>', file=sys.stderr, flush=True)
        exit(1)
    # end if

    corola_folder = sys.argv[1]
    corola_files = []

    # Training files have to be:
    # 1. Already normalized with the RomanianNormalizer
    # 2. Already tokenized with the RomanianPreTokenizer
    # (tokens are obtained by pre-tokenizing with the TrainingPreTokenizer)
    for txt in os.listdir(path=corola_folder):
        if txt.endswith('.txt'):
            corola_files.append(os.path.join(corola_folder, txt))
        # end if
    # end for

    tokenizer = RoBertWordPieceTokenizer(train_mode=True)
    # After inspecting the CoRoLa vocabulary, these are the best values.
    tokenizer.train(files=corola_files, vocab_size=500_000, min_frequency=5)
    tokenizer.save_model(directory='model')

    # Bug: save_model() saves some duplicate tokens...
    vocab_file_in = os.path.join('model', 'vocab.txt')
    vocab_file_out = os.path.join('model', 'vocab2.txt')
    vocab_terms = set()

    with open(vocab_file_out, mode='w', encoding='utf-8') as f:
        with open(vocab_file_in, mode='r', encoding='utf-8') as ff:
            for line in ff:
                term = line.strip()

                if term not in vocab_terms:
                    print(term, file=f, end='\n')
                    vocab_terms.add(term)
                else:
                    print(f'vocab.txt term [{term}] is duplicated', file=sys.stderr, flush=True)
                # end if
            # end for
        # end with
    # end with

    os.remove(path=vocab_file_in)
    os.rename(src=vocab_file_out, dst=vocab_file_in)
