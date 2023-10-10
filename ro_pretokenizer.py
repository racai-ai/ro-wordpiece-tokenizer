import sys
from tokenizers import PreTokenizedString
from tokenizers import NormalizedString
from rodna.tokenizer import RoTokenizer


class RomanianPreTokenizer(object):
    def __init__(self) -> None:
        self._romanian_tokenizer = RoTokenizer()

    @property
    def maxwordlen(self) -> int:
        """The length of the longest word/MWE/ABBR for Romanian."""
        return self._romanian_tokenizer._maxwordlen

    def _romanian_split(self, index: int, normstr: NormalizedString) -> list[NormalizedString]:
        norm_string = normstr.normalized
        result = []

        if not norm_string:
            result.append(norm_string)
            return result
        # end if

        ro_tokens = self._romanian_tokenizer.tokenize(
            input_string=norm_string)
        loff = 0
        roff = 0
        crt_token = ro_tokens.pop(0)
        out_of_sync = False

        while True:
            for i in range(len(crt_token)):
                if crt_token[i] == norm_string[roff] or \
                        (crt_token[i] == '_' and norm_string[roff] == ' '):
                    roff += 1
                else:
                    print(f'Current [{crt_token}] token out of sync (i = {i}, roff = {roff}), in normalized string [{norm_string}]',
                          file=sys.stderr, flush=True)
                    out_of_sync = True
                    break
                # end if
            # end for

            result.append(normstr.slice(range=(loff, roff)))

            if roff == len(norm_string):
                # Reached the EOS
                break
            elif out_of_sync:
                # There is an offset synchronization problem
                # Add the rest of the string as a single slice
                # and bail out
                if roff < len(norm_string):
                    result.append(normstr.slice(range=(roff, len(norm_string))))
                # end if
                
                break
            # end if

            # We can have only one space!
            if norm_string[roff] == ' ':
                roff += 1
            # end if

            loff = roff
            crt_token = ro_tokens.pop(0)
        # end while

        return result

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(func=self._romanian_split)

    def pre_tokenize_str(self, sequence: str) -> list[tuple[str, tuple[int, int]]]:
        result = []

        if not sequence:
            result.append((sequence, (0, len(sequence))))
            return result
        # end if

        ro_tokens = self._romanian_tokenizer.tokenize(
            input_string=sequence)
        loff = 0
        roff = 0
        crt_token = ro_tokens.pop(0)
        out_of_sync = False

        while True:
            for i in range(len(crt_token)):
                if crt_token[i] == sequence[roff] or \
                        (crt_token[i] == '_' and sequence[roff] == ' '):
                    roff += 1
                else:
                    print(f'Current [{crt_token}] token out of sync (i = {i}, roff = {roff}), in normalized string [{sequence}]',
                          file=sys.stderr, flush=True)
                    out_of_sync = True
                    break
                # end if
            # end for

            result.append((sequence[loff:roff], (loff, roff)))

            if roff == len(sequence):
                # Reached the EOS
                break
            elif out_of_sync:
                # There is an offset synchronization problem
                # Add the rest of the string as a single slice
                # and bail out
                if roff < len(sequence):
                    result.append((sequence[roff:], (roff, len(sequence))))
                # end if
                
                break
            # end if

            # We can have only one space!
            if sequence[roff] == ' ':
                roff += 1
            # end if

            loff = roff
            crt_token = ro_tokens.pop(0)
        # end while

        return result


class TrainingPreTokenizer(object):
    """Only used when training on pre-tokenized data.
    Just split at '_tk_' boundary and remove it."""

    delimiter = '_tk_'

    def _train_split(self, index: int, normstr: NormalizedString) -> list[NormalizedString]:
        normstr.strip()
        
        if TrainingPreTokenizer.delimiter in normstr.normalized:
            return normstr.split(pattern=TrainingPreTokenizer.delimiter,
                                 behavior='removed')
        else:
            return [normstr]
        # end if

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(func=self._train_split)
