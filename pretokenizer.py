from tokenizers import PreTokenizedString
from tokenizers import NormalizedString
from rodna.tokenizer import RoTokenizer


class RomanianPreTokenizer(object):
    def __init__(self) -> None:
        self._romanian_tokenizer = RoTokenizer()

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

        while True:
            for i in range(len(crt_token)):
                if crt_token[i] == norm_string[roff] or \
                        (crt_token[i] == '_' and norm_string[roff] == ' '):
                    roff += 1
                else:
                    raise RuntimeError(f'Current [{crt_token}] token out of sync')
                # end if
            # end for

            result.append(normstr.slice(range=(loff , roff)))

            if roff == len(norm_string):
                # Reached the EOS
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
