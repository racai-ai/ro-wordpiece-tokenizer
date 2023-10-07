# Not tested yet.

class RomanianDecoder(object):
    """This class will glue back Romanian clitics to their
    previous or next token."""

    def decode(self, tokens: list[str]) -> str:
        if len(tokens) == 1:
            return tokens[0]
        # end if

        result = []

        for i in range(0, len(tokens) - 1):
            ctok = tokens[i]
            ntok = tokens[i + 1]
            
            if ctok.endswith('-') or ntok.startswith('-'):
                result.append(ctok)
            else:
                result.append(ctok + ' ')
            # end if
        # end for

        result.append(tokens[-1])

        return ''.join(result)
