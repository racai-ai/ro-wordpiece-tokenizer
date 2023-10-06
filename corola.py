# Generates the training data for the Romanian WordPiece tokenzer
# and BERT model.

import sys
import os
import re
from tqdm import tqdm


_token_rx = re.compile(
    r'<w>(.+)</w>|' +
    r'<mwe>(.+)</mwe>|' +
    r'<abbr>(.+)</abbr>|' +
    r'<pct>(.+)</pct>|' +
    r'<num>(.+)</num>|' +
    r'<sym>(.+)</sym>'
)


def entity_expansion(token: str) -> str:
    token = token.replace('&quot;', "'")
    token = token.replace('&gt;', ">")
    token = token.replace('&lt;', "<")
    token = token.replace('&amp;', "&")

    return token


def get_sentences_from_xml(xml_file: str,
                           sentence_size: int = 2) -> list[str]:
    """Takes a sentence split and tokenized file from the 'correct'
    folder of the CoRoLa corpus.
    Minimum sentence size is determined by `sentence_size`, default `2`."""

    result = []
    current_sentence = []
    inside_sentence = False

    with open(xml_file, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == '</s>':
                if len(current_sentence) >= sentence_size:
                    result.append(''.join(current_sentence))
                # end if
                inside_sentence = False
            elif line == '<s>':
                current_sentence = []
                inside_sentence = True
            else:
                m = _token_rx.fullmatch(line)
                no_match = True

                if m:
                    for tok in m.groups():
                        if tok:
                            tok = entity_expansion(token=tok)
                            current_sentence.append(tok)
                            break
                        # end if
                    # end for

                    no_match = False
                elif line == '<spc/>' or line == '<eol/>':
                    current_sentence.append(' ')
                    no_match = False
                # end if

                if no_match and inside_sentence and line != '<w/>':
                    print(f'No match for line [{line}] in file [{xml_file}]',
                          file=sys.stderr, flush=True)
                # end if
            # end if
        # end for
    # end with

    return result


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 corola.py <CoRoLa "correct" folder with .xml files> <output folder>')
        exit(1)
    # end if

    correct_folder = sys.argv[1]
    output_folder = sys.argv[2]
    current_sentence_chunk = []
    file_counter = 1

    for xf in tqdm(os.listdir(correct_folder), desc='CoRoLa'):
        if xf.endswith('.xml'):
            sentences = get_sentences_from_xml(os.path.join(correct_folder, xf))
            
            if len(current_sentence_chunk) + len(sentences) < 100000:
                current_sentence_chunk.extend(sentences)
            else:
                output_sentence_file = os.path.join(output_folder,
                                                    f'corola-sentences-{file_counter}.txt')
                
                with open(output_sentence_file, mode='w', encoding='utf-8') as f:
                    for snt in current_sentence_chunk:
                        print(snt, file=f)
                    # end for
                # end with

                file_counter += 1
                current_sentence_chunk = []
                current_sentence_chunk.extend(sentences)
            # end if
        # end if
    # end for

    if current_sentence_chunk:
        output_sentence_file = os.path.join(output_folder,
                                            f'corola-sentences-{file_counter}.txt')

        with open(output_sentence_file, mode='w', encoding='utf-8') as f:
            for snt in current_sentence_chunk:
                print(snt, file=f)
            # end for
        # end with
    # end if
