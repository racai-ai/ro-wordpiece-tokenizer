# This script takes the output of the corola.py script and
# prepares the sentences for the RoBertWordPieceTokenizer training

import os
import sys
from pathlib import Path
from time import sleep
from tqdm import tqdm
from multiprocessing import Process
from ro_normalizer import RomanianNormalizer
from ro_pretokenizer import RomanianPreTokenizer


def process_file(input_file: str, output_folder: str) -> None:
    input_file_name = Path(input_file).name
    
    print(f'Starting process [{input_file_name}]', file=sys.stderr, flush=True)
    
    output_file = Path(output_folder) / input_file_name
    ro_normal = RomanianNormalizer()
    ro_pretok = RomanianPreTokenizer()
    
    with open(output_file, mode='w', encoding='utf8') as ff:
        with open(input_file, mode='r', encoding='utf-8') as f:
            for sentence in f:
                sentence = ro_normal.normalize_str(sentence)
                tokens = ro_pretok.pre_tokenize_str(sentence)
                only_tokens = [x[0] for x in tokens]
                print('_tk_'.join(only_tokens), file=ff)
            # end for
        # end with
    # end with

    print(f'Finished process [{input_file_name}]', file=sys.stderr, flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 5 or len(sys.argv) < 3:
        print('Usage: python3 ro_traindata.py [-p <count>] <source folder with .txt sentence files> <output folder>',
              file=sys.stderr, flush=True)
    if sys.argv[1] == '-p':
        process_count = int(sys.argv[2])
        sys.argv.pop(2)
        sys.argv.pop(1)
    else:
        process_count = 6
    # end if

    print(f'Running with [{process_count}] processes', file=sys.stderr, flush=True)

    source_folder = sys.argv[1]
    target_folder = sys.argv[2]
    process_queue: list[Process] = []

    for txt in tqdm(os.listdir(source_folder), desc='Processes'):
        if txt.endswith('.txt'):
            txt_file = os.path.join(source_folder, txt)
            
            if len(process_queue) < process_count:
                pr = Process(name=txt, target=process_file, args=(txt_file, target_folder))
                process_queue.append(pr)
                pr.start()
            else:
                all_alive = True

                while all_alive:
                    i = 0

                    while i < len(process_queue):
                        pr = process_queue[i]

                        if not pr.is_alive():
                            # Make room for new process in the queue
                            all_alive = False
                            process_queue.pop(i)
                            
                            # Start a new process
                            pr = Process(name=txt, target=process_file, args=(txt_file, target_folder))
                            process_queue.append(pr)
                            pr.start()

                            # And bail out (take next file)
                            break
                        # end if

                        i += 1
                    # end while

                    sleep(3)
                # end while
            # end if
        # end if
    # end for

    # Wait for everyone to finish
    for pr in process_queue:
        pr.join()
    # end for
