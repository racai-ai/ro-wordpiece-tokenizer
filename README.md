# A Romanian WordPiece Tokenizer for use with HuggingFace models
This is a 'proper' Romanian WordPiece tokenizer, to be used with the HuggingFace `tokenizers` library.

It will do the following:
1. replace all improper Romanian diacritics 'ş' and 'ţ' with their correct versions 'ș' and 'ț'.
2. properly split the Romanian clitics glued to nouns, prepositions, verbs, etc.
3. automatically enforce the current Romanian Academy rules of writing using 'â' and 'sunt/suntem/sunteți' forms of the 'a fi' verb.

The tokenizer will be trained on a cleaned version of the [CoRoLa corpus](https://corola.racai.ro/).
The corpus has 35.999.401 sentences and 763.531.321 words (split with `wc -w` Linux utility).

# Usage example
```python
import os
from ro_wordpiece import RoBertWordPieceTokenizer

corola_vocab_file = os.path.join('model', 'vocab.txt')
tokenizer = RoBertWordPieceTokenizer.from_file(vocab=corola_vocab_file)

input_text = "\t\tSîntem OK şi ar trebui să-mi meargă, în principiu.\n\n"
result_encoded = tokenizer.encode(sequence=input_text)
# We have Romanian tokens such as the clitic pronoun '-mi' or
# the MWE 'în principiu'. Also, the incorrect form of the verb 'Sîntem'
# is normalized as 'Suntem'.
assert result_encoded.tokens[0] == 'Suntem'
assert result_encoded.tokens[6] == '-mi'
assert result_encoded.tokens[9] == 'în principiu'

result_decoded = tokenizer.decode(ids=result_encoded.ids)

assert result_decoded == 'Suntem OK și ar trebui să -mi meargă, în principiu.'
```

Full Romanian decoding isn't currently working (please notice the space between 'să' and '-mi') because `decoders.Decoder.custom()` is not implmemented yet in the `tokenizers` library.

# Transformers usage example
In order to use the tokenizer with the `__call__` method (as preferred in the Transformers documentation), do the following:

```python
import os
from ro_wordpiece import RoBertPreTrainedTokenizer

corola_vocab_file = os.path.join('model', 'vocab.txt')
tokenizer = RoBertPreTrainedTokenizer.from_pretrained(
    corola_vocab_file, model_max_length=256)
input_text = "\t\tSîntem OK şi ar trebui să-mi meargă, în principiu.\n\n"
result_encoded = tokenizer(text=input_text, padding='max_length')
```

# PyPI package example
The tokenizer is now available on PyPI, and can be installed with the command `pip install rwpt`.

The example above can then be rewritten as:

```python
from rwpt import load_ro_pretrained_tokenizer

tokenizer = load_ro_pretrained_tokenizer(max_sequence_len=256)
input_text = "\t\tSîntem OK şi ar trebui să-mi meargă, în principiu.\n\n"
result_encoded = tokenizer(text=input_text, padding='max_length')
```
