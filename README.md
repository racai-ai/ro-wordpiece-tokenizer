# A Romanian WordPiece Tokenizer for use with HuggingFace models
This is a 'proper' Romanian WordPiece tokenizer, to be used with the HuggingFace `tokenizers` library.

It will do the following:
- replace all improper Romanian diacritics 'ş' and 'ţ' with their correct versions 'ș' and 'ț'
- properly split the Romanian clitics glued to nouns, prepositions, verbs, etc.

The tokenizer will be trained on a cleaned version of the [CoRoLa corpus](https://corola.racai.ro/).
