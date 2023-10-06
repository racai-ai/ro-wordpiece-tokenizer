# A Romanian WordPiece Tokenizer for use with HuggingFace models
This is a 'proper' Romanian WordPiece tokenizer, to be used with the HuggingFace `tokenizers` library.

It will do the following:
1. replace all improper Romanian diacritics 'ş' and 'ţ' with their correct versions 'ș' and 'ț'.
2. properly split the Romanian clitics glued to nouns, prepositions, verbs, etc.
3. (_under investigation_): automatically enforce the current Romanian Academy rules of writing using 'â' and 'sunt/suntem/sunteți' forms of the 'a fi' verb.

The tokenizer will be trained on a cleaned version of the [CoRoLa corpus](https://corola.racai.ro/).
The corpus has 35.999.401 sentences and 763.531.321 words (split with `wc -w` Linux utility).
