from . import RomanianNormalizer
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import WhitespaceSplit


def test_normalization_1():
    input_text = '\t Sîntem aici   pe        neîngrădita mirişte din Romînia\n\n'
    input_vocabulary = {
        '[UNK]': 0,
        'Suntem': 1,
        'aici': 2,
        ',': 3,
        'pe': 4,
        'neîngrădita': 5,
        'miriște': 6,
        'din': 7,
        'România': 8
    }
    wp_model = WordPiece(vocab=input_vocabulary, unk_token='[UNK]', max_input_chars_per_word=15)
    tokenizer = Tokenizer(model=wp_model)
    tokenizer.normalizer = Normalizer.custom(RomanianNormalizer())
    tokenizer.pre_tokenizer = WhitespaceSplit()
    result = tokenizer.encode(sequence=input_text)
    
    assert result.tokens == ['Suntem', 'aici', 'pe', 'neîngrădita', 'miriște', 'din', 'România']

def test_normalization_2():
    input_text = '\t Sîntem aici,   pe        neîngrădita mirişte din Romînia!\n\n'
    normalizer = RomanianNormalizer()
    norm_text = normalizer.normalize_str(sequence=input_text)
    
    assert norm_text == 'Suntem aici, pe neîngrădita miriște din România!'
