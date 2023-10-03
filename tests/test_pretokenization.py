from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Normalizer
from . import RomanianNormalizer, RomanianPreTokenizer

def test_pretokenization_1():
    input_text = '\tŢi-am spus    că merg   și abrevierile, ' + \
        'cum ar fi S.U.A., dar în  acelaşi   timp și expresiile! \n\n' + \
        'Sînt curios dacă reântregirea textului se poate pîrî.'
    input_vocabulary = {
        '[UNK]': 0,
        'Ți-': 1,
        'am': 2,
        'spus': 3,
        'că': 4,
        'merg': 5,
        'și': 6,
        'abrevierile': 7,
        ',': 8,
        'cum': 9,
        'ar': 10,
        'fi': 11,
        'S.U.A.': 12,
        'dar': 13,
        'în același timp': 14,
        'expresiile': 15,
        '!': 16,
        'Sunt': 17,
        'curios': 18,
        'dacă': 19,
        'reîntregirea': 20,
        'textului': 21,
        'se': 22,
        'poate': 23,
        'pârî': 24,
        '.': 25
    }
    wp_model = WordPiece(vocab=input_vocabulary, unk_token='[UNK]', max_input_chars_per_word=25)
    tokenizer = Tokenizer(model=wp_model)
    tokenizer.normalizer = Normalizer.custom(RomanianNormalizer())
    tokenizer.pre_tokenizer = PreTokenizer.custom(RomanianPreTokenizer())
    result = tokenizer.encode(sequence=input_text)
    assert result.tokens == [
        'Ți-', 'am', 'spus', 'că',
        'merg', 'și', 'abrevierile',
        ',', 'cum', 'ar', 'fi',
        'S.U.A.', ',', 'dar', 'în același timp',
        'și', 'expresiile', '!', 'Sunt', 'curios',
        'dacă', 'reîntregirea', 'textului',
        'se', 'poate', 'pârî', '.'
    ]
    