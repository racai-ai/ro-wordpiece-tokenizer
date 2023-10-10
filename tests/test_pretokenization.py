from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Normalizer
from . import ro_normalizer, ro_pretokenizer, ro_train_pretokenizer

_unk_token_str = '[UNK]'

def test_pretokenization():
    input_text = '\tŢi-am spus    că merg   și abrevierile, ' + \
        'cum ar fi S.U.A. și nr. 1, dar în  acelaşi   timp și expresiile! \n\n' + \
        'Sînt curios dacă reântregirea textului se poate pîrî.'
    input_vocabulary = {
        _unk_token_str: 0,
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
        '.': 25,
        'nr.': 26,
        '1': 27
    }
    wp_model = WordPiece(vocab=input_vocabulary,
                         unk_token=_unk_token_str, max_input_chars_per_word=25)
    tokenizer = Tokenizer(model=wp_model)
    tokenizer.normalizer = Normalizer.custom(ro_normalizer)
    tokenizer.pre_tokenizer = PreTokenizer.custom(ro_pretokenizer)
    result = tokenizer.encode(sequence=input_text)
    assert result.tokens == [
        'Ți-', 'am', 'spus', 'că',
        'merg', 'și', 'abrevierile',
        ',', 'cum', 'ar', 'fi',
        'S.U.A.', 'și', 'nr.', '1', ',', 'dar', 'în același timp',
        'și', 'expresiile', '!', 'Sunt', 'curios',
        'dacă', 'reîntregirea', 'textului',
        'se', 'poate', 'pârî', '.'
    ]


def test_pretokenization_bug_1():
    input_text = "Dash Berlin, Ben Nicky, Steve Lee, " + \
        "Phil Reynold, Mr Ralz,Omnia , Andrew Rayel etc ▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲ Chris Coles (UK)"
    input_vocabulary = {
        _unk_token_str: 0,
        'Dash': 1,
        'Berlin': 2,
        ',': 3,
        'Ben': 4,
        'Nicky': 5,
        'Steve': 6,
        'Lee': 7,
        'Phil': 8,
        'Reynold': 9,
        'Mr': 10,
        'Ralz': 11,
        'Omnia': 12,
        'Andrew': 13,
        'Rayel': 14,
        'etc': 15,
        '▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲': 16,
        'Chris': 17,
        'Coles': 18,
        '(': 19,
        'UK': 20,
        ')': 21
    }
    wp_model = WordPiece(vocab=input_vocabulary,
                         unk_token=_unk_token_str, max_input_chars_per_word=25)
    tokenizer = Tokenizer(model=wp_model)
    tokenizer.normalizer = Normalizer.custom(ro_normalizer)
    tokenizer.pre_tokenizer = PreTokenizer.custom(ro_pretokenizer)
    result = tokenizer.encode(sequence=input_text)
    assert result.tokens == [
        'Dash', 'Berlin', ',',
        'Ben', 'Nicky', ',',
        'Steve', 'Lee', ',',
        'Phil', 'Reynold', ',',
        'Mr', 'Ralz', ',',
        'Omnia', ',', 'Andrew', 'Rayel',
        'etc', '▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲▼▲',
        'Chris', 'Coles', '(', 'UK', ')'
    ]


def test_pretokenization_str():
    input_text = '\tCa să vedem  dacă  merge   în principiu cu nr. 1, şi „dacă sîntem gîndindu-ne corect!!”  \t\t'
    norm_text = ro_normalizer.normalize_str(sequence=input_text)
    tokens = ro_pretokenizer.pre_tokenize_str(sequence=norm_text)
    assert len(tokens) == 19
    assert tokens[0][0] == 'Ca să'
    assert tokens[4][0] == 'în principiu'
    assert tokens[6][0] == 'nr.'
    assert tokens[12][0] == 'suntem'


def test_training_pretokenization():
    input_text = "  Recunoașterea_tk_artistică_tk_și_tk_comercială_tk_vine_tk_odată cu_tk_lansarea_tk_" + \
        "celui_tk_de-_tk_al_tk_doilea_tk_album_tk_,_tk_“_tk_Wild_tk_Young_tk_Hearts_tk_”_tk_;\r\n"
    input_vocabulary = {
        _unk_token_str: 0,
        'Recunoașterea': 1,
        'artistică': 2,
        'și': 3,
        'comercială': 4,
        'vine': 5,
        'odată cu': 6,
        'lansarea': 7,
        'celui': 8,
        'de-': 9,
        'al': 10,
        'doilea': 11,
        'album': 12,
        ',': 13,
        '“': 14,
        'Wild': 15,
        'Young': 16,
        'Hearts': 17,
        '”': 18,
        ';': 19,
    }
    wp_model = WordPiece(vocab=input_vocabulary,
                         unk_token=_unk_token_str, max_input_chars_per_word=25)
    tokenizer = Tokenizer(model=wp_model)
    tokenizer.pre_tokenizer = PreTokenizer.custom(ro_train_pretokenizer)
    result = tokenizer.encode(sequence=input_text)
    assert result.tokens == [
        'Recunoașterea', 'artistică', 'și', 'comercială',
        'vine', 'odată cu', 'lansarea', 'celui', 'de-',
        'al', 'doilea', 'album', ',', '“', 'Wild',
        'Young', 'Hearts', '”', ';'
    ]
