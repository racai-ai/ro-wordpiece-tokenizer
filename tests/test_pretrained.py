from pathlib import Path
from . import RoBertWordPieceTokenizer

def test_one():
    input_text = "\t\tIa s-o vedem   de fapt, dacă pîrîie cum trebuie, sîntem OK?\n\n"
    vocab_path = Path(__file__).parent.parent / 'model' / 'vocab.txt'
    tokenizer = RoBertWordPieceTokenizer.from_file(vocab=str(vocab_path))
    
    result_encoded = tokenizer.encode(sequence=input_text)
    assert result_encoded.tokens[1] == 's-'
    assert result_encoded.tokens[4] == 'de fapt'
    assert result_encoded.tokens[7] == 'pârâie'
    assert result_encoded.tokens[11] == 'suntem'

    result_decoded = tokenizer.decode(ids=result_encoded.ids)
    assert result_decoded == 'Ia s- o vedem de fapt, dacă pârâie cum trebuie, suntem OK?'


def test_two():
    input_text = 'Într-o zi cu soare, și-a făcut-o și mi-a dus-o făcându-mi-o pe bune.'
    vocab_path = Path(__file__).parent.parent / 'model' / 'vocab.txt'
    tokenizer = RoBertWordPieceTokenizer.from_file(vocab=str(vocab_path))
    result_encoded = tokenizer.encode(sequence=input_text)
    result_decoded = tokenizer.decode(ids=result_encoded.ids)
    assert result_decoded == 'Într- o zi cu soare, și- a făcut -o și mi- a dus -o făcându -mi -o pe bune.'
