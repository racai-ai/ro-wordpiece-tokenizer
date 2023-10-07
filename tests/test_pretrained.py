from . import tokenizer

def test_one():
    input_text = "\t\tIa s-o vedem   de fapt, dacă pîrîie cum trebuie, sîntem OK?\n\n"
    result_encoded = tokenizer.encode(sequence=input_text)
    assert result_encoded.tokens[1] == 's-'
    assert result_encoded.tokens[4] == 'de fapt'
    assert result_encoded.tokens[7] == 'pârâie'
    assert result_encoded.tokens[11] == 'suntem'
    result_decoded = tokenizer.decode(ids=result_encoded.ids)
    assert result_decoded == 'Ia s- o vedem de fapt, dacă pârâie cum trebuie, suntem OK?'


def test_two():
    input_text = 'Într-o zi cu soare, și-a făcut-o și mi-a dus-o făcându-mi-o pe bune.'
    result_encoded = tokenizer.encode(sequence=input_text)
    result_decoded = tokenizer.decode(ids=result_encoded.ids)
    assert result_decoded == 'Într- o zi cu soare, și- a făcut -o și mi- a dus -o făcându -mi -o pe bune.'


def test_howto():
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
