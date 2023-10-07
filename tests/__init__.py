from ro_normalizer import RomanianNormalizer
from ro_pretokenizer import RomanianPreTokenizer, TrainingPreTokenizer
from ro_wordpiece import RoBertWordPieceTokenizer
from pathlib import Path

ro_normalizer = RomanianNormalizer()
ro_pretokenizer = RomanianPreTokenizer()
ro_train_pretokenizer = TrainingPreTokenizer()
corola_vocab_path = Path(__file__).parent.parent / 'model' / 'vocab.txt'
tokenizer = RoBertWordPieceTokenizer.from_file(vocab=str(corola_vocab_path))
