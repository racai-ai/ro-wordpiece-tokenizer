import re
from tokenizers import NormalizedString, Regex


# Prefixes from paper:
# Verginica Barbu Mititelu. "INVENTARUL AFIXELOR ROMÂNEȘTI". Institute for AI, Romanian Academy.
ro_morpho_prefixes = [
    'a', 'ab', 'an', 'ana', 'ante', 'anti', 'antre',
    'apo', 'arhi', 'cata', 'circum', 'cis', 'co',
    'con', 'contra', 'cu', 'de', 'des', 'dia', 'dis',
    'ecto', 'en', 'endo', 'ento', 'epi', 'ex', 'exo',
    'extra', 'hiper', 'hipo', 'in', 'infra', 'inter',
    'intra', 'intro', 'în', 'între', 'întru', 'juxta',
    'me', 'meta', 'nă', 'ne', 'non', 'o', 'ob', 'par',
    'para', 'pen', 'per', 'peri', 'po', 'pod', 'poi',
    'post', 'pre', 'prea', 'pro', 'ră', 'răs', 'răz',
    're', 'retro', 's', 'se', 'sin', 'spre', 'stră',
    'sub', 'super', 'supra', 'sur', 'tă', 'tra', 'trans',
    'tră', 'tre', 'ultra', 'vă', 'văz'
]


class RomanianNormalizer(object):
    """Takes a Romanian text and performs normalizations such as:
    - use of proper diacritics for 'ș' and 'ț'
    - enforce up-to-date Romanian Academy writing norms"""

    def __init__(self) -> None:
        """Nothing to init here."""
        pass

    def _replace_diacs(self, char: str) -> str:
        match char:
            case 'ş':
                return 'ș'
            case 'Ş':
                return 'Ș'
            case 'ţ':
                return 'ț'
            case 'Ţ':
                return 'Ț'
            case _:
                return char
        # end match

    def normalize(self, normalized: NormalizedString) -> None:
        # 1. Remove spaces left and right
        normalized.strip()
        # 2. Use standard Romanian diacritics
        normalized.map(func=self._replace_diacs)
        # 3. Remove consecutive spaces
        normalized.replace(pattern=Regex(r'\s+'), content=' ')
        # 4. Enforce correct forms for 'a fi'
        normalized.replace(pattern=Regex(r'\bsînt\b'), content='sunt')
        normalized.replace(pattern=Regex(r'\bSînt\b'), content='Sunt')
        normalized.replace(pattern=Regex(r'\bsîntem\b'), content='suntem')
        normalized.replace(pattern=Regex(r'\bSîntem\b'), content='Suntem')
        normalized.replace(pattern=Regex(r'\bsînteți\b'), content='sunteți')
        normalized.replace(pattern=Regex(r'\bSînteți\b'), content='Sunteți')
        # 5. Use 'â' everywhere...
        normalized.replace(pattern='î', content='â')
        normalized.replace(pattern='Î', content='Â')
        # 6. But not at start/end of words
        normalized.replace(pattern=Regex(r'â\b'), content='î')
        normalized.replace(pattern=Regex(r'Â\b'), content='Î')
        normalized.replace(pattern=Regex(r'\bâ'), content='î')
        normalized.replace(pattern=Regex(r'\bÂ'), content='Î')

        # 7. And not after morphological prefixes
        for pref in ro_morpho_prefixes:
            normalized.replace(pattern=Regex(f'\\b{pref}â'), content=f'{pref}î')
            normalized.replace(pattern=Regex(f'\\b{pref}-â'), content=f'{pref}-î')
            pref_uc = pref.upper()
            normalized.replace(pattern=Regex(f'\\b{pref_uc}Â'), content=f'{pref_uc}Î')
            normalized.replace(pattern=Regex(f'\\b{pref_uc}-Â'), content=f'{pref_uc}-Î')
        # end for

    def normalize_str(self, sequence: str) -> str:
        # 1. Remove spaces left and right
        sequence = sequence.strip()
        # 2. Use standard Romanian diacritics
        sequence = sequence.replace('ş', 'ș')
        sequence = sequence.replace('Ş', 'Ș')
        sequence = sequence.replace('ţ', 'ț')
        sequence = sequence.replace('Ţ', 'Ț')
        # 3. Remove consecutive spaces
        sequence = re.sub(pattern=r'\s+', repl=' ', string=sequence)
        # 4. Enforce correct forms for 'a fi'
        sequence = re.sub(pattern=r'\bsînt\b', repl='sunt', string=sequence)
        sequence = re.sub(pattern=r'\bSînt\b', repl='Sunt', string=sequence)
        sequence = re.sub(pattern=r'\bsîntem\b', repl='suntem', string=sequence)
        sequence = re.sub(pattern=r'\bSîntem\b', repl='Suntem', string=sequence)
        sequence = re.sub(pattern=r'\bsînteți\b', repl='sunteți', string=sequence)
        sequence = re.sub(pattern=r'\bSînteți\b', repl='Sunteți', string=sequence)
        # 5. Use 'â' everywhere...
        sequence = sequence.replace('î', 'â')
        sequence = sequence.replace('Î', 'Â')
        # 6. But not at start/end of words
        sequence = re.sub(pattern=r'â\b', repl='î', string=sequence)
        sequence = re.sub(pattern=r'Â\b', repl='Î', string=sequence)
        sequence = re.sub(pattern=r'\bâ', repl='î', string=sequence)
        sequence = re.sub(pattern=r'\bÂ', repl='Î', string=sequence)

        # 7. And not after morphological prefixes
        for pref in ro_morpho_prefixes:
            sequence = re.sub(pattern=f'\\b{pref}â', repl=f'{pref}î', string=sequence)
            sequence = re.sub(pattern=f'\\b{pref}-â', repl=f'{pref}-î', string=sequence)
            pref_uc = pref.upper()
            sequence = re.sub(pattern=f'\\b{pref_uc}Â', repl=f'{pref_uc}Î', string=sequence)
            sequence = re.sub(pattern=f'\\b{pref_uc}-Â', repl=f'{pref_uc}-Î', string=sequence)
        # end for

        return sequence
