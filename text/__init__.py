import phonemizer
from text.symbols import symbols

global_phonemizer = phonemizer.backend.EspeakBackend(language='de', preserve_punctuation=True, with_stress=True)


symbols_to_id = {s:i for i, s in enumerate(symbols)}


def tokens_to_ids(phonemes):
    return [symbols_to_id[phon] for phon in phonemes]


def ids_to_tokens(ids):
    return [symbols[id] for id in ids]


def text_to_tokens(text):
    text_ipa = global_phonemizer.phonemize([text])[0]
    return text_ipa


def text_to_ids(text):
    text_ipa = text_to_tokens(text)
    ids = tokens_to_ids(text_ipa)
    return ids

