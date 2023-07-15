""" CLIP tokenizer

Copied from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/tokenizer.py
Copied from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import os

import html
from functools import lru_cache
from typing import Union, List

import ftfy
import regex as re
import numpy as np

# https://stackoverflow.com/q/62691279
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_SOT = "<|start_of_text|>"
DEFAULT_EOT = "<|end_of_text|>"

# optional keys: {"limit_vocab_size": None, "sot": DEFAULT_SOT, "eot": DEFAULT_EOT}
BUILDIN_TOKENIZERS = {
    # [???] limit_vocab_size = 49152-256-2 == 48894 for original bpe vocab, why this number?
    "bpe": {"path": "bpe_simple_vocab_16e6.txt", "file_hash": "f83f3e2479df59e7cd1597baa03f34c8", "limit_vocab_size": 48894},
    # "<|start_of_text|>" not exists in original gpt vocab, using sot == eot here
    "gpt2": {"path": "gpt2_tokenizer.txt", "file_hash": "150fdad3c88ee8f9607ac1808ad2d321", "sot": "<|end_of_text|>", "eot": "<|end_of_text|>"},
}


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word. get_pairs('hello') -> {('e', 'l'), ('h', 'e'), ('l', 'l'), ('l', 'o')}"""
    return set([(pre, cur) for pre, cur in zip(word[:-1], word[1:])])


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    return re.sub(r'\s+', ' ', text).strip()


class Tokenizer(object):
    def __init__(self, name_or_path="clip", special_tokens=None, limit_vocab_size="auto", context_length=77):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {vv: kk for kk, vv in self.byte_encoder.items()}
        byte_vocab = self._init_byte_vacab_()  # Different from gpt2 and clip

        tokens, self.sot, self.eot = self._init_tokenizer_from_file_(name_or_path, limit_vocab_size)
        tokens_split = [ii.split() for ii in tokens if len(ii) != 0]  # exclude empty one from gpt2
        self.bpe_ranks = {tuple(ii): id for id, ii in enumerate(tokens_split)}
        token_vocab = [''.join(ii) for ii in tokens_split]
        vocab = [ii for ii in byte_vocab if ii not in token_vocab] + token_vocab  # filter basic byte_vocab from provided token_vocab

        special_tokens = list(set([self.sot, self.eot])) + (special_tokens if special_tokens else [])
        self.cache = {t:t for t in special_tokens}
        special_regex = "|".join([ii.replace("|", "\|") for ii in special_tokens])
        self.pat = re.compile(special_regex + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        vocab.extend(special_tokens)

        self.decoder = dict(enumerate(vocab))
        self.encoder = {v: k for k, v in self.decoder.items()}
        self.vocab_size = len(vocab)

        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token_id, self.eot_token_id = self.encoder[self.sot], self.encoder[self.eot]
        self.context_length = context_length

    def _init_byte_vacab_(self):
        return list(self.byte_encoder.values()) + [vv + '</w>' for vv in self.byte_encoder.values()]

    def _init_tokenizer_from_file_(self, name_or_path, limit_vocab_size):
        name_or_path = name_or_path.lower()
        if name_or_path in BUILDIN_TOKENIZERS:
            from keras_cv_attention_models.backend import get_file

            config = BUILDIN_TOKENIZERS[name_or_path]
            path, file_hash, sot, eot = config["path"], config["file_hash"], config.get("sot", DEFAULT_SOT), config.get("eot", DEFAULT_EOT)
            limit_vocab_size = config.get("limit_vocab_size", None) if limit_vocab_size == "auto" else limit_vocab_size

            url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/assets/{}".format(path)
            tokenizer_file = os.path.join(os.path.expanduser("~/.keras/datasets"), path)
            print(">>>> Trying to tokenizer file:", tokenizer_file)
            tokenizer_file = get_file(origin=url, file_hash=file_hash)
        else:
            tokenizer_file = name_or_path
            limit_vocab_size = None if limit_vocab_size == "auto" else limit_vocab_size
            sot, eot = DEFAULT_SOT, DEFAULT_EOT

        if tokenizer_file.endswith(".gz") or tokenizer_file.endswith(".zip"):
            import gzip

            with gzip.open(tokenizer_file) as ff:
                tokens = ff.read().decode("utf-8").split('\n')
        else:
            with open(tokenizer_file) as ff:
                tokens = ff.read().split("\n")[1:]  # exclude first line
        if limit_vocab_size:
            tokens = tokens[:limit_vocab_size]
        return tokens, sot, eot

    def bpe(self, token, is_first_token=False, is_space_first=False):
        if token in self.cache:
            return self.cache[token]
        if is_space_first:
            word = tuple(token) if is_first_token else (('</w>' + token[0],) + tuple(token[1:]))  # gpt2 one
        else:
            word = tuple(token[:-1]) + ( token[-1] + '</w>',)  # clip one
        pairs = get_pairs(word)

        if not pairs:
            return ('</w>' + token) if is_space_first else (token + '</w>')

        cur = word
        while len(cur) > 1:
            # print(f"{pairs = }")
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            while len(cur) > 0:
                if first in cur:
                    next_index = cur.index(first)
                    new_word.extend(cur[:next_index])
                    cur = cur[next_index:]
                else:
                    new_word.extend(cur)
                    break

                if cur[0] == first and len(cur) > 1 and cur[1] == second:
                    new_word.append(first + second)
                    cur = cur[2:]
                else:
                    new_word.append(cur[0])
                    cur = cur[1:]
            cur = tuple(new_word)
            pairs = get_pairs(cur)

        word = ' '.join(tuple(new_word))
        # print(f"{word = }")
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # print(f"{token = }")
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def __call__(self, inputs):
        if isinstance(inputs, str) or isinstance(inputs, bytes):
            inputs = inputs.decode() if hasattr(inputs, "decode") else inputs
            tokens = [self.sot_token_id] + self.encode(inputs)[:self.context_length - 2] + [self.eot_token_id]
            return np.pad(tokens, [0, self.context_length - len(tokens)])
        else:
            inputs = inputs.detach() if hasattr(inputs, "detach") else inputs
            inputs = inputs.cpu() if hasattr(inputs, "cpu") else inputs
            inputs = inputs.numpy() if hasattr(inputs, "numpy") else inputs

            inputs = list(inputs)
            start = 1 if inputs[0] == self.sot_token_id else 0
            end = inputs.index(self.eot_token_id) if self.eot_token_id in inputs else None
            return self.decode(inputs[start:end])


class GPT2Tokenizer(Tokenizer):
    def _init_byte_vacab_(self):
        return list(self.byte_encoder.values())

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        is_first_token = True
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # print(f"{token = }")
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token, is_first_token=is_first_token, is_space_first=True).split(' '))
            is_first_token = False
        return bpe_tokens


class HuggingFaceTokenizer:
    """HuggingFace tokenizer wrapper"""

    def __init__(self, tokenizer_name: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.context_length = 77

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: int = 77):
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]
        texts = [whitespace_clean(basic_clean(text)) for text in texts]
        input_ids = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=context_length,
            padding='max_length',
            truncation=True,
        ).input_ids
        return input_ids
