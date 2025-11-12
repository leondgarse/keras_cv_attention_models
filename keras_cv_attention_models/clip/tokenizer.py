"""CLIP tokenizer

Copied from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/tokenizer.py
Copied from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
import html
from functools import lru_cache
import numpy as np


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # https://stackoverflow.com/q/62691279

DEFAULT_SOT = "<|startoftext|>"
DEFAULT_EOT = "<|endoftext|>"

# optional keys: {"limit_vocab_size": None, "sot": DEFAULT_SOT, "eot": DEFAULT_EOT, "is_space_first": False}
BUILDIN_TOKENIZERS = {
    # [???] limit_vocab_size = 49152-256-2 == 48894 for original bpe vocab, why this number?
    "clip": {"path": "bpe_simple_vocab_16e6.txt", "file_hash": "f83f3e2479df59e7cd1597baa03f34c8", "limit_vocab_size": 48894},
    # "<|start_of_text|>" not exists in original gpt vocab, using sot == eot here
    "gpt2": {"path": "gpt2_tokenizer.txt", "file_hash": "150fdad3c88ee8f9607ac1808ad2d321", "sot": DEFAULT_EOT, "eot": DEFAULT_EOT, "is_space_first": True},
    "llama": {"path": "llama_tokenizer.model", "file_hash": "eeec4125e9c7560836b4873b6f8e3025", "sot": "<s>", "eot": "</s>", "is_space_first": True},
    "llama2": {"path": "llama_tokenizer.model", "file_hash": "eeec4125e9c7560836b4873b6f8e3025", "sot": "<s>", "eot": "</s>", "is_space_first": True},
}


def download_tokenizer_file(name):
    from keras_cv_attention_models.backend import get_file

    config = BUILDIN_TOKENIZERS[name]
    path, file_hash = config["path"], config["file_hash"]

    url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/assets/{}".format(path)
    tokenizer_file = os.path.join(os.path.expanduser("~"), ".keras", "datasets", path)
    print(">>>> Load tokenizer from file:", tokenizer_file)
    return get_file(origin=url, file_hash=file_hash)


class SimpleTokenizer(object):
    def __init__(self, name_or_path="clip", special_tokens=None, limit_vocab_size="auto", context_length=77):
        import ftfy  # fixes text for you, importing here avoiding additional requirements if not needed
        import regex

        self.ftfy, self.regex = ftfy, regex

        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {vv: kk for kk, vv in self.byte_encoder.items()}
        byte_vocab = self._init_byte_vacab_()  # Different from gpt2 and clip

        tokens, self.sot, self.eot, self.is_space_first = self._init_tokenizer_from_file_(name_or_path, limit_vocab_size)
        tokens_split = [ii.split() for ii in tokens if len(ii) != 0]  # exclude empty one from gpt2
        self.bpe_ranks = {tuple(ii): id for id, ii in enumerate(tokens_split)}
        token_vocab = ["".join(ii) for ii in tokens_split]
        vocab = [ii for ii in byte_vocab if ii not in token_vocab] + token_vocab  # filter basic byte_vocab from provided token_vocab

        special_tokens = ([self.sot] if self.sot == self.eot else [self.sot, self.eot]) + (special_tokens if special_tokens else [])
        self.cache = {t: t for t in special_tokens}
        special_regex = "|".join([ii.replace("|", r"\|") for ii in special_tokens])
        self.pat = regex.compile(special_regex + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", regex.IGNORECASE)
        vocab.extend(special_tokens)

        self.decoder = dict(enumerate(vocab))
        self.encoder = {v: k for k, v in self.decoder.items()}
        self.vocab_size = len(vocab)

        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token, self.eot_token = self.encoder[self.sot], self.encoder[self.eot]
        self.context_length = context_length

    def _init_byte_vacab_(self):
        return list(self.byte_encoder.values()) + [vv + "</w>" for vv in self.byte_encoder.values()]

    def _init_tokenizer_from_file_(self, name_or_path, limit_vocab_size):
        name_or_path = name_or_path.lower()
        if name_or_path in BUILDIN_TOKENIZERS:
            tokenizer_file = download_tokenizer_file(name_or_path)

            config = BUILDIN_TOKENIZERS[name_or_path]
            sot, eot, is_space_first = config.get("sot", DEFAULT_SOT), config.get("eot", DEFAULT_EOT), config.get("is_space_first", False)
            limit_vocab_size = config.get("limit_vocab_size", None) if limit_vocab_size == "auto" else limit_vocab_size
        else:
            tokenizer_file = name_or_path
            limit_vocab_size = None if limit_vocab_size == "auto" else limit_vocab_size
            sot, eot, is_space_first = DEFAULT_SOT, DEFAULT_EOT, False

        if tokenizer_file.endswith(".gz") or tokenizer_file.endswith(".zip"):
            import gzip

            with gzip.open(tokenizer_file) as ff:
                tokens = ff.read().decode("utf-8").split("\n")
        else:
            with open(tokenizer_file) as ff:
                tokens = ff.read().split("\n")[1:]  # exclude first line
        if limit_vocab_size:
            tokens = tokens[:limit_vocab_size]
        return tokens, sot, eot, is_space_first

    @staticmethod
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
        # Without [0, 32], [127, 160]
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        additional = [b for b in range(2**8) if b not in bs]
        cs = bs + [ii for ii in range(2**8, 2**8 + len(additional))]
        return dict(zip(bs + additional, [chr(ii) for ii in cs]))

    @staticmethod
    def _get_pairs_(word):
        """Return set of symbol pairs in a word. get_pairs('hello') -> {('e', 'l'), ('h', 'e'), ('l', 'l'), ('l', 'o')}"""
        return set([(pre, cur) for pre, cur in zip(word[:-1], word[1:])])

    def bpe(self, token, is_first_token=False):
        token_with_space = (token if is_first_token else ("</w>" + token)) if self.is_space_first else (token + "</w>")
        if token_with_space in self.encoder:
            # self.cache[token_with_space] = token_with_space
            return token_with_space
        if token_with_space in self.cache:
            return self.cache[token_with_space]

        if self.is_space_first:
            word = tuple(token) if is_first_token else (("</w>" + token[0],) + tuple(token[1:]))  # gpt2 one
        else:
            word = tuple(token[:-1]) + (token[-1] + "</w>",)  # clip one

        pairs = self._get_pairs_(word)
        if not pairs:
            return token_with_space

        # print(f"{pairs = }")
        cur = word
        while len(cur) > 1:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
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
            pairs = self._get_pairs_(cur)

        word = " ".join(cur)
        # print(f"{word = }")
        self.cache[token] = word
        return word

    def text_clean(self, text):
        text = self.ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return self.regex.sub(r"\s+", " ", text.strip()).strip()

    def encode(self, text, add_sot=False, add_eot=False):
        bpe_tokens = []
        text = self.text_clean(text).lower()
        is_first_token = True
        for token in self.regex.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            # print(f"{token = }")
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token, is_first_token=is_first_token).split(" "))
            is_first_token = False

        if add_sot:
            bpe_tokens = [self.sot_token] + bpe_tokens
        if add_eot:
            bpe_tokens = bpe_tokens + [self.sot_token]
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace").replace("</w>", " ")
        return text

    def __call__(self, inputs, padding_value=0):
        if isinstance(inputs, str) or isinstance(inputs, bytes):
            inputs = inputs.decode() if hasattr(inputs, "decode") else inputs
            tokens = [self.sot_token] + self.encode(inputs)[: self.context_length - 2] + [self.eot_token]
            # print(f"{tokens = }")
            return np.pad(tokens, [0, self.context_length - len(tokens)], constant_values=padding_value)
        else:
            inputs = inputs.detach() if hasattr(inputs, "detach") else inputs
            inputs = inputs.cpu() if hasattr(inputs, "cpu") else inputs
            inputs = inputs.numpy() if hasattr(inputs, "numpy") else inputs
            inputs = inputs.tolist() if hasattr(inputs, "tolist") else list(inputs)

            inputs = inputs[1 if inputs[0] == self.sot_token else 0 :]
            inputs = inputs[: inputs.index(self.eot_token) if self.eot_token in inputs else None]
            return self.decode(inputs)


class GPT2Tokenizer(SimpleTokenizer):
    def __init__(self, name_or_path="gpt2", special_tokens=None, limit_vocab_size="auto", context_length=77):
        super().__init__(name_or_path, special_tokens, limit_vocab_size, context_length)

    def _init_byte_vacab_(self):
        return list(self.byte_encoder.values())


class SentencePieceTokenizer(SimpleTokenizer):
    """
    >>> from keras_cv_attention_models.clip import tokenizer
    >>> ee = tokenizer.SentencePieceTokenizer()
    >>> print(ee(ee('hello world')))
    >>> #  hello world  <- An additional leading white space
    >>> print(ee.decode(ee.encode('hello world', add_sot=True)))
    >>> # hello world
    """

    def __init__(self, name_or_path="llama", context_length=77):
        from sentencepiece import SentencePieceProcessor

        self.name_or_path = download_tokenizer_file(name_or_path) if name_or_path in BUILDIN_TOKENIZERS else name_or_path
        self.sp_model = SentencePieceProcessor(model_file=self.name_or_path)
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
        self.decoder, self.scores = self._build_vocab_dict_()

        # BOS / EOS token IDs
        self.vocab_size = self.sp_model.vocab_size()
        self.sot_token, self.eot_token = self.sp_model.bos_id(), self.sp_model.eos_id()
        self.pad_token = self.sp_model.pad_id()

        self.sot = self.decoder[self.sot_token]
        self.eot = self.decoder[self.eot_token]
        self.context_length = context_length

    def _build_vocab_dict_(self):
        # https://github.com/karpathy/llama2.c/blob/master/tokenizer.py#L42-L61
        decoder, scores = {}, {}
        for id in range(self.sp_model.vocab_size()):
            token = self.sp_model.id_to_piece(id)
            score = self.sp_model.get_score(id)
            if id == self.sp_model.bos_id():
                token = "\n" + token + "\n"
            elif id == self.sp_model.eos_id():
                token = "\n" + token + "\n"
            elif len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
                token = chr(int(token[3:5], 16))  # e.g. make '<0x01>' into '\x01'
            token = token.replace("▁", " ")  # sentencepiece uses this character as whitespace
            decoder[id] = token  # .encode('utf-8') # bytes of this token, utf-8 encoded
            scores[id] = score
        return decoder, scores

    def encode(self, text, add_sot=False, add_eot=False):
        tokens = self.sp_model.encode(text)
        if add_sot:
            tokens = [self.sot_token] + tokens
        if add_eot:
            tokens = tokens + [self.sot_token]
        return tokens

    # def decode_single(self, token):
    #     return self.decoder[int(tokens)]

    def decode(self, tokens):
        # Not using sp_model.decode, as leading space is omitted when decoding a single word. https://github.com/karpathy/llama2.c/pull/89
        # return self.sp_model.decode(tokens)
        tokens = tokens.tolist() if hasattr(tokens, "tolist") else tokens
        if isinstance(tokens, (list, tuple)) and len(tokens) > 1:
            return self.sp_model.decode(tokens)
            # rr = "".join([self.decoder[int(ii)] for ii in tokens])
            # return (self.sot + rr[len(self.sot) + 1 :]) if tokens[0] == self.sot_token else rr
        else:
            token = int(tokens[0]) if isinstance(tokens, (list, tuple)) else int(tokens)
            return self.decoder[token]


class TikToken(SimpleTokenizer):
    """OpenAI  tiktoken wrapper"""

    def __init__(self, encoding_name="gpt2", context_length=77):
        import tiktoken
        import ftfy  # fixes text for you, importing here avoiding additional requirements if not needed
        import regex

        self.ftfy, self.regex = ftfy, regex

        if encoding_name not in tiktoken.list_encoding_names():
            raise ValueError("[Error] encoding_name should be one of {}".format(tiktoken.list_encoding_names()))

        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.sot = self.eot = "<|endoftext|>"
        self.sot_token = self.eot_token = self.tokenizer.encode(self.eot, allowed_special={self.eot})[0]
        self.vocab_size = self.tokenizer.n_vocab
        self.name = self.encoding_name = encoding_name

        self.context_length = context_length

    def encode(self, text, add_sot=False, add_eot=False):
        text = self.text_clean(text).lower()
        tokens = self.tokenizer.encode(text)

        if add_sot:
            tokens = [self.sot_token] + tokens
        if add_eot:
            tokens = tokens + [self.sot_token]
        return tokens

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


class HuggingFaceTokenizer:
    """HuggingFace tokenizer wrapper"""

    def __init__(self, tokenizer_name: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.context_length = 77

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts, context_length: int = 77):
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]
        texts = [whitespace_clean(basic_clean(text)) for text in texts]
        input_ids = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=context_length,
            padding="max_length",
            truncation=True,
        ).input_ids
        return input_ids
