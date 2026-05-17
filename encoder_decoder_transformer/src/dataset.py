import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = tokenizer_tgt.token_to_id("[SOS]")
        self.eos_token = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_token = tokenizer_tgt.token_to_id("[PAD]")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        src_text = item["translation"][self.src_lang]
        tgt_text = item["translation"][self.tgt_lang]

        enc_tokens = self.tokenizer_src.encode(src_text).ids
        dec_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Filter out-of-vocabulary tokens
        enc_tokens = [t for t in enc_tokens if t < self.tokenizer_src.get_vocab_size()]
        dec_tokens = [t for t in dec_tokens if t < self.tokenizer_tgt.get_vocab_size()]

        # Truncate to fit within seq_len (leaving room for SOS + EOS)
        enc_tokens = enc_tokens[: self.seq_len - 2]
        dec_tokens = dec_tokens[: self.seq_len - 2]

        enc_pad = self.seq_len - len(enc_tokens) - 2
        dec_pad = self.seq_len - len(dec_tokens) - 1

        encoder_input = torch.tensor(
            [self.sos_token]
            + enc_tokens
            + [self.eos_token]
            + [self.pad_token] * enc_pad,
            dtype=torch.long,
        )
        decoder_input = torch.tensor(
            [self.sos_token] + dec_tokens + [self.pad_token] * dec_pad, dtype=torch.long
        )
        label = torch.tensor(
            dec_tokens + [self.eos_token] + [self.pad_token] * dec_pad, dtype=torch.long
        )

        assert encoder_input.size(0) == self.seq_len, (
            f"Encoder input length mismatch: {encoder_input.size(0)} != {self.seq_len}"
        )
        assert decoder_input.size(0) == self.seq_len, (
            f"Decoder input length mismatch: {decoder_input.size(0)} != {self.seq_len}"
        )
        assert label.size(0) == self.seq_len, (
            f"Label length mismatch: {label.size(0)} != {self.seq_len}"
        )

        encoder_mask = (
            (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, seq_len)
        decoder_pad_mask = (
            (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, seq_len)
        decoder_mask = decoder_pad_mask & causal_mask(
            self.seq_len
        )  # (1, seq_len, seq_len)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    return ~torch.triu(torch.ones((1, size, size), dtype=torch.bool), diagonal=1)
