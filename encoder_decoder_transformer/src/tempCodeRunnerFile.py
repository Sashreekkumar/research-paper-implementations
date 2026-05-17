from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from tqdm import tqdm
import warnings
import os

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.tensor([[sos_idx]], dtype=source.dtype, device=device)

    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        _, next_word = torch.max(model.project(out[:, -1]), dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.tensor([[next_word.item()]], dtype=source.dtype, device=device),
            ],
            dim=1,
        )
        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_ds,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_step,
    writer,
    loss_fn,
    num_examples=2,
):
    model.eval()
    count = 0
    source_texts, expected, predicted = [], [], []
    total_val_loss = 0

    try:
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except Exception:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            assert encoder_input.size(0) == 1, "Validation batch size must be 1"

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            proj_output = model.project(decoder_output)

            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            total_val_loss += loss.item()

            model_out = greedy_decode(
                model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device
            )
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(batch["src_text"][0])
            expected.append(batch["tgt_text"][0])
            predicted.append(model_out_text)

            print_msg("-" * console_width)
            print_msg(f"{'SOURCE:':>12} {batch['src_text'][0]}")
            print_msg(f"{'TARGET:':>12} {batch['tgt_text'][0]}")
            print_msg(f"{'PREDICTED:':>12} {model_out_text}")

            if count == num_examples:
                print_msg("-" * console_width)
                break

    if writer:
        expected_bleu = [[text] for text in expected]
        metrics = {
            "validation loss": total_val_loss / count,
            "validation CER": torchmetrics.CharErrorRate()(predicted, expected),
            "validation WER": torchmetrics.WordErrorRate()(predicted, expected),
            "validation BLEU": torchmetrics.BLEUScore()(predicted, expected_bleu),
        }
        for name, value in metrics.items():
            writer.add_scalar(name, value, global_step)
        writer.flush()


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(
            (item["translation"][lang] for item in ds), trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset(
        config["datasource"],
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
    )

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    train_size = int(0.9 * len(ds_raw))
    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_size, len(ds_raw) - train_size]
    )

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def train_model(config):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    device = torch.device(device)

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(
        parents=True, exist_ok=True
    )

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    ).to(device)

    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    initial_epoch, global_step = 0, 0
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(config)
        if preload == "latest"
        else get_weights_file_path(config, preload)
        if preload
        else None
    )

    if model_filename:
        print(f"Loading model: {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
    else:
        print("No preload model found, training from scratch.")

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            proj_output = model.project(decoder_output)

            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            writer.add_scalar("gradient norm", grad_norm.item(), global_step)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        run_validation(
            model=model,
            validation_ds=val_dataloader,
            tokenizer_tgt=tokenizer_tgt,
            max_len=config["seq_len"],
            device=device,
            print_msg=lambda msg: batch_iterator.write(msg),
            global_step=global_step,
            writer=writer,
            loss_fn=loss_fn,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            get_weights_file_path(config, f"{epoch:02d}"),
        )

    writer.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_model(get_config())
