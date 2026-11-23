"""Run grouped out-of-fold QLoRA or fit the final development adapter."""

import argparse
import gc
import hashlib
from inspect import signature
import json
import math
from pathlib import Path
import tempfile


MODEL = "unsloth/Qwen2.5-3B-Instruct"
MAX_SEQ_LEN = 1024
SEED = 42
TRAIN_OPTIONS = {
    "dataset_text_field": "text",
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "warmup_ratio": 0.05,
    "num_train_epochs": 1,
    "learning_rate": 2e-4,
    "bf16": True,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "logging_steps": 20,
    "save_strategy": "epoch",
    "seed": SEED,
}


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_dataset(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "dev.jsonl"
    if file_sha256(path) != manifest["dataset_sha256"]:
        raise ValueError("Development JSONL does not match its manifest digest")

    folds = manifest["folds"]
    dev_runs = [run_id for _, run_id in manifest["dev_assignments"]]
    if len(dev_runs) != 18 or len(set(dev_runs)) != 18 or len(folds) != 3 or any(len(group) != 6 for group in folds):
        raise ValueError("Expected 18 distinct development runs in three folds of six")
    membership = {run_id: fold for fold, group in enumerate(folds) for run_id in group}
    if set(membership) != set(dev_runs):
        raise ValueError("Fold membership does not match the development assignments")
    rows = []
    keys = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["run_id"], row["cutoff_utc"]
            if (
                key in keys or row["run_id"] not in membership or membership[row["run_id"]] != row["fold"]
                or row["label"] not in ("yes", "no") or not isinstance(row["prompt"], str)
            ):
                raise ValueError(f"Invalid or duplicate development example: {key}")
            keys.add(key)
            rows.append(row)
    if {row["fold"] for row in rows} != {0, 1, 2}:
        raise ValueError("Every development fold must contain eligible examples")
    return manifest, rows


def chat_message(prompt, answer=None):
    messages = [{"role": "user", "content": prompt}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def train_model(rows, validation, output_dir):
    # Import in this order so Unsloth can patch Transformers and TRL.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    def dataset(items):
        records = []
        for row in items:
            rendered = tokenizer.apply_chat_template(chat_message(row["prompt"], row["label"]), tokenize=False)
            if len(tokenizer.encode(rendered, add_special_tokens=False)) > MAX_SEQ_LEN:
                raise ValueError("An example exceeds the 1024-token protocol context")
            records.append({"text": rendered})
        return Dataset.from_list(records)

    config = dict(TRAIN_OPTIONS, eval_strategy="epoch" if validation else "no", output_dir=str(output_dir))
    length_key = "max_length" if "max_length" in signature(SFTConfig).parameters else "max_seq_length"
    config[length_key] = MAX_SEQ_LEN
    tokenizer_key = "processing_class" if "processing_class" in signature(SFTTrainer).parameters else "tokenizer"
    trainer = SFTTrainer(
        model=model, train_dataset=dataset(rows), eval_dataset=dataset(validation) if validation else None,
        args=SFTConfig(**config), **{tokenizer_key: tokenizer},
    )
    trainer = train_on_responses_only(
        trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n",
    )
    trainer.train()
    return model, tokenizer


def make_predictor(model, tokenizer):
    from unsloth import FastLanguageModel
    import torch

    FastLanguageModel.for_inference(model)
    model.eval()
    device = model.get_input_embeddings().weight.device

    def predict(prompt):
        messages = chat_message(prompt)
        prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        context = tokenizer.encode(prefix, add_special_tokens=False)
        options = [tokenizer.encode(prefix + word, add_special_tokens=False) for word in ("no", "yes")]
        if not context or any(option[:len(context)] != context or len(option) == len(context) for option in options):
            raise ValueError("The chat template does not give a stable answer-token boundary")
        if any(len(option) > MAX_SEQ_LEN for option in options):
            raise ValueError("An inference example exceeds the 1024-token protocol context")

        with torch.inference_mode():
            if all(len(option) == len(context) + 1 for option in options):
                token_ids = torch.tensor([context], device=device)
                logits = model(input_ids=token_ids).logits[0, -1].float()
                scores = logits[[option[-1] for option in options]]
            else:
                scores = []
                for option in options:
                    token_ids = torch.tensor([option[:-1]], device=device)
                    suffix = model(input_ids=token_ids).logits[0, len(context) - 1:].float()
                    labels = torch.tensor(option[len(context):], device=device).view(-1, 1)
                    scores.append(suffix.log_softmax(dim=-1).gather(1, labels).sum())
                scores = torch.stack(scores)
            probability = scores.softmax(dim=0)[1].item()
        if not math.isfinite(probability):
            raise ValueError("Model returned a non-finite answer probability")
        return probability

    return predict


def adapter_digests(directory):
    return {
        path.name: file_sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }


def validate_adapter(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    actual = adapter_digests(directory)
    if (
        manifest.get("mode") != "final" or manifest.get("model") != MODEL
        or manifest.get("adapter_sha256") != actual or "adapter_config.json" not in actual
        or not any(name in actual for name in ("adapter_model.safetensors", "adapter_model.bin"))
    ):
        raise ValueError("Final adapter files do not match their training manifest")
    return manifest


def load_predictor(adapter_dir):
    validate_adapter(adapter_dir)
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir), max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True,
    )
    return make_predictor(model, tokenizer)


def write_oof_predictions(rows, path, scratch):
    with path.open("w", encoding="utf-8") as stream:
        for fold in range(3):
            training = [row for row in rows if row["fold"] != fold]
            validation = [row for row in rows if row["fold"] == fold]
            print(f"Fold {fold}: training on {len(training)} rows, scoring {len(validation)} held-back rows", flush=True)
            with tempfile.TemporaryDirectory(prefix=f"fold-{fold}-", dir=scratch) as checkpoints:
                model, tokenizer = train_model(training, validation, Path(checkpoints))
                predict = make_predictor(model, tokenizer)
                for row in validation:
                    result = {key: row[key] for key in ("run_id", "cutoff_utc", "fold")}
                    result["probability"] = predict(row["prompt"])
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                del predict, model, tokenizer
            gc.collect()
            import torch
            torch.cuda.empty_cache()


def train(mode, dataset_dir, output_dir):
    from llm_arm.protocol_data import verify_v32_freeze

    freeze = verify_v32_freeze(Path(__file__).resolve().parents[1] / "protocol")
    manifest, rows = read_dataset(dataset_dir)
    if manifest.get("freeze_commit") != freeze:
        raise ValueError("Development dataset does not match the current v3.2 freeze")
    if output_dir.exists():
        raise ValueError(f"Training output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".llm-training-", dir=output_dir.parent) as temporary:
        scratch = Path(temporary)
        staging = scratch / "artifact"
        staging.mkdir()
        report = {**manifest, "mode": mode, "model": MODEL}
        if mode == "cv":
            predictions = staging / "oof_predictions.jsonl"
            write_oof_predictions(rows, predictions, scratch)
            report["predictions_sha256"] = file_sha256(predictions)
        else:
            model, tokenizer = train_model(rows, [], scratch / "checkpoints")
            model.save_pretrained(staging)
            tokenizer.save_pretrained(staging)
            report["adapter_sha256"] = adapter_digests(staging)
        (staging / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staging.rename(output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cv", "final"))
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train(args.mode, args.dataset_dir, args.output_dir)
    print(f"Saved {args.mode} training artifact to {args.output_dir}")


if __name__ == "__main__":
    main()
