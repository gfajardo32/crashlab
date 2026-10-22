"""
train_llm.py
Fine-tune a small LM (Qwen2.5-3B-Instruct) to predict crashes from event-log
windows, using QLoRA. Runs on a CUDA GPU (your A100), NOT on a Mac or CPU.

Pipeline: load model in 4-bit -> attach LoRA adapters -> train on train.jsonl
(watching eval.jsonl) -> save the adapter.

On HPC, once: pip install unsloth
Then: python train_llm.py   (inside a GPU job)
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL = "unsloth/Qwen2.5-3B-Instruct"
MAX_SEQ_LEN = 1024          # our prompts are short; this is comfortable headroom
R = 16                      # LoRA rank (the "8" from the explanation, a bit bigger)
LORA_ALPHA = 16             # LoRA scaling
EPOCHS = 3                  # small dataset; watch eval loss to pick the real number
LR = 2e-4                   # standard LoRA learning rate
SEED = 42

# ---------- 1. load the base model in 4-bit (the "Q" in QLoRA) ----------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,      # quantize the frozen base to 4-bit -> tiny memory
    dtype=None,             # auto: bf16 on Ampere (your A100)
)

# ---------- 2. attach the LoRA adapters (the trainable ~1%) ----------
model = FastLanguageModel.get_peft_model(
    model,
    r=R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    # which matrices get an adapter: attention (q,k,v,o) + the MLP projections
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",   # extra memory savings
    random_state=SEED,
)

# ---------- 3. load + format the data into the model's chat template ----------
def to_text(ex):
    msgs = [
        {"role": "user", "content": ex["prompt"]},
        {"role": "assistant", "content": ex["completion"].strip()},  # "yes"/"no"
    ]
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

train_ds = load_dataset("json", data_files="llm_data/train.jsonl", split="train").map(to_text)
eval_ds = load_dataset("json", data_files="llm_data/eval.jsonl", split="train").map(to_text)

# ---------- 4. the trainer ----------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,      # effective batch = 8 * 2 = 16
        warmup_ratio=0.05,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        bf16=True,
        optim="adamw_8bit",                 # 8-bit optimizer, more memory savings
        weight_decay=0.01,
        lr_scheduler_type="linear",
        logging_steps=20,
        eval_strategy="epoch",              # check eval loss after each epoch
        save_strategy="epoch",              # keep a checkpoint each epoch
        seed=SEED,
        output_dir="outputs",
    ),
)

# train the loss ONLY on the model's answer (yes/no), not on the prompt text,
# so it doesn't waste capacity learning to parrot the events back.
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# ---------- 5. train and save just the adapter ----------
trainer.train()
model.save_pretrained("crash_lora")         # a few MB, not the whole 3B model
tokenizer.save_pretrained("crash_lora")
print("done -> adapter saved to crash_lora/")
