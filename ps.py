# =========================
# 0. INSTALL DEPENDENCIES
# =========================
!pip install -q transformers datasets peft accelerate bitsandbytes

# =========================
# 1. IMPORTS
# =========================
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import re

print("GPU:", torch.cuda.get_device_name(0))

# =========================
# 2. LOAD DATASET
# =========================
ds = load_dataset("deepmind/code_contests")

# =========================
# 3. FORMAT DATA (PYTHON ONLY)
# language id == 1 → Python
# =========================
def format_example(example):
    solutions = example["solutions"]
    python_solution = None

    if isinstance(solutions, dict):
        langs = solutions.get("language", [])
        codes = solutions.get("solution", [])
        for lang, code in zip(langs, codes):
            if lang == 1:  # Python
                python_solution = code
                break

    if not python_solution:
        return {"text": ""}

    return {
        "text": (
            "### Problem:\n"
            f"{example['description']}\n\n"
            "### Write a Python solution.\n\n"
            "### Solution:\n"
            f"{python_solution}"
        )
    }

dataset = ds["train"].map(format_example)
dataset = dataset.filter(lambda x: x["text"].strip() != "")
dataset = dataset.shuffle(seed=42).select(range(3000))

print("Training samples:", len(dataset))
print(dataset[0]["text"][:300])

# =========================
# 4. TOKENIZER + LABELS
# =========================
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
MAX_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_ds = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset.column_names
)

# =========================
# 5. LOAD MODEL (4-BIT)
# =========================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto"
)

# =========================
# 6. APPLY LoRA
# =========================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# 7. TRAINING
# =========================
training_args = TrainingArguments(
    output_dir="/content/my-code-llm",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=1,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
)

trainer.train()

trainer.save_model("/content/my-code-llm")
tokenizer.save_pretrained("/content/my-code-llm")

print("\n✅ TRAINING COMPLETE")

# =========================================================
# 8. FINAL BEAUTIFUL INFERENCE-ONLY CLI (NO RETRAINING)
# =========================================================
print("\n" + "═"*70)
print("🧠  CODING LLM CLI (INFERENCE ONLY)")
print("✍️  Ask coding questions | type 'exit' to quit")
print("═"*70 + "\n")

def is_code_prompt(text):
    keywords = [
        "code", "function", "python", "algorithm",
        "write", "implement", "solve", "program"
    ]
    return any(k in text.lower() for k in keywords)

def extract_code(text):
    code_lines = []
    for line in text.splitlines():
        if (
            line.strip().startswith((
                "def ", "for ", "while ", "if ", "elif ",
                "else:", "return", "print", "import", "class "
            ))
            or line.startswith((" ", "\t"))
        ):
            code_lines.append(line)
    return "\n".join(code_lines).strip()

model.eval()

while True:
    user_prompt = input("🟢 Input > ")

    if user_prompt.strip().lower() in {"exit", "quit"}:
        print("\n👋 Bye! Happy coding.\n")
        break

    if not is_code_prompt(user_prompt):
        print("\n🔴 Output >")
        print("```")
        print("Not found")
        print("```")
        print()
        continue

    prompt = (
        "### Problem:\n"
        f"{user_prompt}\n\n"
        "### Write a Python solution.\n\n"
        "### Solution:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    code = extract_code(decoded)

    print("\n🔵 Output >")
    print("```python")
    print(code if code else "Not found")
    print("```")
    print()
