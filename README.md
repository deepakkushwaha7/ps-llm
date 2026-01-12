# 🧠 Problem solving LLM – End-to-End Single-File Architecture (Colab)

This project demonstrates a **complete end-to-end pipeline** for building a **coding-focused Large Language Model (LLM)** using **one single Python file** in **Google Colab**.

The file contains **everything**:

* Dependency installation
* Dataset loading
* Data preprocessing
* Tokenization
* Model loading
* LoRA fine-tuning
* Model saving
* Inference-only interactive CLI

This design is intentional to make the full lifecycle easy to understand and reproduce.

---

## 📌 High-Level Architecture

```
┌────────────────────────────┐
│  Single Python File        │
│  (Colab Notebook Cell)     │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Dependency Installation    │
│ transformers, datasets,    │
│ peft, accelerate, bnb      │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Dataset Loading            │
│ deepmind/code_contests     │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Data Preprocessing         │
│ • Python-only filtering    │
│ • Prompt formatting        │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Tokenization + Labels      │
│ • input_ids                │
│ • attention_mask           │
│ • labels = input_ids       │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Base Model (4-bit)         │
│ DeepSeek-Coder 1.3B        │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ LoRA Fine-Tuning           │
│ • q_proj, v_proj adapters  │
│ • <1% trainable params     │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Hugging Face Trainer       │
│ • Causal LM objective      │
│ • FP16 + grad accumulation │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Saved Model Artifacts      │
│ /content/my-code-llm       │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Inference-Only CLI         │
│ • Coding prompts only      │
│ • Clean code output        │
└────────────────────────────┘
```

---

## 📂 File Structure

There is **only one file**:

```
ps.py   (or a single Colab cell)
```

This file contains **all steps in sequence**.

---

## 🧩 Detailed Architecture Explanation

### 1️⃣ Dependency Installation (Runtime Layer)

The script installs all required libraries at runtime to ensure it works in a **fresh Colab session**.

---

### 2️⃣ Dataset Layer

**Dataset:** `deepmind/code_contests`

* Competitive programming problems
* High-quality algorithmic solutions
* Language encoded using numeric IDs (`1 = Python`)

---

### 3️⃣ Data Preprocessing Layer

Each problem is converted into an instruction-style prompt:

```text
### Problem:
<problem description>

### Write a Python solution.

### Solution:
<ground truth python code>
```

Only Python solutions are retained.

---

### 4️⃣ Tokenization & Labeling

* Tokenizer reused from base model
* Max length: `512`
* Causal LM labels:

  ```python
  labels = input_ids
  ```

---

### 5️⃣ Base Model Layer

**Model:** `deepseek-ai/deepseek-coder-1.3b-base`

* Loaded in **4-bit quantized mode**
* Optimized for low VRAM usage

---

### 6️⃣ LoRA Fine-Tuning Layer

* Low-rank adapters on `q_proj` and `v_proj`
* ~0.23% trainable parameters
* Enables training on free Colab GPUs

---

### 7️⃣ Training Layer

* Hugging Face `Trainer`
* Mixed precision (`fp16`)
* Gradient accumulation for stability

---

### 8️⃣ Model Saving Layer

Artifacts saved to:

```
/content/my-code-llm
```

---

### 9️⃣ Inference-Only CLI Layer

* Activated after training
* Rejects non-coding input
* Outputs clean Python code only

---

## ▶️ How to Run

### Step 1: Open Google Colab

Go to: [https://colab.research.google.com](https://colab.research.google.com)
Create a **new notebook**.

---

### Step 2: Enable GPU

In Colab menu:

```
Runtime → Change runtime type → Hardware accelerator → GPU
```

---

### Step 3: Paste the Code

* Copy the **entire single-file script**
* Paste it into **one cell**
* Do **not split** into multiple files or cells

---

### Step 4: Run the Cell

Click **Run** and wait:

1. Dependencies install
2. Dataset downloads
3. Model loads
4. Training starts
5. Model is saved
6. CLI starts automatically

Training may take **30–60 minutes** depending on GPU availability.

---

### Step 5: Use the CLI

Once training finishes, you will see:

```
🟢 Input >
```

Example input:

```
Write a Python function to reverse a list
```

Type `exit` to quit the CLI.

---

## 🧪 Example Interaction

````
🟢 Input > Write a Python function to check if a number is prime

🔵 Output >
```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
````

---

## 🛠️ Troubleshooting

### ❌ Dataset size becomes 0

**Cause:** Python solutions were not detected correctly.
**Fix:** Ensure language ID `1` is used for Python during preprocessing.

---

### ❌ `num_samples should be a positive integer`

**Cause:** Training dataset is empty.
**Fix:** Print dataset length after filtering:

```python
print(len(dataset))
```

---

### ❌ Training is very slow

**Cause:** Free Colab GPU limitations.
**Fixes:**

* Reduce dataset size (e.g. 2000 samples)
* Reduce `num_train_epochs`
* Be patient (expected behavior)

---

### ❌ Inference is slow

**Cause:** Large model + Colab latency.
**Fixes:**

* Use greedy decoding (`do_sample=False`)
* Reduce `max_new_tokens`
* Avoid interactive loops for long sessions

---

### ❌ Output is garbage or repetitive

**Cause:** Over-generation or sampling.
**Fix:**

* Reduce `max_new_tokens`
* Disable sampling

---

### ❌ CUDA Out of Memory

**Cause:** GPU memory exceeded.
**Fixes:**

* Reduce dataset size
* Ensure 4-bit loading is enabled
* Restart runtime and rerun

---

## 🏁 Summary

This single-file project demonstrates **real LLM engineering**:

> Dataset → Preprocessing → Tokenization → Base Model → LoRA → Training → CLI Inference

It is designed for **learning, experimentation, and architectural understanding**, not production deployment.


