# Fine-Tuning Large Language Models (LLMs) with Hugging Face

Welcome to the repository for **Fine-Tuning Large Language Models (LLMs)** using **Hugging Face Transformers** and **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA** (Low-Rank Adaptation). This guide demonstrates the steps to fine-tune a **LLaMA** model to create a customized, domain-specific language model, optimized for tasks like answering questions about medical terminology. By leveraging Hugging Face’s tools, this model can efficiently generate relevant and contextual responses in a conversational format.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Step-by-Step Guide](#step-by-step-guide)
- [Example Usage](#example-usage)
- [References](#references)
- [License](#license)

## Project Overview
This project walks through fine-tuning a large language model using Hugging Face’s ecosystem, specifically with:
- **Quantized Model**: Reduces memory usage with 4-bit quantization.
- **LoRA Fine-Tuning**: Parameter-efficient tuning method that allows effective adaptation with reduced computation.
- **Customized Dataset**: Fine-tunes the model on a medical terminology dataset for precise and relevant response generation.

## Requirements
- **Python 3.7+**
- **pip** (Python package manager)

### Required Libraries
Install all dependencies with the following command:
```bash
pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 huggingface_hub
```

## Setup and Installation
1. Clone this repository.
2. Install the dependencies using the command above.

## Step-by-Step Guide

### Step 1: Import Libraries and Install Dependencies
Install and import libraries essential for model fine-tuning, quantization, and accelerating the training process.

### Step 2: Load the Pre-trained Model
We use a quantized **LLaMA** model from Hugging Face’s model hub, set up for 4-bit computation to optimize GPU memory use. Here’s how:
   ```python
   from transformers import AutoModelForCausalLM, BitsAndBytesConfig
   import torch

   llama_model = AutoModelForCausalLM.from_pretrained(
       "aboonaji/llama2finetune-v2",
       quantization_config=BitsAndBytesConfig(
           load_in_4bit=True, 
           bnb_4bit_compute_dtype=torch.float16, 
           bnb_4bit_quant_type="nf4"
       )
   )
   llama_model.config.use_cache = False
   llama_model.config.pretraining_tp = 1
   ```

### Step 3: Load and Configure the Tokenizer
Load the tokenizer and set padding options to ensure input compatibility with the model:
   ```python
   from transformers import AutoTokenizer

   llama_tokenizer = AutoTokenizer.from_pretrained("aboonaji/llama2finetune-v2", trust_remote_code=True)
   llama_tokenizer.pad_token = llama_tokenizer.eos_token
   llama_tokenizer.padding_side = "right"
   ```

### Step 4: Define Training Arguments
Specify training configuration with `TrainingArguments` to manage batch size, output location, and training duration:
   ```python
   from transformers import TrainingArguments

   training_arguments = TrainingArguments(
       output_dir="./results",
       per_device_train_batch_size=4,
       max_steps=100
   )
   ```

### Step 5: Create the Fine-Tuning Trainer
We use the **SFTTrainer** class from Hugging Face’s `trl` library, integrating **LoRA** for efficient parameter adaptation:
   ```python
   from trl import SFTTrainer
   from peft import LoraConfig
   from datasets import load_dataset

   llama_sft_trainer = SFTTrainer(
       model=llama_model,
       args=training_arguments,
       train_dataset=load_dataset("aboonaji/wiki_medical_terms_llam2_format", split="train"),
       tokenizer=llama_tokenizer,
       peft_config=LoraConfig(task_type="CAUSAL_LM", r=64, lora_alpha=16, lora_dropout=0.1),
       dataset_text_field="text"
   )
   ```

### Step 6: Start Training
Run the fine-tuning process:
   ```python
   llama_sft_trainer.train()
   ```

### Step 7: Chat with the Model
After training, test the model's responses with a pipeline set up for text generation:
   ```python
   from transformers import pipeline

   text_generation_pipeline = pipeline(
       task="text-generation", model=llama_model, tokenizer=llama_tokenizer, max_length=300
   )
   user_prompt = "Please tell me about Bursitis"
   model_answer = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")
   print(model_answer[0]['generated_text'])
   ```

## Example Usage
Once the model is fine-tuned, it can generate domain-specific responses. Here’s an example:
```python
user_prompt = "Please tell me about Paracetamol"
model_answer = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")
print(model_answer[0]['generated_text'])
```

## References
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685)

## License
This repository is licensed under the MIT License. Please refer to the `LICENSE` file for more details.
