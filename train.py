import os
import sys
import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an LLM on custom dataset using QLoRA with standard Trainer")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Hugging Face model ID (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--data_path", type=str, default="data/dataset.json",
                        help="Path to JSON dataset file")
    parser.add_argument("--output_dir", type=str, default="output/my_custom_model",
                        help="Directory to save trained LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    return parser.parse_args()

def prepare_tokenized_dataset(data_path, tokenizer, max_seq_length=512):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"📖 Loaded {len(data)} training examples from {data_path}")
    
    formatted_texts = []
    for item in data:
        instruction = item.get("instruction", "").strip()
        extra_input = item.get("input", "").strip()
        output = item.get("output", "").strip()
        
        user_message = f"{instruction}\n{extra_input}".strip() if extra_input else instruction
        text = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        formatted_texts.append(text)
        
    raw_dataset = Dataset.from_dict({"text": formatted_texts})

    def tokenize_fn(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    return tokenized_dataset

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("🚀 Starting Custom AI Fine-Tuning Pipeline (Standard Hugging Face Trainer)")
    print(f"📌 Base Model: {args.model_id}")
    print(f"📌 Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"📌 Output Directory: {args.output_dir}")
    print("=" * 60)

    # 1. Load Tokenizer
    print("⬇️ Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Configure 4-bit Quantization (QLoRA)
    bnb_config = None
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    # 3. Load Model
    print("⬇️ Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )

    if device == "cuda":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 4. LoRA Configuration
    print("🔧 Setting up LoRA Adapter...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Tokenize Dataset
    print("📂 Tokenizing dataset for Trainer...")
    train_dataset = prepare_tokenized_dataset(args.data_path, tokenizer, args.max_seq_length)

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True if device == "cuda" else False,
        gradient_checkpointing_kwargs={"use_reentrant": False} if device == "cuda" else None,
        learning_rate=args.learning_rate,
        fp16=(device == "cuda"),
        logging_steps=1,
        save_strategy="epoch",
        optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
        report_to="none"
    )

    # 7. Standard Trainer with DataCollatorForSeq2Seq (dynamic padding)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        args=training_args
    )

    # 8. Train
    print("\n🔥 Training in progress...")
    trainer.train()

    # 9. Save
    print(f"\n💾 Saving model adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 60)
    print("✅ Training completed successfully! 🎉")
    print(f"👉 You can now run chat: python chat.py --adapter_path {args.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
