import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Chat with your custom fine-tuned AI")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model ID on Hugging Face")
    parser.add_argument("--adapter_path", type=str, default="output/my_custom_model",
                        help="Path to trained LoRA adapter (leave empty to test base model)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to test (if not provided, launches interactive chat)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    return parser.parse_args()

def generate_response(model, tokenizer, prompt, max_new_tokens, temperature):
    system_msg = "You are a helpful, smart AI assistant. You understand and speak Arabic, French, and Algerian Darija (including Arabizi with numbers like 3, 7, 9)."
    formatted_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant\n" in full_output:
        return full_output.split("assistant\n")[-1].strip()
    return full_output.strip()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("🤖 Loading Custom AI Assistant...")
    print(f"📌 Base Model: {args.base_model}")
    print(f"📌 Device: {device}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization if CUDA
    bnb_config = None
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

    # Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )

    # Attach Adapter if it exists
    if args.adapter_path and os.path.exists(args.adapter_path):
        print(f"🔗 Attaching custom LoRA adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        print("⚠️ No adapter found. Using base model directly.")

    model.eval()
    print("✅ Model is ready!")
    print("=" * 60)

    # Single Prompt Mode
    if args.prompt:
        print(f"\n👤 User: {args.prompt}")
        response = generate_response(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature)
        print(f"🤖 Assistant: {response}\n")
        return

    # Interactive Chat Mode
    print("\n💬 Interactive Chat Session Started! (Type 'exit' or 'quit' to stop)\n")
    while True:
        try:
            user_input = input("👤 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Bye!")
                break
                
            response = generate_response(model, tokenizer, user_input, args.max_new_tokens, args.temperature)
            print(f"🤖 AI: {response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Chat ended.")
            break

if __name__ == "__main__":
    main()
