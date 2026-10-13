import os
import argparse
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import uvicorn

# Request Schema
class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

app = FastAPI(title="My Custom AI API", description="REST API for serving custom fine-tuned LLM")

# Enable CORS for web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model & tokenizer
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_ai_model(base_model_id="Qwen/Qwen2.5-7B-Instruct", adapter_path="output/my_custom_model"):
    global model, tokenizer
    print("=" * 60)
    print("🚀 Initializing AI Model for API Server...")
    print(f"📌 Base Model: {base_model_id}")
    print(f"📌 Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"🔗 Attaching custom LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print("⚠️ No adapter found. Using base model directly.")

    model.eval()
    print("✅ Model loaded and ready for API requests!")
    print("=" * 60)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None
    }

@app.post("/api/chat")
def chat(request: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=(request.temperature > 0),
                pad_token_id=tokenizer.eos_token_id
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = full_output.split("assistant\n")[-1].strip() if "assistant\n" in full_output else full_output.strip()

        return {
            "prompt": prompt,
            "response": response_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static web directory if exists
web_dir = os.path.join(os.path.dirname(__file__), "web")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    def serve_web_ui():
        return FileResponse(os.path.join(web_dir, "index.html"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="output/my_custom_model")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_ai_model(args.base_model, args.adapter_path)
    uvicorn.run(app, host=args.host, port=args.port)
