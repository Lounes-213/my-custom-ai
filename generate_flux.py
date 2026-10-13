import os
import argparse
from huggingface_hub import InferenceClient

def main():
    parser = argparse.ArgumentParser(description="Generate ultra-realistic images with FLUX.1 [schnell]")
    parser.add_argument("--prompt", type=str, 
                        default="A majestic Algerian fennec fox in the golden sand dunes of the Sahara, cinematic lighting, 8k resolution, photorealistic", 
                        help="Image prompt")
    parser.add_argument("--output", type=str, default="flux_output.png", help="Output image file path")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN", ""), help="Hugging Face token")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError("Please provide a valid Hugging Face token via --token or HF_TOKEN env var")

    print("=" * 60)
    print("👑 FLUX.1 [schnell] Cloud Generation (Instant ~2s)")
    print(f"📌 Prompt: {args.prompt}")
    print("=" * 60)

    # Initialize client with fallback support
    image = None
    try:
        client = InferenceClient(provider="fal-ai", api_key=token)
        image = client.text_to_image(args.prompt, model="black-forest-labs/FLUX.1-schnell")
    except Exception as e:
        print(f"ℹ️ Provider fal-ai note: {e}. Switching to standard HF endpoint...")
        client = InferenceClient(api_key=token)
        image = client.text_to_image(args.prompt, model="black-forest-labs/FLUX.1-schnell")

    image.save(args.output)
    print(f"\n✅ Image generated and saved successfully to: {args.output} 🎉")

if __name__ == "__main__":
    main()
