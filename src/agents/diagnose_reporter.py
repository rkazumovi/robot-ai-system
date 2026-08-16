"""One-time diagnostic: verify the AutoModelForSeq2SeqLM generation path works."""
import traceback

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print("transformers imported OK")
except ImportError as e:
    print(f"IMPORT FAILED: {e}")
    raise SystemExit(1)

try:
    print("Loading tokenizer + model (google/flan-t5-small)...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    print("Loaded OK")

    prompt = "Write a one-sentence summary: The robot joint shows moderate wear."
    print("Generating...")
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=60, num_beams=4)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("SUCCESS. Output:")
    print(text)
except Exception:
    print("FAILED with exception:")
    traceback.print_exc()