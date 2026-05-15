import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_MODEL = "Qwen/Qwen2.5-0.5B"


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    prompt = """请根据主题、关键词和体裁创作一首古诗。
主题：思乡；关键词：明月、故人、秋风；体裁：七言绝句"""

    messages = [
        {
            "role": "system",
            "content": "你是一个擅长中国古诗词创作的助手。请严格按照用户给定的主题、关键词和体裁生成古诗，关键词要出现在输出里，不要输出解释。"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    print(response)


if __name__ == "__main__":
    main()