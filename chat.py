import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


SYSTEM_PROMPT = (
    "你是一个擅长中国古诗词创作的助手。"
    "请严格按照用户给定的主题、关键词和体裁生成古诗，关键词需要出现在输出,不要输出解释。"
)


def get_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if args.lora_path:
        print("Loading LoRA adapter:", args.lora_path)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model_name = "Qwen + LoRA"
    else:
        model_name = "Base Qwen"

    model.to(device)
    model.eval()

    print("=" * 80)
    print(f"{model_name} 交互模式已启动。输入 exit / quit / q 退出。")
    print("推荐输入：主题：思乡；关键词：明月、故人、秋风；体裁：七言绝句")
    print("=" * 80)

    while True:
        user_input = input("\nUser >>> ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Bye.")
            break

        if not user_input:
            continue

        # 如果用户没有写 instruction，就自动补上，方便交互
        if "请根据主题" not in user_input and "创作" not in user_input:
            user_content = (
                "请根据主题、关键词和体裁创作一首古诗。\n"
                + user_input
            )
        else:
            user_content = user_input

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(device)

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "repetition_penalty": 1.1,
        }

        if args.do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            })
        else:
            gen_kwargs.update({"do_sample": False})

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        print("\nAssistant >>>")
        print(response)


if __name__ == "__main__":
    main()