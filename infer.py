import os
import json
import argparse
import re
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


SYSTEM_PROMPT = (
    "你是一个擅长中国古诗词创作的助手。"
    "请严格按照用户给定的主题、关键词和体裁生成古诗，关键词需要出现在输出中，不要输出解释。"
)


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(items: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_keywords(input_text: str) -> List[str]:
    m = re.search(r"关键词[:：]([^；;\n]+)", input_text)
    if not m:
        return []
    return [k.strip() for k in re.split(r"[、,，/ ]+", m.group(1)) if k.strip()]


def missing_keywords(output: str, keywords: List[str]) -> List[str]:
    output_norm = re.sub(r"\s+", "", output or "")
    return [kw for kw in keywords if kw not in output_norm]


def build_prompt(tokenizer, item: Dict, extra_requirement: str = "") -> str:
    instruction = item.get("instruction", "").strip()
    input_text = item.get("input", "").strip()

    if instruction and input_text:
        user_content = instruction + "\n" + input_text
    else:
        user_content = instruction or input_text
    if extra_requirement:
        user_content = user_content + "\n" + extra_requirement

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return text


@torch.no_grad()
def generate_prompts(
    model,
    tokenizer,
    device,
    prompts: List[str],
    max_input_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": 1.05,
    }

    if do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        })
    else:
        gen_kwargs.update({
            "do_sample": False,
        })

    outputs = model.generate(
        **inputs,
        **gen_kwargs,
    )

    prompt_len = inputs["input_ids"].shape[1]
    responses = []
    for output_ids in outputs:
        response_ids = output_ids[prompt_len:]
        responses.append(tokenizer.decode(response_ids, skip_special_tokens=True).strip())
    return responses


def get_dtype():
    if not torch.cuda.is_available():
        return torch.float32

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def load_model_and_tokenizer(model_name: str, lora_path: str = None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # decoder-only 模型批量生成时建议 left padding
    tokenizer.padding_side = "left"

    dtype = get_dtype()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading base model:", model_name)
    print("Using device:", device)
    print("Using dtype:", dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if lora_path is not None:
        print("Loading LoRA adapter:", lora_path)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)

    model.to(device)
    model.eval()

    print("Model device:", next(model.parameters()).device)

    return model, tokenizer, device


@torch.no_grad()
def batch_generate(
    model,
    tokenizer,
    device,
    items: List[Dict],
    batch_size: int = 4,
    max_input_length: int = 512,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.9,
    keyword_retry: bool = True,
):
    results = []

    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]

        prompts = [build_prompt(tokenizer, item) for item in batch_items]
        responses = generate_prompts(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        for item, response in zip(batch_items, responses):
            keywords = parse_keywords(item.get("input", ""))
            missed = missing_keywords(response, keywords)
            if keyword_retry and missed:
                extra_requirement = (
                    "硬性要求：输出中必须逐字包含全部关键词，不能替换、拆分或省略。"
                    f"当前必须包含：{'、'.join(keywords)}。"
                )
                retry_prompt = build_prompt(tokenizer, item, extra_requirement=extra_requirement)
                retry_response = generate_prompts(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompts=[retry_prompt],
                    max_input_length=max_input_length,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )[0]
                if len(missing_keywords(retry_response, keywords)) <= len(missed):
                    response = retry_response

            new_item = dict(item)
            new_item["output"] = response
            results.append(new_item)

        print(f"Finished {min(start + batch_size, len(items))}/{len(items)}")

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="如果要测微调后的 LoRA，就传 LoRA adapter 路径；测原始 Qwen 时不传。",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="/root/aigc/qwen/data/test.jsonl"
       
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="加上这个参数则使用采样生成；不加则使用贪心生成，适合做指标对比。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--keyword_retry",
        dest="keyword_retry",
        action="store_true",
        help="漏关键词时，用更强关键词约束提示重试一次。",
    )
    parser.add_argument(
        "--no_keyword_retry",
        dest="keyword_retry",
        action="store_false",
    )
    parser.set_defaults(keyword_retry=True)

    args = parser.parse_args()

    print("Loading test file:", args.test_file)
    items = load_jsonl(args.test_file)
    print("Test samples:", len(items))

    model, tokenizer, device = load_model_and_tokenizer(
        model_name=args.model,
        lora_path=args.lora_path,
    )

    results = batch_generate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        items=items,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        keyword_retry=args.keyword_retry,
    )

    save_jsonl(results, args.output_file)
    print("Saved to:", args.output_file)


if __name__ == "__main__":
    main()
