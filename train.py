import os
import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)


# =========================
# 1. 基本配置
# =========================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "/root/aigc/qwen/data/poem_sfttang.jsonl"
OUTPUT_DIR = "/root/aigc/qwen/output/qwen_poem_lora_tang2"

MAX_LENGTH = 512
TRAIN_RATIO = 0.95
SEED = 42
KEYWORD_LOSS_WEIGHT = 3.5


# =========================
# 2. 固定随机种子
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 3. 数据集
# =========================

class PoemSFTDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def build_prompt(self, item: Dict) -> str:
        instruction = item["instruction"].strip()
        input_text = item["input"].strip()

        user_content = instruction + "\n" + input_text

        messages = [
            {
                "role": "system",
                "content": "你是一个擅长中国古诗词创作的助手。请严格按照用户给定的主题、关键词和体裁生成古诗，关键词需要出现在输出中，不要输出解释。"
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        # 使用 Qwen 自带 chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def parse_keywords(self, input_text: str) -> List[str]:
        m = re.search(r"关键词[:：]([^；;\n]+)", input_text)
        if not m:
            return []
        return [k.strip() for k in re.split(r"[、,，/ ]+", m.group(1)) if k.strip()]

    def build_answer_loss_weights(self, answer: str, answer_ids: List[int], keywords: List[str]) -> List[float]:
        weights = [1.0] * len(answer_ids)
        if not keywords:
            return weights

        for keyword in keywords:
            start = 0
            while True:
                pos = answer.find(keyword, start)
                if pos < 0:
                    break
                before_ids = self.tokenizer(
                    answer[:pos],
                    add_special_tokens=False,
                )["input_ids"]
                keyword_ids = self.tokenizer(
                    keyword,
                    add_special_tokens=False,
                )["input_ids"]
                token_start = len(before_ids)
                token_end = min(token_start + len(keyword_ids), len(weights))
                for i in range(token_start, token_end):
                    weights[i] = KEYWORD_LOSS_WEIGHT
                start = pos + len(keyword)

        return weights

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = self.build_prompt(item)
        answer = item["output"].strip()

        if self.tokenizer.eos_token is not None:
            answer = answer + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False
        )["input_ids"]

        answer_ids = self.tokenizer(
            answer,
            add_special_tokens=False
        )["input_ids"]

        input_ids = prompt_ids + answer_ids

        # 关键：prompt 部分不计算 loss，只让模型学习 output
        labels = [-100] * len(prompt_ids) + answer_ids
        keywords = self.parse_keywords(item["input"])
        answer_loss_weights = self.build_answer_loss_weights(answer, answer_ids, keywords)
        loss_weights = [0.0] * len(prompt_ids) + answer_loss_weights

        # 截断
        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        loss_weights = loss_weights[: self.max_length]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": loss_weights,
        }


# =========================
# 4. Data Collator
# =========================

@dataclass
class SFTDataCollator:
    tokenizer: AutoTokenizer
    max_length: int = 512

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        loss_weights = [f["loss_weights"] for f in features]

        max_len = min(
            max(len(x) for x in input_ids),
            self.max_length
        )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_loss_weights = []

        for ids, mask, lbs, weights in zip(input_ids, attention_mask, labels, loss_weights):
            ids = ids[:max_len]
            mask = mask[:max_len]
            lbs = lbs[:max_len]
            weights = weights[:max_len]

            pad_len = max_len - len(ids)

            batch_input_ids.append(ids + [pad_id] * pad_len)
            batch_attention_mask.append(mask + [0] * pad_len)
            batch_labels.append(lbs + [-100] * pad_len)
            batch_loss_weights.append(weights + [0.0] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "loss_weights": torch.tensor(batch_loss_weights, dtype=torch.float32),
        }


class KeywordWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_weights = inputs.pop("loss_weights", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)

        valid_mask = shift_labels.ne(-100).float()
        if loss_weights is not None:
            shift_weights = loss_weights[..., 1:].to(token_loss.device)
            weights = shift_weights * valid_mask
        else:
            weights = valid_mask

        loss = (token_loss * weights).sum() / weights.sum().clamp_min(1.0)
        return (loss, outputs) if return_outputs else loss


# =========================
# 5. 读取数据
# =========================

def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            if not all(k in item for k in ["instruction", "input", "output"]):
                continue

            if not item["instruction"].strip():
                continue
            if not item["input"].strip():
                continue
            if not item["output"].strip():
                continue

            data.append(item)

    return data


# =========================
# 6. 主函数
# =========================

def main():
    set_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # 关闭 cache，训练时更稳
    model.config.use_cache = False

    # =========================
    # 7. 注入 LoRA
    # =========================

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # =========================
    # 8. 构建训练集 / 验证集
    # =========================

    data = load_jsonl(DATA_PATH)
    random.shuffle(data)

    split_idx = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    print(f"Total samples: {len(data)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")

    train_dataset = PoemSFTDataset(train_data, tokenizer, MAX_LENGTH)
    eval_dataset = PoemSFTDataset(eval_data, tokenizer, MAX_LENGTH)

    data_collator = SFTDataCollator(tokenizer, MAX_LENGTH)

    # =========================
    # 9. 训练参数
    # =========================

    training_args = TrainingArguments(
        
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,

        num_train_epochs=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,

        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),


        report_to="tensorboard",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),

        remove_unused_columns=False,
    )

    # =========================
    # 10. Trainer
    # =========================

    trainer = KeywordWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_data) > 0 else None,
        data_collator=data_collator,
    )

    print("Start training...")
    trainer.train()

    print("Saving LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Done. Adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
