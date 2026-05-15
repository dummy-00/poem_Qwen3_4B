# Qwen 古诗生成微调实验

本项目基于 Qwen 模型进行古诗生成微调。输入主题、关键词和体裁后，模型生成对应的五言或七言古诗。

---

## 文件说明

### `data/bulid.py`

从原始数据集中构建适合微调训练的 jsonl 数据。

### `data/poem_sfttang.jsonl`

构建好的微调训练数据集，共 5000 条。

### `train.py`

LoRA 微调训练代码，直接运行即可开始训练。

### `data/test.jsonl`

测试集，共 50 条样本。

### `infer.py`

推理生成代码。

不加 `--lora_path` 时，使用原始 Qwen 模型推理。

加上 `--lora_path` 时，使用微调后的 Qwen 模型推理。

运行示例：

```bash
python infer.py --lora_path 训好的lora地址 --output_file 结果输出地址
```

### `chat.py`

命令行交互式对话脚本。

不加 `--lora_path` 时，和原始 Qwen 模型对话。

加上 `--lora_path` 时，和微调后的 Qwen 模型对话。

运行示例：

```bash
python chat.py --lora_path 训好的lora地址
```

### `eval.py`

评测脚本。

将 `infer.py` 生成的结果 json 文件导入后，计算三个指标：

- 格式正确率
- 关键词覆盖率
- BLEU-4(char)

---

## 运行方式

### 1. 构建训练数据

```bash
python data/bulid.py
```

运行后会生成：

```text
data/poem_sfttang.jsonl
```

### 2. 训练 LoRA

```bash
python train.py
```

训练完成后会得到微调后的 LoRA 权重。

### 3. Base 模型推理

```bash
python infer.py --output_file 保存json结果
```

### 4. LoRA 微调后模型推理

```bash
python infer.py --lora_path 训好的lora地址 --output_file 保存json结果
```

### 5. 命令行对话

使用原始 Qwen：

```bash
python chat.py
```

使用微调后的 Qwen：

```bash
python chat.py --lora_path 训好的lora地址
```

### 6. 评测结果

```bash
python eval.py （改代码中的读取文件来评测）
```



---

## 评测指标

### 格式正确率

判断模型生成的诗是否符合指定体裁，例如五言、七言。

### 关键词覆盖率

统计输入关键词在生成诗句中的出现情况。

### BLEU-4(char)

衡量模型生成的诗和参考诗在字面 n-gram 上的相似程度。

由于古诗生成是开放式任务，BLEU 只作为参考，主要关注格式正确率和关键词覆盖率。

---

## 实验结果

| 模型 | 样本数 | 格式正确率 | 平均关键词覆盖率 | BLEU-4(char) |
|---|---:|---:|---:|---:|
| Qwen Base | 50 | 60.00% | 93.33% | 6.20 |
| Qwen + LoRA | 50 | 86.00% | 97.33% | 5.42 |

---

## 结果说明

LoRA 微调后，模型的格式正确率从 60.00% 提升到 86.00%，关键词覆盖率从 93.33% 提升到 97.33%。

这说明微调后的模型更能遵守古诗格式要求，也更能覆盖输入关键词。

BLEU-4(char) 略有下降，但由于古诗生成具有开放性，因此该指标只作为辅助参考。