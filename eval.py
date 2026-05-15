import argparse
import math
import json
import re
from collections import Counter
from pathlib import Path


def normalize_text(text: str) -> str:
    """去掉空白。"""
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", "", text)
    return text


def parse_keywords(input_text: str):
    """
    从 input 里解析关键词：
    例如：主题：思乡；关键词：明月、故人、秋风；体裁：七言绝句
    """
    m = re.search(r"关键词[:：]([^；;\n]+)", input_text)
    if not m:
        return []

    kw_text = m.group(1).strip()
    keywords = re.split(r"[、,，/ ]+", kw_text)
    keywords = [k.strip() for k in keywords if k.strip()]
    return keywords


def parse_form(input_text: str):
    """
    从 input 里解析体裁：
    例如：体裁：七言绝句
    """
    m = re.search(r"体裁[:：]([^；;\n]+)", input_text)
    if not m:
        return ""
    return m.group(1).strip()


def split_poem_lines(output: str):
    """
    把模型输出切成诗句。
    例如：
    秋风吹梦到江楼，明月无声照客愁。
    故人一别千山远，独倚寒窗望旧游。

    -> 四句
    """
    output = output.strip()

    # 去掉常见引号和书名号
    output = re.sub(r"[《》「」『』“”\"']", "", output)

    # 按中文标点和换行切分
    parts = re.split(r"[，。！？；、\n\r]+", output)

    lines = []
    for p in parts:
        p = normalize_text(p)
        if p:
            lines.append(p)

    return lines


def count_chinese_chars(line: str):
    """
    统计一句里的中文字符数量。
    这样可以避免空格、标点影响。
    """
    return len(re.findall(r"[\u4e00-\u9fff]", line))


def chinese_char_tokens(text: str):
    """
    BLEU 用的字符级 token。
    中文诗句没有天然空格分词，按汉字计算能避免 jieba 等额外依赖。
    """
    return re.findall(r"[\u4e00-\u9fff]", normalize_text(text))


def ngram_counts(tokens, n: int):
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(candidate_texts, reference_texts, max_order: int = 4, smooth: float = 1.0):
    """
    计算 corpus BLEU。返回 0-1 之间的小数。
    使用 add-k smoothing，避免短诗某个高阶 ngram 为 0 时 BLEU 直接归零。
    """
    matches_by_order = [0] * max_order
    possible_by_order = [0] * max_order
    cand_len = 0
    ref_len = 0

    for cand_text, ref_text in zip(candidate_texts, reference_texts):
        cand = chinese_char_tokens(cand_text)
        ref = chinese_char_tokens(ref_text)
        cand_len += len(cand)
        ref_len += len(ref)

        for order in range(1, max_order + 1):
            cand_ngrams = ngram_counts(cand, order)
            ref_ngrams = ngram_counts(ref, order)
            overlap = cand_ngrams & ref_ngrams
            matches_by_order[order - 1] += sum(overlap.values())
            possible_by_order[order - 1] += max(len(cand) - order + 1, 0)

    if cand_len == 0:
        return 0.0

    precisions = []
    for matches, possible in zip(matches_by_order, possible_by_order):
        if possible == 0:
            precisions.append(0.0)
        else:
            precisions.append((matches + smooth) / (possible + smooth))

    if min(precisions) <= 0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)

    brevity_penalty = 1.0 if cand_len > ref_len else math.exp(1.0 - ref_len / cand_len)
    return brevity_penalty * geo_mean


def check_format(output: str, form: str):
    """
    判断格式是否正确。
    五言绝句：4 句，每句 5 个汉字
    七言绝句：4 句，每句 7 个汉字
    """
    lines = split_poem_lines(output)

    if "五言绝句" in form:
        return len(lines) == 4 and all(count_chinese_chars(line) == 5 for line in lines)

    if "七言绝句" in form:
        return len(lines) == 4 and all(count_chinese_chars(line) == 7 for line in lines)

    # 没有识别体裁时，不计为正确
    return False


def keyword_coverage(output: str, keywords):
    """
    关键词覆盖率：
    输入 3 个关键词，输出中出现 2 个，则为 2/3。
    """
    if not keywords:
        return None

    output_norm = normalize_text(output)

    hit = 0
    hit_keywords = []
    missed_keywords = []

    for kw in keywords:
        if kw in output_norm:
            hit += 1
            hit_keywords.append(kw)
        else:
            missed_keywords.append(kw)

    return hit / len(keywords), hit_keywords, missed_keywords


def evaluate_file(path: str, show_bad: int = 10):
    total = 0
    format_ok = 0

    cov_sum = 0.0
    cov_count = 0

    bad_format_cases = []
    bad_keyword_cases = []
    bleu_candidates = []
    bleu_references = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            input_text = item.get("input", "")
            output = item.get("output", "")
            reference = item.get("reference", "")

            form = parse_form(input_text)
            keywords = parse_keywords(input_text)

            total += 1

            if reference:
                bleu_candidates.append(output)
                bleu_references.append(reference)

            # 1. 格式正确率
            is_format_ok = check_format(output, form)
            if is_format_ok:
                format_ok += 1
            else:
                if len(bad_format_cases) < show_bad:
                    bad_format_cases.append({
                        "idx": idx,
                        "form": form,
                        "input": input_text,
                        "output": output,
                        "lines": split_poem_lines(output),
                        "line_lengths": [count_chinese_chars(x) for x in split_poem_lines(output)]
                    })

            # 2. 关键词覆盖率
            cov_result = keyword_coverage(output, keywords)
            if cov_result is not None:
                cov, hit_keywords, missed_keywords = cov_result
                cov_sum += cov
                cov_count += 1

                if cov < 1.0 and len(bad_keyword_cases) < show_bad:
                    bad_keyword_cases.append({
                        "idx": idx,
                        "input": input_text,
                        "output": output,
                        "keywords": keywords,
                        "hit_keywords": hit_keywords,
                        "missed_keywords": missed_keywords,
                        "coverage": cov
                    })

    format_rate = format_ok / total if total else 0.0
    avg_keyword_coverage = cov_sum / cov_count if cov_count else 0.0
    bleu4 = corpus_bleu(bleu_candidates, bleu_references) if bleu_references else None

    print("=" * 80)
    print(f"文件: {path}")
    print(f"样本数: {total}")
    print(f"格式正确数: {format_ok}")
    print(f"格式正确率: {format_rate:.4f}  ({format_rate * 100:.2f}%)")
    print(f"关键词样本数: {cov_count}")
    print(f"平均关键词覆盖率: {avg_keyword_coverage:.4f}  ({avg_keyword_coverage * 100:.2f}%)")
    if bleu4 is not None:
        print(f"BLEU-4(char): {bleu4:.4f}  ({bleu4 * 100:.2f})")
        print(f"BLEU样本数: {len(bleu_references)}")

    print("\n" + "=" * 80)
    print(f"格式错误案例，最多显示 {show_bad} 个:")
    for case in bad_format_cases:
        print("-" * 80)
        print("idx:", case["idx"])
        print("体裁:", case["form"])
        print("input:", case["input"])
        print("output:", case["output"])
        print("切分句子:", case["lines"])
        print("每句字数:", case["line_lengths"])

    print("\n" + "=" * 80)
    print(f"关键词未完全覆盖案例，最多显示 {show_bad} 个:")
    for case in bad_keyword_cases:
        print("-" * 80)
        print("idx:", case["idx"])
        print("input:", case["input"])
        print("output:", case["output"])
        print("关键词:", case["keywords"])
        print("命中:", case["hit_keywords"])
        print("遗漏:", case["missed_keywords"])
        print("覆盖率:", f"{case['coverage']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, default="/root/aigc/qwen/output/lora_results2.jsonl")
    parser.add_argument("--show_bad", type=int, default=10)
    args = parser.parse_args()

    evaluate_file(args.result_file, show_bad=args.show_bad)


if __name__ == "__main__":
    main()
