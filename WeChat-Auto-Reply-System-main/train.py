import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用 HF 镜像加速下载（国内推荐）
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from snownlp import SnowNLP


# === 1. 情感分析：SnowNLP 打标 ===
def add_sentiment_analysis(texts):
    """
    使用 SnowNLP 对文本打情感标签（0:负, 1:中, 2:正）
    """
    labels = []
    for text in texts:
        if not text or not text.strip():
            labels.append(1)  # 空消息视为中性
            continue
        try:
            score = SnowNLP(text).sentiments  # 0~1 的正面概率
        except Exception:
            score = 0.5
        if score < 0.4:
            label = 0
        elif score > 0.6:
            label = 2
        else:
            label = 1
        labels.append(label)
    return labels


# === 2. 读取聊天记录 ===
def read_chat_data(file_path):
    """
    从文件读取消息，格式：Name:消息内容
    返回纯文本列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and ':' in line:
                content = line.split(':', 1)[1].strip()
                if content:
                    data.append(content)
    return data


# === 3. 创建数据集 ===
def create_dataset(texts, labels):
    return Dataset.from_dict({
        'text': texts,
        'sentiment': labels
    })


# === 4. 数据预处理 ===
def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )


# === 5. 主训练流程 ===
def main():
    name = "xxx好大儿"  #  修改为你自己的联系人备注名

    # 关键：获取 train.py 所在目录（不是当前工作目录！）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"{name}所有聊天记录.txt")

    # 检查聊天记录是否存在
    if not os.path.exists(file_path):
        print(f"聊天记录文件不存在: {file_path}")
        print("请先获取聊天记录（例如使用 wxauto 获取）")
        return

    print(" 正在读取聊天记录...")
    chat_data = read_chat_data(file_path)
    if not chat_data:
        print(" 聊天记录为空或格式不正确（需包含 ':' 分隔的行）")
        return

    print(f" 成功读取 {len(chat_data)} 条消息。")

    print(" 正在进行中文情感分析（使用 SnowNLP）...")
    sentiment_labels = add_sentiment_analysis(chat_data)

    print(" 标签分布统计:")
    neg = sum(1 for x in sentiment_labels if x == 0)
    neu = sum(1 for x in sentiment_labels if x == 1)
    pos = sum(1 for x in sentiment_labels if x == 2)
    print(f"   - 负向: {neg}")
    print(f"   - 中性: {neu}")
    print(f"   - 正向: {pos}")

    print("️ 创建并划分数据集（8:2）...")
    dataset = create_dataset(chat_data, sentiment_labels)
    dataset = dataset.train_test_split(test_size=0.2)

    print(" 加载 BERT 中文模型与分词器...")
    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    print("️ 预处理数据（分词、填充、截断）...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    #  重命名列：'sentiment' → 'labels'（Trainer 要求）
    tokenized_dataset = tokenized_dataset.rename_column("sentiment", "labels")

    # 设置 PyTorch 格式
    tokenized_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )

    print(" 开始训练（兼容旧版 transformers）...")
    #  使用兼容旧版本的 TrainingArguments（支持 transformers >= 3.0）
    training_args = TrainingArguments(
        output_dir=os.path.join(current_dir, './results'),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(current_dir, './logs'),
        logging_steps=10,
        save_strategy="steps",      # 旧版只支持 "steps" 或 "no"
        save_steps=100,             # 每100步保存一次 checkpoint
        report_to="none"            # 不上报到 wandb 等
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer
    )

    trainer.train()

    print(" 保存微调模型和分词器...")
    model_save_path = os.path.join(current_dir, f"{name}_finetuned_model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f" 模型已成功保存至: {model_save_path}")
    print("🎉 训练完成！现在可在同目录下的推理脚本中加载此模型。")


if __name__ == "__main__":
    main()




#