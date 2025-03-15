from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1. 加载 BERT 预训练的 tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # 二分类任务

# 2. 输入文本
text = "BERT is a powerful language model."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# 3. 进行推理（前向传播）
with torch.no_grad():
    outputs = model(**inputs)

# 4. 获取预测结果
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()

print(f"Predicted class: {predicted_class}")
