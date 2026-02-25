# -*- coding: utf-8 -*-  # 防止中文乱码

import pandas as pd

input_path = "../../training_data/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv"
output_path = "../../training_data/converted_dataset.csv"
df = pd.read_csv(input_path, encoding="utf-8")

# ====== 2. 定义映射表（中文 -> 常量） ======
label_map = {
    "开心": "Positive",        # 快乐
    "生气": "Negative",      # 愤怒
    "伤心": "Negative",    # 悲伤
    "厌恶": "Negative",    # 厌恶
    "惊讶": "Positive",   # 惊讶
    "平静": "Neutral",
    "关心": "Neutral",
    "疑问": "Neutral",
}

# ====== 3. 检查是否有未映射的标签 ======
unique_labels = set(df["label"].unique())
unmapped = unique_labels - set(label_map.keys())

if unmapped:
    raise ValueError(f"以下标签没有映射规则: {unmapped}")

# ====== 4. 执行映射 ======
df["label"] = df["label"].map(label_map)

# ====== 5. 保存结果 ======
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"✅ 转换完成，已保存到: {output_path}")