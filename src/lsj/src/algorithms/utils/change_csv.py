# -*- coding: utf-8 -*-  # 防止中文乱码

import pandas as pd

label_map = {
    "开心": "Positive",
    "生气": "Negative",
    "伤心": "Negative",
    "厌恶": "Negative",
    "惊讶": "Positive",
    "平静": "Neutral",
    "关心": "Neutral",
    "疑问": "Neutral"
}

def convert_labels(input_csv_path, output_csv_path):
    """
    读取输入 CSV 文件，提取 text 和 label 两列，
    将 label 按照映射表转换后，过滤掉无效标签，
    最终保存为新的 CSV 文件。
    """
    # 读取输入 CSV 文件
    df = pd.read_csv(input_csv_path, encoding="utf-8")

    # 检查必要列是否存在
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("输入 CSV 文件必须至少包含 'text' 和 'label' 两列。")

    # 只保留 text 和 label 两列
    processed_df = df[["text", "label"]].copy()

    # 对 label 列进行映射，不在映射表中的值会变为 NaN
    processed_df["label"] = processed_df["label"].map(label_map)

    # 过滤掉映射失败（即原始 label 不在映射表中）的行
    processed_df = processed_df.dropna(subset=["label"])

    # 保存为新的 CSV 文件，使用 UTF-8 编码，不保存行索引
    processed_df.to_csv(output_csv_path, index=False, encoding="utf-8")

    print(f"处理完成，共保留 {len(processed_df)} 行数据，输出文件已保存至: {output_csv_path}")


if __name__ == "__main__":
    # 输入文件路径和输出文件路径
    input_csv_path = "../../training_data/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv"
    output_csv_path = "../../training_data/converted_dataset.csv"

    convert_labels(input_csv_path, output_csv_path)