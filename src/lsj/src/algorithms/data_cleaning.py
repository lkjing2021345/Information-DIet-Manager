import json
import sys
from pathlib import Path
from collections import Counter
import random
import logging
import os

logs_folder_path = "../../logs"
if not os.path.exists(logs_folder_path):
    os.makedirs(logs_folder_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../logs/data_clean.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================
# 第一部分：基础数据检查
# ============================================================

def step1_load_and_inspect(data_path):
    """
    步骤1：加载数据并进行初步检查
    
    学习目标：
    - 了解数据的基本结构
    - 发现明显的问题
    """
    logger.info("步骤1：加载并检查数据")

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"加载数据成功: {data_path}")
        logger.info(f"总数据量: {len(data)} 条")

        if data:
            logger.info("数据结构示例（第1条）:")
            first_item = data[0]
            for key, value in first_item.items():
                logger.info(f"  {key}: {value}")

        logger.info("开始检查数据完整性...")
        missing_count = 0
        for i, item in enumerate(data):
            if 'input' not in item or 'label' not in item:
                logger.warning(f"第{i + 1}条数据缺失字段")
                missing_count += 1
            elif not item.get('input') or not item.get('label'):
                logger.warning(f"第{i + 1}条数据字段为空")
                missing_count += 1

        if missing_count == 0:
            logger.info("所有数据字段完整")
        else:
            logger.warning(f"发现 {missing_count} 条数据有问题")

        return data

    except FileNotFoundError:
        logger.error(f"文件不存在: {data_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"加载数据失败: {e}")
        sys.exit(1)


def step2_check_text_length(data):
    logger.info("步骤2：检查文本长度")
    
    try:
        texts = [item.get('input', '') for item in data]
        lengths = [len(text) for text in texts]

        if not lengths:
            logger.warning("数据为空，无法检查文本长度")
            return data

        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)

        logger.info("文本长度统计:")
        logger.info(f"  平均长度: {avg_length:.1f} 字符")
        logger.info(f"  最短: {min_length} 字符")
        logger.info(f"  最长: {max_length} 字符")

        short_texts = [(i, text) for i, text in enumerate(texts) if len(text) < 3]

        if short_texts:
            logger.warning(f"发现 {len(short_texts)} 条过短文本（<3字符）")
            for i, text in short_texts[:5]:
                logger.info(f"  [{i + 1}] '{text}'")
            if len(short_texts) > 5:
                logger.info(f"  ... 还有 {len(short_texts) - 5} 条")
        else:
            logger.info("没有发现过短文本")

        logger.info("长度分布:")
        ranges = [(0, 10), (10, 50), (50, 100), (100, 500), (500, float('inf'))]
        for start, end in ranges:
            count = sum(1 for l in lengths if start <= l < end)
            percentage = count / len(lengths) * 100
            logger.info(f"  {start}-{end if end != float('inf') else '∞'} 字符: {count} 条 ({percentage:.1f}%)")

    except Exception as e:
        logger.exception(f"检查文本长度时发生错误: {e}")

    return data

def step3_check_duplicates(data):
    logger.info("步骤3：检查重复数据")

    try:
        text_counts = {}

        for item in data:
            text = item.get('input', '')
            if text in text_counts:
                text_counts[text] += 1
            else:
                text_counts[text] = 1

        duplicates = {text: count for text, count in text_counts.items() if count > 1}

        total_texts = len(data)
        unique_texts = len(text_counts)
        duplicate_rate = (total_texts - unique_texts) / total_texts * 100

        if duplicates:
            logger.warning("发现重复数据")
            logger.info(f"  总数据量: {total_texts} 条")
            logger.info(f"  唯一文本: {unique_texts} 条")
            logger.info(f"  重复率: {duplicate_rate:.2f}%")

            logger.info("重复最多的文本（前5个）:")
            sorted_dups = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)
            for text, count in sorted_dups[:5]:
                logger.info(f"  出现{count}次: '{text[:50]}...'")
        else:
            logger.info("没有发现重复数据")

    except Exception as e:
        logger.exception(f"检查重复数据时发生错误: {e}")

    return data

def step4_check_label_distribution(data):
    logger.info("步骤4：检查标签分布")
    
    try:
        labels = [item.get('label', '') for item in data]
        label_counts = Counter(labels)

        if not labels:
            logger.warning("数据为空，无法检查标签分布")
            return data

        total = len(labels)
        logger.info("标签分布:")
        for label, count in sorted(label_counts.items()):
            percentage = count / total * 100
            bar = "█" * int(percentage / 2)
            logger.info(f"  {label:15s}: {count:5d} ({percentage:5.1f}%) {bar}")

        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        balance_ratio = min_count / max_count

        logger.info("平衡度分析:")
        logger.info(f"  最少类别: {min_count} 条")
        logger.info(f"  最多类别: {max_count} 条")
        logger.info(f"  平衡比例: {balance_ratio:.2f}")

        if balance_ratio >= 0.8:
            logger.info("数据较为平衡")
        elif balance_ratio >= 0.5:
            logger.warning("数据轻度不平衡，建议平衡处理")
        else:
            logger.warning("数据严重不平衡，必须进行平衡处理")

    except Exception as e:
        logger.exception(f"检查标签分布时发生错误: {e}")

    return data

def step5_check_label_consistency(data):
    logger.info("步骤5：检查标签一致性")

    try:
        text_labels = {}
        for item in data:
            text = item.get('input', '')
            label = item.get('label', '')

            if text not in text_labels:
                text_labels[text] = []
            text_labels[text].append(label)

        conflicts = []
        for text, labels in text_labels.items():
            unique_labels = set(labels)
            if len(unique_labels) > 1:
                conflicts.append({
                    'text': text,
                    'labels': list(unique_labels),
                    'counts': Counter(labels)
                })

        if conflicts:
            logger.warning(f"发现 {len(conflicts)} 条文本标签不一致")
            for i, conflict in enumerate(conflicts[:5], 1):
                logger.warning(f"  [{i}] 文本: '{conflict['text'][:50]}...'")
                logger.warning(f"      标签: {conflict['labels']}")
                logger.warning(f"      分布: {dict(conflict['counts'])}")

            if len(conflicts) > 5:
                logger.warning(f"  ... 还有 {len(conflicts) - 5} 条冲突")

            logger.warning("建议：人工检查这些样本，统一标签")
        else:
            logger.info("所有文本标签一致")

    except Exception as e:
        logger.exception(f"检查标签一致性时发生错误: {e}")

    return data

def step6_remove_short_texts(data, min_length=3):
    logger.info(f"步骤6：移除短于{min_length}字符的文本")

    try:
        original_count = len(data)

        filtered_data = []
        removed_texts = []

        for item in data:
            text = item.get('input', '')
            if len(text) >= min_length:
                filtered_data.append(item)
            else:
                removed_texts.append(text)

        removed_count = original_count - len(filtered_data)

        logger.info(f"原始数据: {original_count} 条")
        logger.info(f"过滤后: {len(filtered_data)} 条")
        logger.info(f"移除: {removed_count} 条 ({removed_count / original_count * 100:.1f}%)")

        if removed_texts:
            logger.info("移除的文本示例（前5个）:")
            for text in removed_texts[:5]:
                logger.info(f"  '{text}'")

        return filtered_data

    except Exception as e:
        logger.exception(f"移除短文本时发生错误: {e}")
        return data


def step7_remove_duplicates(data):
    logger.info("步骤7：移除重复数据")

    try:
        original_count = len(data)

        seen_texts = {}
        dedup_data = []

        for item in data:
            text = item.get('input', '')
            if text not in seen_texts:
                seen_texts[text] = True
                dedup_data.append(item)

        removed_count = original_count - len(dedup_data)

        logger.info(f"原始数据: {original_count} 条")
        logger.info(f"去重后: {len(dedup_data)} 条")
        logger.info(f"移除: {removed_count} 条 ({removed_count / original_count * 100:.1f}%)")

        return dedup_data

    except Exception as e:
        logger.exception(f"移除重复数据时发生错误: {e}")
        return data

def step8_balance_data(data, strategy='downsample'):
    logger.info(f"步骤8：平衡数据（策略：{strategy}）")

    try:
        label_groups = {}
        for item in data:
            label = item.get('label', '')
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)

        logger.info("原始分布:")
        for label, items in sorted(label_groups.items()):
            logger.info(f"  {label}: {len(items)} 条")

        if strategy == 'downsample':
            # 下采样：减少多数类到最少类的数量
            min_count = min(len(items) for items in label_groups.values())
            logger.info(f"下采样目标: 每类 {min_count} 条")

            balanced_data = []
            for label, items in label_groups.items():
                random.shuffle(items)
                balanced_data.extend(items[:min_count])

        elif strategy == 'upsample':
            # 上采样：增加少数类到最多类的数量
            max_count = max(len(items) for items in label_groups.values())
            logger.info(f"上采样目标: 每类 {max_count} 条")

            balanced_data = []
            for label, items in label_groups.items():
                # 重复采样直到达到目标数量
                while len(items) < max_count:
                    items.extend(random.sample(items, min(len(items), max_count - len(items))))
                balanced_data.extend(items[:max_count])

        logger.info("平衡后分布:")
        balanced_labels = Counter(item.get('label', '') for item in balanced_data)
        for label, count in sorted(balanced_labels.items()):
            logger.info(f"  {label}: {count} 条")

        logger.info(f"总数据量: {len(data)} → {len(balanced_data)}")

        return balanced_data

    except Exception as e:
        logger.exception(f"平衡数据时发生错误: {e}")
        return data

def step9_save_cleaned_data(data, output_path):
    logger.info("步骤9：保存清洗后的数据")

    try:
        output_path = Path(output_path)
        if output_path.is_dir():
            logger.error(f"输出路径是目录而不是文件: {output_path}")
            logger.info("请指定完整的文件路径，例如: ../training_data/cleaned_data.json")
            sys.exit(1)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"数据已保存到: {output_path}")
        logger.info(f"总数据量: {len(data)} 条")

    except PermissionError:
        logger.error(f"没有权限写入文件: {output_path}")
        logger.info("请检查文件权限或选择其他输出路径")
        sys.exit(1)
    except OSError as e:
        logger.error(f"保存文件时发生系统错误: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("保存数据时发生未知错误")
        sys.exit(1)

def full_cleaning_pipeline(input_path, output_path):
    try:
        data = step1_load_and_inspect(input_path)

        step2_check_text_length(data)
        step3_check_duplicates(data)
        step4_check_label_distribution(data)
        step5_check_label_consistency(data)

        data = step6_remove_short_texts(data, min_length=3)
        data = step7_remove_duplicates(data)
        data = step8_balance_data(data, strategy='downsample')

        step9_save_cleaned_data(data, output_path)

        logger.info("数据清洗完成")

    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"数据清洗流程发生错误: {e}")
        sys.exit(1)

def main():
    """
    主程序入口
    
    用法：
    python data_cleaning.py <输入文件> <输出文件>
    
    示例：
    python data_cleaning.py data.json cleaned_data.json
    """
    try:
        if len(sys.argv) < 3:
            logger.error("参数不足")
            logger.info("用法: python data_cleaning.py <输入文件> <输出文件>")
            logger.info("示例: python data_cleaning.py data.json cleaned_data.json")
            sys.exit(1)

        input_path = sys.argv[1]
        output_path = sys.argv[2]

        if not Path(input_path).exists():
            logger.error(f"输入文件不存在: {input_path}")
            sys.exit(1)

        full_cleaning_pipeline(input_path, output_path)

    except Exception as e:
        logger.exception(f"程序执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()