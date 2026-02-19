import json
import random
import sys
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB

sys.path.insert(0, str(Path(__file__).parent))
from classifier import ContentClassifier

def load_and_clean_data(data_path, min_length=5, max_dup_rate=0.1):
    """
    加载并清洗训练数据
    
    Args:
        data_path: 数据文件路径
        min_length: 最小文本长度
        max_dup_rate: 最大重复率阈值
    
    Returns:
        清洗后的数据
    """
    print("正在加载数据...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据: {len(data)} 条")
    
    # 1. 过滤过短文本
    filtered_data = []
    for item in data:
        text = item.get('input', '').strip()
        if len(text) >= min_length:
            filtered_data.append({'text': text, 'label': item['label']})
    
    print(f"过滤短文本后: {len(filtered_data)} 条")
    
    # 2. 去重 - 保留第一次出现的标签
    seen_texts = {}
    dedup_data = []
    
    for item in filtered_data:
        text = item['text']
        if text not in seen_texts:
            seen_texts[text] = item['label']
            dedup_data.append(item)
    
    print(f"去重后: {len(dedup_data)} 条")
    print(f"去重率: {(1 - len(dedup_data) / len(filtered_data)):.2%}")
    
    # 3. 检查标签分布
    labels = [item['label'] for item in dedup_data]
    label_counts = Counter(labels)
    
    print("\n清洗后标签分布:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(dedup_data)*100:.1f}%)")
    
    # 4. 平衡数据（可选）
    min_count = min(label_counts.values())
    max_count = max(label_counts.values())
    balance_ratio = min_count / max_count
    
    print(f"\n数据平衡度: {balance_ratio:.3f}")
    
    if balance_ratio < 0.7:  # 如果不平衡
        print("数据不平衡，进行下采样...")
        target_count = int(min_count * 1.2)  # 稍微高于最少类别
        
        balanced_data = []
        label_counters = {label: 0 for label in label_counts.keys()}
        
        # 随机打乱
        random.shuffle(dedup_data)
        
        for item in dedup_data:
            label = item['label']
            if label_counters[label] < target_count:
                balanced_data.append(item)
                label_counters[label] += 1
        
        dedup_data = balanced_data
        print(f"平衡后数据量: {len(dedup_data)} 条")
        
        # 重新统计
        labels = [item['label'] for item in dedup_data]
        label_counts = Counter(labels)
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} ({count/len(dedup_data)*100:.1f}%)")
    
    return dedup_data

def create_fixed_test_set(data, test_size=0.2, random_seed=42):
    """
    创建固定的测试集
    
    Args:
        data: 清洗后的数据
        test_size: 测试集比例
        random_seed: 随机种子
    
    Returns:
        (train_data, test_data)
    """
    random.seed(random_seed)
    
    # 按标签分层采样
    label_groups = {}
    for item in data:
        label = item['label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item)
    
    train_data = []
    test_data = []
    
    for label, items in label_groups.items():
        random.shuffle(items)
        split_idx = int(len(items) * (1 - test_size))
        
        train_data.extend(items[:split_idx])
        test_data.extend(items[split_idx:])
    
    # 再次打乱
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    
    return train_data, test_data

def train_improved_model(train_data, test_data):
    """
    使用改进的方法训练模型
    """
    print("\n开始训练模型...")
    
    # 准备数据
    train_texts = [item['text'] for item in train_data]
    train_labels = [item['label'] for item in train_data]
    test_texts = [item['text'] for item in test_data]
    test_labels = [item['label'] for item in test_data]
    
    # 使用 ContentClassifier 的预处理方法
    classifier = ContentClassifier()
    
    # 预处理文本
    def preprocess_text(text):
        words = classifier._segment_text(text)
        clean_words = classifier._remove_stopwords(words)
        return ' '.join(clean_words)
    
    print("预处理文本...")
    train_processed = [preprocess_text(text) for text in train_texts]
    test_processed = [preprocess_text(text) for text in test_texts]
    
    # TF-IDF 向量化
    print("向量化...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    
    X_train = vectorizer.fit_transform(train_processed)
    X_test = vectorizer.transform(test_processed)
    
    print(f"特征维度: {X_train.shape[1]}")
    
    # 训练模型
    print("训练朴素贝叶斯...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, train_labels)
    
    # 评估
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(train_labels, train_pred)
    test_acc = accuracy_score(test_labels, test_pred)
    
    print(f"\n=== 训练结果 ===")
    print(f"训练准确率: {train_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"过拟合程度: {train_acc - test_acc:.4f}")
    
    # 详细报告
    print(f"\n=== 分类报告 ===")
    report = classification_report(test_labels, test_pred)
    print(report)
    
    # 保存模型
    classifier.model = model
    classifier.vectorizer = vectorizer
    
    model_path = "improved_model.pkl"
    classifier.save_model(model_path)
    print(f"\n模型已保存: {model_path}")
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'overfitting': train_acc - test_acc,
        'model': classifier
    }

def main():
    if len(sys.argv) < 2:
        print("用法: python train_model.py <数据文件路径>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    # 1. 清洗数据
    clean_data = load_and_clean_data(data_path)
    
    # 2. 创建固定测试集
    train_data, test_data = create_fixed_test_set(clean_data)
    
    # 3. 训练模型
    results = train_improved_model(train_data, test_data)
    
    print(f"\n=== 总结 ===")
    print(f"使用清洗后的数据训练，测试准确率: {results['test_accuracy']:.4f}")
    print(f"建议: 后续训练都使用相同的测试集进行评估")

if __name__ == "__main__":
    main()