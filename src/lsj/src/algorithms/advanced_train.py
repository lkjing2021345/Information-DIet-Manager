import logging
import os
import random
import sys
from pathlib import Path

from scipy.sparse import hstack
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

sys.path.insert(0, str(Path(__file__).parent))
from classifier import ContentClassifier
from data_cleaning import (
    step1_load_and_inspect,
    step2_check_text_length,
    step3_check_duplicates,
    step4_check_label_distribution,
    step5_check_label_consistency,
    step6_remove_short_texts,
    step7_remove_duplicates,
    step8_balance_data
)

# 配置日志
logs_folder_path = "../../logs"
if not os.path.exists(logs_folder_path):
    os.makedirs(logs_folder_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../logs/advanced_train.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path, min_length=3, balance_strategy='downsample'):
    """
    加载并准备训练数据（使用 data_cleaning 模块的完整清洗流程）

    Args:
        data_path: 数据文件路径
        min_length: 最小文本长度
        balance_strategy: 数据平衡策略 ('downsample' 或 'upsample')

    Returns:
        清洗后的数据列表
    """
    logger.info("="*60)
    logger.info("开始完整数据清洗流程")
    logger.info("="*60)

    try:
        # 步骤1：加载并检查数据
        data = step1_load_and_inspect(data_path)

        # 步骤2：检查文本长度
        step2_check_text_length(data)

        # 步骤3：检查重复数据
        step3_check_duplicates(data)

        # 步骤4：检查标签分布
        step4_check_label_distribution(data)

        # 步骤5：检查标签一致性
        step5_check_label_consistency(data)

        # 转换数据格式
        logger.info("转换数据格式...")
        formatted_data = []
        for item in data:
            text = item.get('input', '').strip()
            label = item.get('label', '')
            if text and label:
                formatted_data.append({'input': text, 'label': label})

        logger.info(f"格式化后数据: {len(formatted_data)} 条")

        # 步骤6：移除短文本
        cleaned_data = step6_remove_short_texts(formatted_data, min_length=min_length)

        # 步骤7：去重
        cleaned_data = step7_remove_duplicates(cleaned_data)

        # 步骤8：平衡数据
        cleaned_data = step8_balance_data(cleaned_data, strategy=balance_strategy)

        # 转换为训练格式
        final_data = []
        for item in cleaned_data:
            final_data.append({
                'text': item['input'],
                'label': item['label']
            })

        logger.info("="*60)
        logger.info(f"数据清洗完成，最终数据量: {len(final_data)} 条")
        logger.info("="*60)

        return final_data

    except Exception as e:
        logger.exception("加载和准备数据时发生错误")
        sys.exit(1)


def create_stratified_split(data, test_size=0.15, random_seed=42):
    """
    创建分层划分的训练集和测试集

    Args:
        data: 数据列表
        test_size: 测试集比例
        random_seed: 随机种子

    Returns:
        (train_data, test_data)
    """
    logger.info(f"划分数据集（测试集比例: {test_size}）")

    try:
        random.seed(random_seed)

        # 按标签分组
        label_groups = {}
        for item in data:
            label = item['label']
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)

        train_data = []
        test_data = []

        # 分层采样
        for label, items in label_groups.items():
            random.shuffle(items)
            split_idx = int(len(items) * (1 - test_size))

            train_data.extend(items[:split_idx])
            test_data.extend(items[split_idx:])

        # 打乱数据
        random.shuffle(train_data)
        random.shuffle(test_data)

        logger.info(f"训练集: {len(train_data)} 条")
        logger.info(f"测试集: {len(test_data)} 条")

        return train_data, test_data

    except Exception as e:
        logger.exception("划分数据集时发生错误")
        sys.exit(1)


def extract_advanced_features(texts, classifier):
    """
    提取多层次特征

    Args:
        texts: 文本列表
        classifier: 分类器实例

    Returns:
        处理后的文本列表
    """
    try:
        processed = []
        for text in texts:
            words = classifier._segment_text(text)
            clean_words = classifier._remove_stopwords(words)
            processed.append(' '.join(clean_words))
        return processed
    except Exception as e:
        logger.exception("提取特征时发生错误")
        raise


def train_with_cross_validation(train_data, n_folds=5):
    """
    使用交叉验证训练模型并进行 GridSearchCV 超参数调优

    Args:
        train_data: 训练数据
        n_folds: 交叉验证折数

    Returns:
        训练好的分类器实例, 交叉验证结果字典, 训练集准确率
    """
    logger.info(f"使用 {n_folds} 折交叉验证训练模型（包含 GridSearchCV 调参）")

    try:
        # 准备数据
        texts = [item['text'] for item in train_data]
        labels = [item['label'] for item in train_data]

        classifier = ContentClassifier()

        # 预处理
        logger.info("预处理文本...")
        processed_texts = extract_advanced_features(texts, classifier)

        # 优化的特征提取配置
        feature_configs = [
            {
                'name': '词级(1-5gram)',
                'vectorizer': TfidfVectorizer(
                    max_features=50000,  # 增加特征数
                    min_df=2,
                    max_df=0.80,  # 降低最大文档频率阈值
                    ngram_range=(1, 5),  # 扩展到5-gram
                    sublinear_tf=True,
                    norm='l2',
                    use_idf=True,
                    smooth_idf=True
                )
            },
            {
                'name': '字符级(2-6gram)',
                'vectorizer': TfidfVectorizer(
                    max_features=30000,  # 增加字符特征数
                    analyzer='char',
                    ngram_range=(2, 6),  # 扩展到6-gram
                    min_df=2,  # 降低最小文档频率
                    max_df=0.85,
                    sublinear_tf=True,
                    norm='l2'
                )
            },
            {
                'name': '词级(1-2gram)高频',
                'vectorizer': TfidfVectorizer(
                    max_features=15000,
                    min_df=5,  # 只保留高频词
                    max_df=0.70,
                    ngram_range=(1, 2),
                    sublinear_tf=True
                )
            }
        ]

        # 提取特征
        logger.info("提取多层次特征...")
        feature_matrices = []
        vectorizers = []

        for config in feature_configs:
            vec = config['vectorizer']
            X = vec.fit_transform(processed_texts)
            feature_matrices.append(X)
            vectorizers.append(vec)
            logger.info(f"  {config['name']}: {X.shape[1]} 维")

        # 合并特征
        X_combined = hstack(feature_matrices)
        logger.info(f"总特征维度: {X_combined.shape[1]}")

        # 优化的模型配置
        models = {
            'NB': MultinomialNB(alpha=0.01),  # 降低平滑参数
            'LR': LogisticRegression(
                max_iter=30000,  # 增加迭代次数确保收敛
                C=5.0,  # 增加正则化强度
                solver='saga',  # 使用更快的求解器
                class_weight='balanced',
                random_state=42,
                verbose=0,
                # n_jobs=-1  # 使用所有CPU核心
            )
        }

        # 交叉验证评估每个模型
        logger.info("交叉验证评估（单模型）")

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_results = {}

        for name, model in models.items():
            logger.info(f"评估 {name} 模型...")
            scores = cross_val_score(
                model, X_combined, labels,
                cv=cv, scoring='accuracy', n_jobs=1
            )
            cv_results[name] = scores
            logger.info(f"  准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        # 训练最终集成模型并进行 GridSearchCV 调参
        logger.info("训练最终集成模型并进行 GridSearchCV 调参")

        ensemble = VotingClassifier(
            estimators=[
                ('nb', models['NB']),
                ('lr', models['LR'])
            ],
            voting='soft',
            weights=[1, 3]
        )

        # 优化的超参数网格
        param_grid = {
            'lr__C': [3.0, 5.0, 7.0, 10.0],  # 更大的C值范围
            'lr__max_iter': [30000],  # 固定更大的迭代次数
            'lr__solver': ['saga', 'lbfgs'],  # 测试不同求解器
            'nb__alpha': [0.001, 0.01, 0.03, 0.05],  # 更小的alpha值
            'weights': [[1, 2], [1, 3], [1, 4], [1, 5]]  # 测试不同的投票权重
        }

        logger.info("使用 GridSearchCV 进行超参数搜索（可能较慢）...")
        grid_search = GridSearchCV(
            estimator=ensemble,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            # n_jobs=-1,  # 使用所有CPU核心加速
            refit=True,
            verbose=2,
            error_score='raise'  # 遇到错误时抛出异常
        )

        grid_search.fit(X_combined, labels)

        logger.info(f"最佳参数: {grid_search.best_params_}")
        logger.info(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")

        # 使用最佳模型
        best_ensemble = grid_search.best_estimator_
        classifier.model = best_ensemble
        classifier.vectorizer = vectorizers[0]
        classifier.char_vectorizer = vectorizers[1]
        classifier.high_freq_vectorizer = vectorizers[2]  # 保存第三个向量化器

        # 训练集准确率
        train_pred = classifier.model.predict(X_combined)
        train_acc = accuracy_score(labels, train_pred)

        logger.info(f"训练集准确率（使用最佳模型）: {train_acc:.4f}")
        logger.info(f"交叉验证（LR）平均准确率: {cv_results['LR'].mean():.4f}")

        return classifier, cv_results, train_acc

    except Exception as e:
        logger.exception("训练模型时发生错误")
        sys.exit(1)


def evaluate_on_test_set(classifier, test_data):
    """
    在测试集上评估模型

    Args:
        classifier: 训练好的分类器
        test_data: 测试数据

    Returns:
        评估结果
    """
    logger.info("测试集评估")

    try:
        test_texts = [item['text'] for item in test_data]
        test_labels = [item['label'] for item in test_data]

        # 预处理
        processed_texts = extract_advanced_features(test_texts, classifier)

        # 提取特征（使用所有三个向量化器）
        X_test_word = classifier.vectorizer.transform(processed_texts)
        X_test_char = classifier.char_vectorizer.transform(processed_texts)
        X_test_high_freq = classifier.high_freq_vectorizer.transform(processed_texts)
        X_test = hstack([X_test_word, X_test_char, X_test_high_freq])

        # 预测
        predictions = classifier.model.predict(X_test)
        test_acc = accuracy_score(test_labels, predictions)

        logger.info(f"测试准确率: {test_acc:.4f}")

        # 详细报告
        logger.info("详细分类报告:")
        report = classification_report(test_labels, predictions, digits=4)
        for line in report.split('\n'):
            if line.strip():
                logger.info(f"  {line}")

        # 错误分析
        errors = []
        for i, (true, pred, text) in enumerate(zip(test_labels, predictions, test_texts)):
            if true != pred:
                errors.append({
                    'text': text,
                    'true': true,
                    'pred': pred
                })

        error_rate = len(errors) / len(test_labels) * 100
        logger.info(f"错误样本数: {len(errors)}/{len(test_labels)} ({error_rate:.2f}%)")

        if errors:
            logger.info("前10个错误样本:")
            for i, err in enumerate(errors[:10], 1):
                logger.info(f"  [{i}] 文本: {err['text'][:60]}...")
                logger.info(f"      真实: {err['true']} | 预测: {err['pred']}")

        return {
            'accuracy': test_acc,
            'errors': errors,
            'report': report
        }

    except Exception as e:
        logger.exception("评估测试集时发生错误")
        sys.exit(1)


def main():
    try:
        if len(sys.argv) < 2:
            logger.error("参数不足")
            logger.info("用法: python advanced_train.py <数据文件路径>")
            sys.exit(1)

        data_path = sys.argv[1]

        if not Path(data_path).exists():
            logger.error(f"数据文件不存在: {data_path}")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("高级模型训练 - 目标准确率 98%+")
        logger.info("=" * 60)

        # 1. 加载和清洗数据（使用完整清洗流程）
        logger.info("步骤 1: 完整数据清洗")
        clean_data = load_and_prepare_data(
            data_path,
            min_length=3,
            balance_strategy='upsample'  # 使用下采样平衡数据
        )

        # 2. 划分数据集
        logger.info("步骤 2: 划分数据集")
        train_data, test_data = create_stratified_split(clean_data, test_size=0.15)

        # 3. 交叉验证训练（含 GridSearchCV）
        logger.info("步骤 3: 交叉验证训练（含调参）")
        classifier, cv_results, train_acc = train_with_cross_validation(train_data, n_folds=5)

        # 4. 测试集评估
        logger.info("步骤 4: 测试集评估")
        test_results = evaluate_on_test_set(classifier, test_data)

        # 5. 保存模型
        logger.info("保存模型")

        model_folder = Path(__file__).parent / "models"
        model_folder.mkdir(exist_ok=True)
        model_path = model_folder / "advanced_model.pkl"

        classifier.save_model(str(model_path))
        logger.info(f"模型已保存: {model_path}")

        # 6. 最终总结
        logger.info("=" * 60)
        logger.info("训练总结")
        logger.info("=" * 60)
        logger.info(f"训练集准确率: {train_acc:.4f}")
        logger.info(f"交叉验证准确率: {cv_results['LR'].mean():.4f} (+/- {cv_results['LR'].std() * 2:.4f})")
        logger.info(f"测试集准确率: {test_results['accuracy']:.4f}")
        logger.info("=" * 60)

        if test_results['accuracy'] >= 0.98:
            logger.info("恭喜！模型准确率达到98%以上！")
        elif test_results['accuracy'] >= 0.95:
            logger.info("模型表现良好，准确率超过95%")
            logger.info("改进建议:")
            logger.info("  1. 增加更多训练数据")
            logger.info("  2. 分析错误样本，改进特征工程")
            logger.info("  3. 尝试更复杂的模型（如深度学习）")
        else:
            logger.info(f"当前准确率: {test_results['accuracy']:.2%}")
            logger.info("改进建议:")
            logger.info("  1. 检查数据质量和标注一致性")
            logger.info("  2. 增加更多训练数据（目标15000+）")
            logger.info("  3. 优化特征提取方法")
            logger.info("  4. 调整模型超参数")

    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        sys.exit(0)
    except Exception as e:
        logger.exception("程序执行失败")
        sys.exit(1)


if __name__ == "__main__":
    main()