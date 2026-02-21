#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import asyncio
from pathlib import Path
from smart_auto_generate import SmartAutoGenerator

async def main():
    import sys

    # Load config
    config_path = Path(__file__).parent / "config.json"

    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        print("\n请创建 config.json 文件，包含以下字段:")
        print("  - api_key: API密钥")
        print("  - base_url: API基础URL")
        print("  - model: 模型名称")
        print("  - training_data_path: 训练数据文件路径")
        print("  - max_retries: 最大重试次数（可选，默认3）")
        print("  - concurrent_requests: 并发请求数（可选，默认5）")
        sys.exit(1)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"错误: 读取配置文件失败: {str(e)}")
        sys.exit(1)

    # 验证必需字段
    # required_fields = ['api_key', 'base_url', 'model', 'training_data_path']
    # missing_fields = [field for field in required_fields if field not in config]
    # if missing_fields:
    #     print(f"错误: 配置文件缺少必需字段: {', '.join(missing_fields)}")
    #     sys.exit(1)

    # 检查是否启用多模型模式
    enable_multi_model = config.get('enable_multi_model', False)

    if enable_multi_model and 'models' in config:
        # 多模型模式
        models_config = config['models']
        print(f"✓ 启用多模型模式，共 {len(models_config)} 个模型")
        for i, model in enumerate(models_config, 1):
            print(f"  模型{i}: {model['name']} - {model['model']}")
    else:
        # 单模型模式（向后兼容）
        models_config = [{
            'name': 'default',
            'api_key': config['api_key'],
            'base_url': config['base_url'],
            'model': config['model'],
            'max_retries': config.get('max_retries', 3),
            'concurrent_requests': config.get('concurrent_requests', 5),
            'weight': 1.0
        }]
        print("✓ 使用单模型模式")

    # 创建生成器
    generator = SmartAutoGenerator(
        models_config=models_config,
        training_data_path=config['training_data_path'],
        load_balance_strategy=config.get('load_balance_strategy', 'round_robin')
    )
    
    # 智能批量生成配置
    TARGET_TOTAL = 1000000  # 目标总数
    BATCH_SIZE_PER_CATEGORY = 100  # 每个类别每批生成数量
    MAX_BATCHES = 100000  # 最大批次数
    BALANCE_THRESHOLD = 0.15  # 平衡阈值（15%）
    COOLDOWN_SECONDS = 10  # 批次间冷却时间

    print(f"\n{'#'*60}")
    print(f"# 智能批量数据生成")
    print(f"# 目标: {TARGET_TOTAL:,} 条数据")
    print(f"# 每批每类别: {BATCH_SIZE_PER_CATEGORY} 条")
    print(f"# 平衡阈值: {BALANCE_THRESHOLD*100}%")
    print(f"{'#'*60}\n")

    consecutive_failures = 0
    max_consecutive_failures = 3

    for batch_num in range(1, MAX_BATCHES + 1):
        print(f"\n{'='*60}")
        print(f"批次 {batch_num}/{MAX_BATCHES}")
        print(f"{'='*60}")
        
        # 分析当前数据分布
        current_counts = generator.analyze_existing_data()
        total_current = sum(current_counts.values())
        
        # 检查是否达到目标
        if total_current >= TARGET_TOTAL:
            print(f"\n🎉 目标达成! 当前总数: {total_current:,}")
            break
        
        # 计算剩余需要生成的数量
        remaining = TARGET_TOTAL - total_current
        print(f"\n进度: {total_current:,}/{TARGET_TOTAL:,} ({total_current/TARGET_TOTAL*100:.1f}%)")
        print(f"剩余: {remaining:,} 条")

        # 智能生成计划：优先补充数量少的类别
        avg_count = total_current / len(current_counts)
        plan = {}

        for cat, count in current_counts.items():
            # 如果该类别低于平均值的(1-阈值)，则优先生成
            if count < avg_count * (1 - BALANCE_THRESHOLD):
                # 生成更多以快速平衡
                plan[cat] = BATCH_SIZE_PER_CATEGORY * 2
            elif count < avg_count * (1 + BALANCE_THRESHOLD):
                # 正常生成
                plan[cat] = BATCH_SIZE_PER_CATEGORY
            else:
                # 该类别已足够，少量生成或跳过
                plan[cat] = max(0, BATCH_SIZE_PER_CATEGORY // 2)

        # 显示本批次计划
        total_planned = sum(plan.values())
        if total_planned == 0:
            print("\n⚠️ 所有类别已平衡，调整为均匀生成")
            plan = {cat: BATCH_SIZE_PER_CATEGORY for cat in generator.CATEGORY_INFO.keys()}
            total_planned = sum(plan.values())

        print(f"\n本批次计划生成 {total_planned} 条:")
        for cat, count in sorted(plan.items()):
            if count > 0:
                current = current_counts[cat]
                print(f"  {cat:15}: +{count:2} (当前: {current:5})")

        # 执行生成
        try:
            result = await generator.execute_generation_plan(plan)

            if result['success'] > 0:
                consecutive_failures = 0
                print(f"\n✓ 批次 {batch_num} 完成: 成功 {result['success']} 条")
            else:
                consecutive_failures += 1
                print(f"\n✗ 批次 {batch_num} 失败: 无数据生成")

                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n⚠️ 连续 {consecutive_failures} 次失败，停止生成")
                    break

        except Exception as e:
            consecutive_failures += 1
            print(f"\n✗ 批次 {batch_num} 异常: {str(e)}")

            if consecutive_failures >= max_consecutive_failures:
                print(f"\n⚠️ 连续 {consecutive_failures} 次异常，停止生成")
                break

        # 批次间冷却
        if batch_num < MAX_BATCHES:
            print(f"\n等待 {COOLDOWN_SECONDS} 秒后继续...")
            await asyncio.sleep(COOLDOWN_SECONDS)

    # 最终统计
    print(f"\n{'='*60}")
    print("最终统计")
    print(f"{'='*60}")
    final_counts = generator.analyze_existing_data()
    final_total = sum(final_counts.values())

    print(f"\n总数据量: {final_total:,}")
    print(f"目标完成度: {final_total/TARGET_TOTAL*100:.1f}%")

    # 检查平衡度
    if final_total > 0:
        avg = final_total / len(final_counts)
        max_deviation = max(abs(count - avg) / avg for count in final_counts.values())
        print(f"数据平衡度: {(1-max_deviation)*100:.1f}% (偏差: {max_deviation*100:.1f}%)")

        if max_deviation <= BALANCE_THRESHOLD:
            print("\n✓ 数据分布已平衡")
        else:
            print("\n⚠️ 数据分布需要进一步平衡")

if __name__ == "__main__":
    asyncio.run(main())
