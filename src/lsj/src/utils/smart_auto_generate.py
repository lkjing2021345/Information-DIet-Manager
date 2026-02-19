#!/usr/bin/env python3
"""
智能自动生成训练数据脚本
分析现有数据分布，自动平衡各类别数据量
"""

import asyncio
import json
import aiohttp
from typing import List, Dict, Tuple
from pathlib import Path
import logging
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from generate_training_data import TrainingDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartAutoGenerator:
    """智能自动数据生成器"""
    
    # 类别描述和示例
    CATEGORY_INFO = {
        "Shopping": {
            "description": "购物、电商网站",
            "examples": [
                "淘宝、京东、亚马逊等电商平台",
                "品牌官网商店",
                "二手交易平台",
                "团购、优惠券网站"
            ]
        },
        "Learning": {
            "description": "学习、教育、文档",
            "examples": [
                "编程教程和文档",
                "在线课程平台",
                "技术博客和论坛",
                "学术资源和论文"
            ]
        },
        "News": {
            "description": "新闻、资讯",
            "examples": [
                "新闻门户网站",
                "财经资讯",
                "科技新闻",
                "地方新闻"
            ]
        },
        "Entertainment": {
            "description": "娱乐、视频、音乐、游戏",
            "examples": [
                "视频网站（B站、YouTube等）",
                "音乐平台",
                "游戏相关",
                "电影、电视剧"
            ]
        },
        "Social": {
            "description": "社交媒体、论坛、社区",
            "examples": [
                "社交平台（微博、Twitter等）",
                "即时通讯",
                "论坛和社区",
                "问答平台"
            ]
        },
        "Tools": {
            "description": "工具、实用程序",
            "examples": [
                "在线工具（翻译、转换等）",
                "云存储和协作",
                "邮件和日历",
                "开发工具"
            ]
        }
    }
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        training_data_path: str = "../training_data/training_data.json"
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.training_data_path = training_data_path
    
    def analyze_existing_data(self) -> Dict[str, int]:
        """
        分析现有训练数据的分布
        
        Returns:
            各类别的数量统计
        """
        path = Path(self.training_data_path)
        
        if not path.exists():
            logger.info("训练数据文件不存在，将创建新文件")
            return {cat: 0 for cat in self.CATEGORY_INFO.keys()}
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            labels = [item['label'] for item in data]
            label_counts = Counter(labels)
            
            # 确保所有类别都有计数
            result = {cat: label_counts.get(cat, 0) for cat in self.CATEGORY_INFO.keys()}
            
            logger.info(f"\n现有数据分析:")
            logger.info(f"总数据量: {len(data)}")
            for cat, count in sorted(result.items()):
                logger.info(f"  {cat:15}: {count:4}")
            
            return result
            
        except Exception as e:
            logger.error(f"分析数据失败: {str(e)}")
            return {cat: 0 for cat in self.CATEGORY_INFO.keys()}
    
    def calculate_generation_plan(
        self,
        current_counts: Dict[str, int],
        target_total: int = None,
        balance_threshold: float = 0.2
    ) -> Dict[str, int]:
        """
        计算生成计划，平衡各类别数据
        
        Args:
            current_counts: 当前各类别数量
            target_total: 目标总数据量（None表示只平衡现有数据）
            balance_threshold: 平衡阈值（类别间差异不超过平均值的20%）
            
        Returns:
            各类别需要生成的数量
        """
        total_current = sum(current_counts.values())
        
        if target_total is None:
            # 只平衡现有数据
            avg_count = total_current // len(current_counts)
            plan = {}
            for cat, count in current_counts.items():
                if count < avg_count * (1 - balance_threshold):
                    plan[cat] = int(avg_count * (1 + balance_threshold) - count)
                else:
                    plan[cat] = 0
        else:
            # 达到目标总量并平衡
            target_per_category = target_total // len(current_counts)
            plan = {}
            for cat, count in current_counts.items():
                needed = max(0, target_per_category - count)
                plan[cat] = needed
        
        logger.info(f"\n生成计划:")
        total_to_generate = sum(plan.values())
        logger.info(f"总共需要生成: {total_to_generate}")
        for cat, count in sorted(plan.items()):
            if count > 0:
                logger.info(f"  {cat:15}: +{count:4}")
        
        return plan
    
    async def generate_titles_for_category(
        self,
        category: str,
        count: int
    ) -> List[str]:
        """
        为特定类别生成标题
        
        Args:
            category: 类别名称
            count: 生成数量
            
        Returns:
            生成的标题列表
        """
        cat_info = self.CATEGORY_INFO[category]
        
        prompt = f"""请生成{count}个{category}类别的真实浏览器标签页标题。

类别说明: {cat_info['description']}

包括但不限于:
{chr(10).join('- ' + ex for ex in cat_info['examples'])}

要求:
1. 标题要真实、自然，像真实用户浏览器中的标签页
2. 同时包含中文和英文网站（各占一半）
3. 标题格式多样化：
   - 网站名称 + 页面描述
   - 具体内容标题
   - 产品/服务名称
4. 避免重复和相似的标题
5. 标题长度适中（10-80字符）

请以JSON格式返回：
{{
  "titles": ["标题1", "标题2", ...]
}}

只返回JSON，不要有其他文字。"""
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.9,
                    "response_format": {"type": "json_object"}
                }
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        data = json.loads(content)
                        titles = data.get('titles', [])
                        
                        logger.info(f"✓ {category}: 生成 {len(titles)} 个标题")
                        return titles
                    else:
                        error_text = await response.text()
                        logger.error(f"API错误 {response.status}: {error_text}")
                        return []
                        
        except Exception as e:
            logger.error(f"生成{category}标题失败: {str(e)}")
            return []
    
    async def execute_generation_plan(
        self,
        plan: Dict[str, int]
    ) -> Dict:
        """
        执行生成计划
        
        Args:
            plan: 生成计划（各类别需要生成的数量）
            
        Returns:
            执行结果统计
        """
        all_titles = []
        
        # 为每个类别生成标题
        for category, count in plan.items():
            if count > 0:
                logger.info(f"\n生成 {category} 类别的 {count} 个标题...")
                titles = await self.generate_titles_for_category(category, count)
                all_titles.extend(titles)
                
                # 避免速率限制
                await asyncio.sleep(2)
        
        if not all_titles:
            logger.warning("没有生成任何标题")
            return {"success": 0, "failed": 0}
        
        logger.info(f"\n总共生成了 {len(all_titles)} 个标题，开始分类...")
        
        # 分类所有标题
        classifier = TrainingDataGenerator(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            max_retries=3,
            concurrent_requests=5
        )
        
        results = await classifier.classify_batch(all_titles)
        
        # 格式化并保存
        formatted_results = [
            {"input": r["input"], "label": r["label"]}
            for r in results
        ]
        
        if formatted_results:
            classifier.append_to_training_data(
                formatted_results,
                self.training_data_path
            )
            
            # 统计
            label_counts = Counter(r['label'] for r in formatted_results)
            
            logger.info("\n" + "="*60)
            logger.info("生成完成")
            logger.info("="*60)
            logger.info(f"成功: {len(formatted_results)}/{len(all_titles)}")
            logger.info("\n实际分类分布:")
            for label, count in sorted(label_counts.items()):
                logger.info(f"  {label:15}: {count}")
            
            return {
                "success": len(formatted_results),
                "failed": len(all_titles) - len(formatted_results),
                "distribution": dict(label_counts)
            }
        
        return {"success": 0, "failed": len(all_titles)}
    
    async def auto_generate(
        self,
        target_total: int = None,
        balance_only: bool = False
    ):
        """
        自动生成和平衡数据
        
        Args:
            target_total: 目标总数据量
            balance_only: 是否只平衡现有数据
        """
        logger.info("\n" + "#"*60)
        logger.info("# 智能自动数据生成器")
        logger.info("#"*60)
        
        # 分析现有数据
        current_counts = self.analyze_existing_data()
        
        # 计算生成计划
        if balance_only:
            plan = self.calculate_generation_plan(current_counts)
        else:
            plan = self.calculate_generation_plan(current_counts, target_total)
        
        # 执行生成
        if sum(plan.values()) > 0:
            await self.execute_generation_plan(plan)
            
            # 再次分析，显示最终结果
            logger.info("\n" + "="*60)
            logger.info("最终数据分布")
            logger.info("="*60)
            final_counts = self.analyze_existing_data()
        else:
            logger.info("\n数据已经平衡，无需生成")


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='智能自动训练数据生成器')
    parser.add_argument(
        '--api-key',
        default='your-api-key-here',
        help='API密钥'
    )
    parser.add_argument(
        '--target',
        type=int,
        help='目标总数据量（不指定则只平衡现有数据）'
    )
    parser.add_argument(
        '--balance-only',
        action='store_true',
        help='只平衡现有数据，不增加总量'
    )
    
    args = parser.parse_args()
    
    generator = SmartAutoGenerator(
        api_key=args.api_key,
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini"
    )
    
    await generator.auto_generate(
        target_total=args.target,
        balance_only=args.balance_only
    )


if __name__ == "__main__":
    asyncio.run(main())
