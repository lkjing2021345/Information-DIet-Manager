#!/usr/bin/env python3
"""
智能自动生成训练数据脚本
分析现有数据分布，自动平衡各类别数据量
"""

import asyncio
import json
import aiohttp
from typing import List, Dict, Tuple, Set
from pathlib import Path
import logging
import sys
from collections import Counter
import hashlib

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
        self.existing_titles: Set[str] = set()
        self._load_existing_titles()

    def _load_existing_titles(self):
        """加载现有标题用于去重"""
        path = Path(self.training_data_path)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.existing_titles = {self._normalize_title(item['input']) for item in data}
                logger.info(f"已加载 {len(self.existing_titles)} 个现有标题用于去重")
            except Exception as e:
                logger.error(f"加载现有标题失败: {str(e)}")
                self.existing_titles = set()

    def _normalize_title(self, title: str) -> str:
        """标准化标题用于去重比较"""
        # 移除空格、转小写、移除特殊字符
        normalized = ''.join(title.lower().split())
        normalized = ''.join(c for c in normalized if c.isalnum() or ord(c) > 127)
        return normalized

    def _is_duplicate(self, title: str) -> bool:
        """检查标题是否重复"""
        normalized = self._normalize_title(title)
        return normalized in self.existing_titles

    def _add_title(self, title: str):
        """添加标题到去重集合"""
        normalized = self._normalize_title(title)
        self.existing_titles.add(normalized)

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
        count: int,
        batch_size: int = 20
    ) -> List[str]:
        """
        为特定类别生成标题（分批生成以提高多样性）

        Args:
            category: 类别名称
            count: 生成数量
            batch_size: 每批生成数量

        Returns:
            生成的标题列表
        """
        cat_info = self.CATEGORY_INFO[category]
        all_titles = []

        # 分批生成，每批使用不同的temperature和提示词变化
        batches = (count + batch_size - 1) // batch_size

        for batch_idx in range(batches):
            batch_count = min(batch_size, count - len(all_titles))
            if batch_count <= 0:
                break

            # 动态调整temperature以增加多样性
            temperature = 0.9 + (batch_idx * 0.1)
            temperature = min(temperature, 1.5)

            # 添加批次特定的提示词变化
            focus_areas = [
                "专注于具体的产品、服务或内容名称",
                "专注于用户操作场景（搜索、浏览、购买等）",
                "专注于不同的子领域和细分市场",
                "专注于不同的网站类型和平台"
            ]
            focus = focus_areas[batch_idx % len(focus_areas)]

            prompt = f"""请生成{batch_count}个{category}类别的真实浏览器标签页标题。

类别说明: {cat_info['description']}

包括但不限于:
{chr(10).join('- ' + ex for ex in cat_info['examples'])}

本批次重点: {focus}

要求:
1. 标题要真实、自然、具体，像真实用户浏览器中的标签页
2. 同时包含中文和英文网站（各占一半）
3. 标题格式多样化：
   - 网站名称 + 具体页面/内容
   - 详细的内容标题（包含数字、日期、版本号等）
   - 产品/服务的具体型号或名称
   - 包含动作词（如：如何、教程、指南、评测等）
4. 每个标题都要独特，避免模板化
5. 标题长度适中（15-80字符）
6. 包含具体细节（如：版本号、日期、作者、系列名等）

请以JSON格式返回：
{{
  "titles": ["标题1", "标题2", ...]
}}

只返回JSON，不要有其他文字。"""

            titles = await self._call_api_for_titles(prompt, temperature)

            # 去重过滤
            unique_titles = []
            for title in titles:
                if not self._is_duplicate(title):
                    unique_titles.append(title)
                    self._add_title(title)
                else:
                    logger.debug(f"跳过重复标题: {title}")

            all_titles.extend(unique_titles)
            logger.info(f"✓ {category} 批次{batch_idx+1}/{batches}: 生成 {len(unique_titles)}/{len(titles)} 个唯一标题")

            # 避免速率限制
            if batch_idx < batches - 1:
                await asyncio.sleep(1)

        return all_titles

    async def _call_api_for_titles(
        self,
        prompt: str,
        temperature: float = 0.9
    ) -> List[str]:
        """调用API生成标题"""

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
                    "temperature": temperature,
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
                        return titles
                    else:
                        error_text = await response.text()
                        logger.error(f"API错误 {response.status}: {error_text}")
                        return []
                        
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
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
        total_generated = 0  # 记录API生成的总数
        total_duplicates = 0  # 记录被过滤的重复数

        # 为每个类别生成标题
        for category, count in plan.items():
            if count > 0:
                logger.info(f"\n生成 {category} 类别的 {count} 个标题...")
                # 多生成20%以应对去重损失
                target_count = int(count * 1.2)

                # 记录生成前的标题数
                before_count = len(self.existing_titles)
                titles = await self.generate_titles_for_category(category, target_count)
                after_count = len(self.existing_titles)

                # 计算本批次的统计
                batch_duplicates = target_count - len(titles)
                total_generated += target_count
                total_duplicates += batch_duplicates

                # 只取需要的数量
                titles = titles[:count]
                all_titles.extend(titles)
                
                logger.info(f"实际获得 {len(titles)} 个唯一标题（过滤了 {batch_duplicates} 个重复）")

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
            
            # 计算去重率（被过滤的重复数 / API生成的总数）
            duplicate_rate = (total_duplicates / total_generated) if total_generated > 0 else 0

            logger.info("\n" + "="*60)
            logger.info("生成完成")
            logger.info("="*60)
            logger.info(f"API生成总数: {total_generated}")
            logger.info(f"过滤重复数: {total_duplicates}")
            logger.info(f"唯一标题数: {len(all_titles)}")
            logger.info(f"重复率: {duplicate_rate:.2%}")
            logger.info(f"分类成功: {len(formatted_results)}/{len(all_titles)}")
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
