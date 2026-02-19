"""
训练数据生成脚本
使用大模型对浏览器标签页标题进行分类，并添加到训练数据集中
"""

import json
import asyncio
import aiohttp
from typing import List, Dict, Optional
from pathlib import Path
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """训练数据生成器"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        concurrent_requests: int = 5
    ):
        """
        初始化生成器
        
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model: 使用的模型名称
            max_retries: 最大重试次数
            concurrent_requests: 并发请求数
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.max_retries = max_retries
        self.concurrent_requests = concurrent_requests
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        
        # 分类提示词
        self.system_prompt = """你是一个专业的浏览器标签页分类助手。
请根据标签页的标题和URL，将其分类到以下类别之一：
- Shopping: 购物、电商相关
- Learning: 学习、教育、文档相关
- News: 新闻、资讯相关
- Entertainment: 娱乐、视频、音乐、游戏相关
- Social: 社交媒体、论坛、社区相关
- Tools: 工具、实用程序相关

请以JSON格式返回结果，格式如下：
{
  "label": "分类标签",
  "confidence": 0.95
}

只返回JSON，不要有其他文字。"""

    async def classify_single(
        self,
        session: aiohttp.ClientSession,
        input_text: str,
        retry_count: int = 0
    ) -> Optional[Dict[str, any]]:
        """
        分类单个标签页标题
        
        Args:
            session: aiohttp会话
            input_text: 输入文本
            retry_count: 当前重试次数
            
        Returns:
            分类结果字典，包含label和confidence
        """
        async with self.semaphore:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"请分类这个标签页标题：{input_text}"}
                    ],
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"}
                }
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        classification = json.loads(content)
                        
                        logger.info(f"✓ 分类成功: {input_text[:50]}... -> {classification['label']}")
                        return {
                            "input": input_text,
                            "label": classification['label'],
                            "confidence": classification.get('confidence', 1.0)
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"API错误 {response.status}: {error_text}")
                        
                        if retry_count < self.max_retries:
                            logger.info(f"重试 {retry_count + 1}/{self.max_retries}: {input_text[:50]}...")
                            await asyncio.sleep(2 ** retry_count)  # 指数退避
                            return await self.classify_single(session, input_text, retry_count + 1)
                        
                        return None
                        
            except asyncio.TimeoutError:
                logger.error(f"请求超时: {input_text[:50]}...")
                if retry_count < self.max_retries:
                    logger.info(f"重试 {retry_count + 1}/{self.max_retries}")
                    await asyncio.sleep(2 ** retry_count)
                    return await self.classify_single(session, input_text, retry_count + 1)
                return None
                
            except Exception as e:
                logger.error(f"分类失败: {input_text[:50]}... - {str(e)}")
                if retry_count < self.max_retries:
                    logger.info(f"重试 {retry_count + 1}/{self.max_retries}")
                    await asyncio.sleep(2 ** retry_count)
                    return await self.classify_single(session, input_text, retry_count + 1)
                return None

    async def classify_batch(self, inputs: List[str]) -> List[Dict[str, any]]:
        """
        批量分类标签页标题
        
        Args:
            inputs: 输入文本列表
            
        Returns:
            分类结果列表
        """
        logger.info(f"开始批量分类 {len(inputs)} 条数据，并发数: {self.concurrent_requests}")
        
        async with aiohttp.ClientSession() as session:
            tasks = [self.classify_single(session, input_text) for input_text in inputs]
            results = await asyncio.gather(*tasks)
            
        # 过滤掉失败的结果
        valid_results = [r for r in results if r is not None]
        logger.info(f"分类完成: 成功 {len(valid_results)}/{len(inputs)}")
        
        return valid_results

    def append_to_training_data(
        self,
        new_data: List[Dict[str, any]],
        training_data_path: str
    ) -> None:
        """
        将新数据追加到训练数据文件
        
        Args:
            new_data: 新的训练数据
            training_data_path: 训练数据文件路径
        """
        try:
            # 读取现有数据
            path = Path(training_data_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            # 去重：检查input是否已存在
            existing_inputs = {item['input'] for item in existing_data}
            unique_new_data = [
                item for item in new_data 
                if item['input'] not in existing_inputs
            ]
            
            if not unique_new_data:
                logger.warning("没有新的唯一数据需要添加")
                return
            
            # 合并数据
            combined_data = existing_data + unique_new_data
            
            # 备份原文件
            if path.exists():
                backup_path = path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                path.rename(backup_path)
                logger.info(f"已备份原文件到: {backup_path}")
            
            # 写入新数据
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✓ 成功添加 {len(unique_new_data)} 条新数据到 {training_data_path}")
            logger.info(f"总数据量: {len(combined_data)}")
            
        except Exception as e:
            logger.error(f"写入训练数据失败: {str(e)}")
            raise


async def main():
    """主函数"""
    
    # 配置参数
    API_KEY = "your-api-key-here"  # 替换为你的API密钥
    BASE_URL = "https://api.openai.com/v1"  # 或其他兼容的API地址
    MODEL = "gpt-4o-mini"
    
    # 训练数据文件路径
    TRAINING_DATA_PATH = "src/lsj/src/training_data/training_data.json"
    
    # 待分类的标签页标题列表
    sample_inputs = [
        "GitHub - microsoft/vscode: Visual Studio Code",
        "Python Documentation - Built-in Functions",
        "Amazon.com: Online Shopping for Electronics",
        "CNN - Breaking News, Latest News and Videos",
        "YouTube - Broadcast Yourself",
        "Stack Overflow - Where Developers Learn",
        "Netflix - Watch TV Shows Online",
        "Twitter / X - Home",
        "Google Translate",
        "Spotify - Web Player",
        "淘宝网 - 淘！我喜欢",
        "知乎 - 有问题，就会有答案",
        "哔哩哔哩 (゜-゜)つロ 干杯~",
        "微博 - 随时随地发现新鲜事",
        "百度网盘 - 自由存，随心看",
        "CSDN - 专业开发者社区",
        "人民网 - 网上的人民日报",
        "豆瓣电影 - 你的光影记录",
        "腾讯新闻 - 事实派",
        "网易云音乐 - 听见好时光",
    ]
    
    # 创建生成器
    generator = TrainingDataGenerator(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        max_retries=3,
        concurrent_requests=5
    )
    
    # 批量分类
    results = await generator.classify_batch(sample_inputs)
    
    # 格式化结果（移除confidence字段，只保留input和label）
    formatted_results = [
        {"input": r["input"], "label": r["label"]}
        for r in results
    ]
    
    # 追加到训练数据
    if formatted_results:
        generator.append_to_training_data(formatted_results, TRAINING_DATA_PATH)
        
        # 打印统计信息
        label_counts = {}
        for item in formatted_results:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info("\n分类统计:")
        for label, count in sorted(label_counts.items()):
            logger.info(f"  {label}: {count}")
    else:
        logger.warning("没有成功分类的数据")


if __name__ == "__main__":
    asyncio.run(main())