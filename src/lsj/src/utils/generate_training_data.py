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
        
        # 分类提示词（增强版）- 7大类别
        self.system_prompt = """你是一个专业的浏览器标签页分类专家。请根据标签页标题，精确分类到以下7个类别之一。

## 类别定义（共7类）

### News（新闻）
- 新闻门户：人民网、新华网、CNN、BBC
- 科技资讯：36氪、虎嗅、TechCrunch、The Verge
- 财经新闻：华尔街日报、财新、Bloomberg
- 地方新闻：各地日报、地方媒体
- 特征词：新闻、资讯、报道、快讯、头条、最新、今日、突发

### Tools（工具）
- 在线工具：翻译、转换、压缩、格式化
- 云存储：百度网盘、Google Drive、Dropbox
- 办公协作：腾讯文档、Notion、飞书、石墨
- 开发工具：GitHub、GitLab、Postman、VS Code
- 效率工具：Trello、Todoist、印象笔记
- 特征词：工具、在线、转换、编辑、文档、云盘、API、仓库、协作

### Learning（学习）
- 编程文档：Python Docs、MDN、API文档
- 在线课程：Coursera、网易云课堂、慕课、Udemy
- 技术社区：CSDN、Stack Overflow、掘金
- 学术资源：论文、arXiv、知网、Google Scholar
- 语言学习：多邻国、扇贝单词、Quizlet
- 特征词：教程、文档、课程、学习、如何、指南、文献、论文

### Shopping（购物）
- 电商平台：淘宝、京东、亚马逊、拼多多
- 品牌官网：Apple Store、Nike官网
- 二手交易：闲鱼、转转、eBay
- 垂直电商：唯品会、网易严选、小红书商城
- 特征词：购买、购物车、商品、价格、促销、优惠券、订单、特卖

### Social（社交）
- 社交网络：微博、Twitter、Facebook、Instagram
- 即时通讯：微信、QQ、Telegram、Discord
- 论坛社区：知乎、Reddit、豆瓣、贴吧
- 问答平台：Quora、SegmentFault、V2EX
- 内容社区：小红书、即刻、少数派
- 特征词：动态、评论、点赞、关注、话题、讨论、问答、社区、分享

### Entertainment（娱乐）
- 视频平台：B站、YouTube、抖音、爱奇艺
- 音乐平台：网易云音乐、Spotify、QQ音乐
- 游戏相关：Steam、原神、游戏攻略、TapTap
- 影视剧集：豆瓣电影、Netflix、腾讯视频
- 直播平台：斗鱼、虎牙、Twitch
- 特征词：视频、音乐、游戏、电影、电视剧、番剧、直播、UP主、播放

### Other（其他）
- 系统设置：Windows设置、浏览器设置、扩展管理
- 本地文件：文件管理器、本地HTML、PDF阅读
- 帮助文档：软件帮助、用户手册、FAQ
- 错误页面：404错误、500错误、网络错误
- 其他：空白页、新标签页、书签管理、历史记录
- 特征词：设置、配置、帮助、错误、本地、file://

## 分类规则

1. 优先根据核心功能分类，而非网站类型
2. 如果标题包含多个特征，选择最主要的功能
3. 技术文档和教程归类为Learning
4. 开发工具（GitHub、VS Code等）归类为Tools
5. 科技新闻归类为News，而非Learning
6. 无法明确分类或系统相关的归为Other
7. 必须从7个类别中选择一个：News, Tools, Learning, Shopping, Social, Entertainment, Other

## 输出格式

返回JSON格式，包含分类标签和置信度：
{
  "label": "类别名称",
  "confidence": 0.95,
  "reasoning": "简短的分类理由"
}

label必须是以下7个之一：News, Tools, Learning, Shopping, Social, Entertainment, Other

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
                        {"role": "user", "content": f"请分析并分类这个标签页标题：\n\n标题：{input_text}\n\n请给出分类、置信度和理由。"}
                    ],
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 200
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
                        
                        # 验证分类结果 - 7大类别
                        valid_labels = ['News', 'Tools', 'Learning', 'Shopping', 'Social', 'Entertainment', 'Other']
                        if classification['label'] not in valid_labels:
                            logger.warning(f"无效分类标签: {classification['label']}, 标题: {input_text[:50]}")
                            return None

                        confidence = classification.get('confidence', 1.0)
                        reasoning = classification.get('reasoning', '')
                        logger.info(
                            f"✓ 分类成功: {input_text[:40]}... -> {classification['label']} "
                            f"(置信度: {confidence:.2f}) {reasoning[:30] if reasoning else ''}"
                        )
                        return {
                            "input": input_text,
                            "label": classification['label'],
                            "confidence": classification.get('confidence', 1.0)
                        }
                    else:
                        error_text = await response.text()

                        # 处理速率限制
                        if response.status == 429:
                            wait_time = 2 ** (retry_count + 1)
                            logger.warning(f"触发速率限制，等待{wait_time}秒: {input_text[:40]}...")
                            await asyncio.sleep(wait_time)
                            return await self.classify_single(session, input_text, retry_count + 1)

                        logger.error(f"API错误 {response.status}: {error_text[:200]}")

                        if retry_count < self.max_retries:
                            logger.info(f"重试 {retry_count + 1}/{self.max_retries}: {input_text[:40]}...")
                            await asyncio.sleep(2 ** retry_count)
                            return await self.classify_single(session, input_text, retry_count + 1)
                        
                        return None
                        
            except asyncio.TimeoutError:
                logger.error(f"请求超时: {input_text[:40]}...")
                if retry_count < self.max_retries:
                    logger.info(f"重试 {retry_count + 1}/{self.max_retries}")
                    await asyncio.sleep(2 ** retry_count)
                    return await self.classify_single(session, input_text, retry_count + 1)
                return None

            except json.JSONDecodeError as je:
                logger.error(f"JSON解析失败: {input_text[:40]}... - {str(je)}")
                if retry_count < self.max_retries:
                    await asyncio.sleep(2 ** retry_count)
                    return await self.classify_single(session, input_text, retry_count + 1)
                return None

            except Exception as e:
                logger.error(f"分类失败: {input_text[:40]}... - {str(e)}")
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
            
        # 过滤掉失败的结果并统计
        valid_results = [r for r in results if r is not None]
        failed_count = len(inputs) - len(valid_results)

        # 统计各类别分布
        if valid_results:
            label_dist = {}
            for r in valid_results:
                label = r['label']
                label_dist[label] = label_dist.get(label, 0) + 1

            logger.info(f"\n分类完成: 成功 {len(valid_results)}/{len(inputs)}, 失败 {failed_count}")
            logger.info("分类分布:")
            for label, count in sorted(label_dist.items()):
                logger.info(f"  {label}: {count}")
        else:
            logger.warning(f"分类失败: 0/{len(inputs)} 成功")

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
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='训练数据生成和分类工具')
    parser.add_argument(
        '--config',
        default='config.json',
        help='配置文件路径（默认: config.json）'
    )
    parser.add_argument(
        '--input',
        nargs='+',
        help='要分类的标题列表（可选，用于测试）'
    )

    args = parser.parse_args()

    # 加载配置文件
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        logger.info("请创建 config.json 文件，包含以下字段:")
        logger.info("  - api_key: API密钥")
        logger.info("  - base_url: API基础URL")
        logger.info("  - model: 模型名称")
        logger.info("  - training_data_path: 训练数据文件路径")
        logger.info("  - max_retries: 最大重试次数（可选，默认3）")
        logger.info("  - concurrent_requests: 并发请求数（可选，默认5）")
        sys.exit(1)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"读取配置文件失败: {str(e)}")
        sys.exit(1)

    # 验证必需字段
    required_fields = ['api_key', 'base_url', 'model', 'training_data_path']
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        logger.error(f"配置文件缺少必需字段: {', '.join(missing_fields)}")
        sys.exit(1)

    # 待分类的标签页标题列表（示例或从命令行参数）
    if args.input:
        sample_inputs = args.input
    else:
        # 默认示例数据
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
        api_key=config['api_key'],
        base_url=config['base_url'],
        model=config['model'],
        max_retries=config.get('max_retries', 3),
        concurrent_requests=config.get('concurrent_requests', 5)
    )
    
    logger.info(f"开始分类 {len(sample_inputs)} 条标题...")

    # 批量分类
    results = await generator.classify_batch(sample_inputs)
    
    # 格式化结果（移除confidence字段，只保留input和label）
    formatted_results = [
        {"input": r["input"], "label": r["label"]}
        for r in results
    ]
    
    # 追加到训练数据
    if formatted_results:
        generator.append_to_training_data(formatted_results, config['training_data_path'])

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