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
    
    # 类别描述和示例（增强版）- 7大类别
    CATEGORY_INFO = {
        "News": {
            "description": "新闻、资讯",
            "subcategories": [
                {"name": "综合新闻", "examples": ["人民网头条", "新华网要闻", "央视新闻", "澎湃新闻"]},
                {"name": "科技资讯", "examples": ["36氪科技", "虎嗅网", "TechCrunch", "The Verge"]},
                {"name": "财经金融", "examples": ["华尔街日报", "财新网", "Bloomberg", "东方财富网"]},
                {"name": "国际新闻", "examples": ["BBC News", "CNN国际", "路透社", "法新社"]},
                {"name": "地方资讯", "examples": ["北京日报", "上海观察", "南方都市报", "齐鲁晚报"]}
            ],
            "title_patterns": [
                "{事件标题} - {媒体名称} {日期}",
                "{地区/领域}最新动态：{具体事件}",
                "快讯：{突发新闻} | {新闻源}",
                "{专题报道} 深度解析 - {媒体}"
            ]
        },
        "Tools": {
            "description": "工具、实用程序",
            "subcategories": [
                {"name": "在线工具", "examples": ["Google翻译", "PDF转换器", "图片压缩", "代码格式化"]},
                {"name": "云存储", "examples": ["百度网盘", "Google Drive", "Dropbox", "OneDrive"]},
                {"name": "办公协作", "examples": ["腾讯文档", "石墨文档", "Notion", "飞书文档"]},
                {"name": "开发工具", "examples": ["GitHub代码仓库", "GitLab CI/CD", "Postman API", "VS Code插件"]},
                {"name": "效率工具", "examples": ["Trello看板", "Todoist待办", "印象笔记", "Evernote"]}
            ],
            "title_patterns": [
                "{工具名称} - {具体功能}在线工具",
                "{文件名} - {云存储平台}分享",
                "{文档标题} - {协作平台}编辑中",
                "{项目名} - {开发工具}工作区"
            ]
        },
        "Learning": {
            "description": "学习、教育、文档",
            "subcategories": [
                {"name": "编程开发", "examples": ["Python官方文档", "MDN Web Docs", "Stack Overflow问答", "GitHub仓库README"]},
                {"name": "在线课程", "examples": ["Coursera机器学习", "中国大学MOOC", "网易云课堂", "Udemy课程"]},
                {"name": "技术博客", "examples": ["CSDN技术博客", "掘金社区", "Medium文章", "Dev.to教程"]},
                {"name": "学术研究", "examples": ["arXiv论文预印本", "Google Scholar", "知网文献", "ResearchGate"]},
                {"name": "语言学习", "examples": ["多邻国Duolingo", "扇贝单词", "Quizlet记忆卡", "italki语言交换"]}
            ],
            "title_patterns": [
                "{技术主题} 完整教程 - {作者/平台}",
                "{编程语言} {版本} 官方文档 - {章节}",
                "如何实现{功能} - {技术栈} 最佳实践",
                "{课程名} 第{N}章 - {平台}在线学习"
            ]
        },
        "Shopping": {
            "description": "购物、电商网站",
            "subcategories": [
                {"name": "综合电商", "examples": ["淘宝商品详情", "京东购物车", "亚马逊Prime会员", "拼多多百亿补贴"]},
                {"name": "品牌官网", "examples": ["Apple Store官网", "Nike官方旗舰店", "小米商城新品", "华为商城折扣"]},
                {"name": "二手交易", "examples": ["闲鱼二手交易", "转转验机服务", "eBay拍卖", "Mercari日本代购"]},
                {"name": "垂直电商", "examples": ["唯品会特卖", "网易严选", "小红书商城", "得物App鉴定"]},
                {"name": "跨境电商", "examples": ["速卖通全球购", "Shein时尚", "Wish特价", "洋码头海淘"]}
            ],
            "title_patterns": [
                "{品牌/商品} - {具体型号/规格} | {平台名}",
                "{促销活动} - {商品类别} 限时优惠 - {平台}",
                "{商品名称} 用户评价/晒单 - {平台}",
                "购物车({数量}件) - {平台}结算中心"
            ]
        },
        "Social": {
            "description": "社交媒体、论坛、社区",
            "subcategories": [
                {"name": "社交网络", "examples": ["微博热搜", "Twitter推文", "Facebook动态", "Instagram照片"]},
                {"name": "即时通讯", "examples": ["微信聊天", "QQ空间", "Telegram群组", "Discord服务器"]},
                {"name": "论坛社区", "examples": ["知乎问答", "Reddit讨论", "豆瓣小组", "贴吧"]},
                {"name": "问答平台", "examples": ["Stack Overflow", "Quora", "SegmentFault", "V2EX"]},
                {"name": "内容社区", "examples": ["小红书笔记", "即刻动态", "少数派", "什么值得买"]}
            ],
            "title_patterns": [
                "{用户名}的{内容类型} - {平台}",
                "{话题标签} 热门讨论 - {社区}",
                "{问题标题} - {回答数}个回答 | {平台}",
                "{群组/频道名称} - {平台}消息({未读数})"
            ]
        },
        "Entertainment": {
            "description": "娱乐、视频、音乐、游戏",
            "subcategories": [
                {"name": "视频平台", "examples": ["B站番剧", "YouTube视频", "抖音短视频", "爱奇艺电影"]},
                {"name": "音乐流媒体", "examples": ["网易云音乐", "QQ音乐", "Spotify", "Apple Music"]},
                {"name": "游戏相关", "examples": ["Steam游戏商店", "原神官网", "英雄联盟", "TapTap游戏社区"]},
                {"name": "影视剧集", "examples": ["豆瓣电影", "腾讯视频", "Netflix", "Disney+"]},
                {"name": "直播平台", "examples": ["斗鱼直播", "虎牙直播", "Twitch", "快手直播"]}
            ],
            "title_patterns": [
                "{视频标题} - {UP主/频道} - {平台}",
                "{歌曲名} - {歌手} | {音乐平台}",
                "{游戏名} {版本/活动} - {游戏平台}",
                "{影视剧名} 第{N}集 - {视频平台}在线观看"
            ]
        },
        "Other": {
            "description": "其他未分类内容",
            "subcategories": [
                {"name": "系统设置", "examples": ["Windows设置", "Mac系统偏好", "浏览器设置", "扩展管理"]},
                {"name": "本地文件", "examples": ["文件管理器", "本地HTML", "PDF阅读", "图片查看"]},
                {"name": "帮助文档", "examples": ["软件帮助", "用户手册", "FAQ页面", "关于页面"]},
                {"name": "错误页面", "examples": ["404错误", "500错误", "网络错误", "加载失败"]},
                {"name": "其他", "examples": ["空白页", "新标签页", "书签管理", "历史记录"]}
            ],
            "title_patterns": [
                "{系统/软件}设置 - {具体选项}",
                "{文件名}.{扩展名} - 本地文件",
                "{错误代码} - {错误描述}",
                "{功能名称} - 帮助文档"
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

    def _is_quality_title(self, title: str) -> bool:
        """检查标题质量"""
        # 长度检查
        if len(title) < 10 or len(title) > 150:
            return False

        # 过滤明显的模板化标题
        template_patterns = [
            '标题1', '标题2', '示例', 'example', 'test',
            '网站名称', '页面标题', '内容标题'
        ]
        title_lower = title.lower()
        if any(pattern in title_lower for pattern in template_patterns):
            return False

        # 必须包含一些实质内容（中文或英文单词）
        import re
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', title))
        has_english = bool(re.search(r'[a-zA-Z]{2,}', title))

        if not (has_chinese or has_english):
            return False

        # 检查是否过于重复（同一字符连续出现）
        if re.search(r'(.)\1{4,}', title):
            return False

        return True

    def _calculate_similarity(self, title1: str, title2: str) -> float:
        """计算两个标题的相似度（简单版本）"""
        # 使用Jaccard相似度
        set1 = set(title1.lower())
        set2 = set(title2.lower())

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

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
        batch_size: int = 15
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
        quality_filtered = 0
        similarity_filtered = 0

        # 分批生成，每批使用不同的子类别和场景
        batches = (count + batch_size - 1) // batch_size
        subcategories = cat_info['subcategories']

        for batch_idx in range(batches):
            batch_count = min(batch_size, count - len(all_titles))
            if batch_count <= 0:
                break

            # 动态调整temperature以增加多样性
            temperature = 1.0 + (batch_idx * 0.05)
            temperature = min(temperature, 1.3)

            # 轮换子类别以增加多样性
            subcat = subcategories[batch_idx % len(subcategories)]

            # 构建更详细的提示词 - 直接生成带label的训练数据格式
            prompt = f"""你是一个浏览器标签页训练数据生成专家。请生成{batch_count}条{category}类别中{subcat['name']}子类别的训练数据。

## 类别说明
类别: {category}
子类别: {subcat['name']}
示例: {', '.join(subcat['examples'])}

## 标题格式参考
{chr(10).join('- ' + pattern for pattern in cat_info['title_patterns'])}

## 生成要求

### 真实性要求
1. 标题必须像真实用户浏览器中的标签页，不能是虚构或模板化的
2. 包含真实存在的网站、平台、品牌名称
3. 体现用户的实际使用场景（正在浏览、搜索、购买、学习等）

### 多样性要求
1. 中英文网站各占50%，自然混合
2. 包含具体细节：
   - 数字（价格、数量、版本号、集数等）
   - 日期时间（2024年、最新、今日等）
   - 状态词（进行中、已完成、限时等）
   - 具体名称（产品型号、文章标题、视频名等）
3. 不同的标题结构和长度（20-100字符）

### 质量要求
1. 避免使用"标题"、"示例"、"网站名称"等占位符
2. 每个标题都要独特，不能相似
3. 标题要完整，不能截断或省略
4. 符合该类别的典型特征

### 本批次特殊要求
- 重点关注{subcat['name']}场景
- 参考示例：{', '.join(subcat['examples'][:3])}
- 生成更加具体和细节化的标题

## 输出格式
请以JSON数组格式返回训练数据，每条数据包含input和label字段。
label必须是以下7个类别之一：News, Tools, Learning, Shopping, Social, Entertainment, Other

只返回JSON数组，不要有任何其他文字：
[
  {{
    "input": "具体的标签页标题1",
    "label": "{category}"
  }},
  {{
    "input": "具体的标签页标题2",
    "label": "{category}"
  }}
]

现在开始生成{batch_count}条高质量的训练数据。"""

            titles = await self._call_api_for_titles(prompt, temperature)

            # 多层过滤：质量 -> 去重 -> 相似度
            filtered_titles = []
            batch_quality_filtered = 0
            batch_duplicate_filtered = 0
            batch_similarity_filtered = 0

            for title in titles:
                # 1. 质量检查
                if not self._is_quality_title(title):
                    batch_quality_filtered += 1
                    continue

                # 2. 精确去重
                if self._is_duplicate(title):
                    batch_duplicate_filtered += 1
                    continue

                # 3. 相似度检查（与已有标题对比）
                is_similar = False
                for existing_title in list(all_titles)[-50:]:  # 只检查最近50个
                    if self._calculate_similarity(title, existing_title) > 0.7:
                        is_similar = True
                        batch_similarity_filtered += 1
                        break

                if not is_similar:
                    filtered_titles.append(title)
                    self._add_title(title)

            quality_filtered += batch_quality_filtered
            similarity_filtered += batch_similarity_filtered

            all_titles.extend(filtered_titles)
            logger.info(
                f"✓ {category} 批次{batch_idx+1}/{batches}: "
                f"生成{len(titles)} -> 质量过滤{batch_quality_filtered} -> "
                f"去重{batch_duplicate_filtered} -> 相似度过滤{batch_similarity_filtered} -> "
                f"保留{len(filtered_titles)}"
            )

            # 避免速率限制
            if batch_idx < batches - 1:
                await asyncio.sleep(1)

        logger.info(
            f"\n{category} 总计: 保留{len(all_titles)}个标题 "
            f"(质量过滤: {quality_filtered}, 相似度过滤: {similarity_filtered})"
        )
        return all_titles

    async def _call_api_for_titles(
        self,
        prompt: str,
        temperature: float = 0.9,
        max_retries: int = 3
    ) -> List[str]:
        """调用API生成标题（带重试和降级策略）"""

        for retry in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }

                    # 重试时降低temperature
                    adjusted_temp = temperature - (retry * 0.1)

                    payload = {
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": max(0.7, adjusted_temp),
                        "response_format": {"type": "json_object"},
                        "max_tokens": 2000
                    }

                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=90)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            content = result['choices'][0]['message']['content']

                            # 尝试解析JSON - 新格式是数组
                            try:
                                data = json.loads(content)

                                # 支持两种格式：新格式（数组）和旧格式（对象）
                                if isinstance(data, list):
                                    # 新格式：直接是训练数据数组
                                    titles = [item['input'] for item in data if 'input' in item]
                                elif isinstance(data, dict) and 'titles' in data:
                                    # 旧格式：兼容性支持
                                    titles = data['titles']
                                else:
                                    titles = []

                                # 验证返回的标题
                                if titles and isinstance(titles, list):
                                    return titles
                                else:
                                    logger.warning(f"API返回格式异常，重试 {retry+1}/{max_retries}")

                            except json.JSONDecodeError as je:
                                logger.error(f"JSON解析失败: {str(je)}, 内容: {content[:200]}")
                                if retry < max_retries - 1:
                                    await asyncio.sleep(2 ** retry)
                                    continue

                        elif response.status == 429:  # Rate limit
                            wait_time = 2 ** (retry + 1)
                            logger.warning(f"触发速率限制，等待{wait_time}秒后重试")
                            await asyncio.sleep(wait_time)
                            continue

                        else:
                            error_text = await response.text()
                            logger.error(f"API错误 {response.status}: {error_text[:200]}")

                            if retry < max_retries - 1:
                                await asyncio.sleep(2 ** retry)
                                continue

            except asyncio.TimeoutError:
                logger.error(f"API请求超时，重试 {retry+1}/{max_retries}")
                if retry < max_retries - 1:
                    await asyncio.sleep(2 ** retry)
                    continue

            except Exception as e:
                logger.error(f"API调用失败: {str(e)}")
                if retry < max_retries - 1:
                    await asyncio.sleep(2 ** retry)
                    continue

        logger.error(f"API调用失败，已重试{max_retries}次")
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
