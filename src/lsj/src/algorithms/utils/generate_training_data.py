"""
训练数据生成脚本
使用大模型对浏览器标签页标题进行分类，并添加到训练数据集中
"""

import json
import asyncio
import aiohttp
from typing import List, Dict, Optional, Any
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

    VALID_LABELS = ['News', 'Tools', 'Learning', 'Shopping', 'Social', 'Entertainment', 'Other']

    def __init__(
        self,
        models_config: Optional[List[Dict[str, Any]]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        concurrent_requests: int = 5
    ):
        """
        初始化生成器

        Args:
            models_config: 多模型配置列表（优先）
            api_key: 单模型API密钥（向后兼容）
            base_url: 单模型API基础URL
            model: 单模型名称
            max_retries: 最大重试次数
            concurrent_requests: 并发请求数
        """
        # 分类提示词（增强版）- 7大类别
        self.system_prompt = """你是一个专业的浏览器标签页分类专家。任务：根据“标签页标题”判断其核心用途，并且只能输出 7 个类别之一：
News, Tools, Learning, Shopping, Social, Entertainment, Other。
【输入】
- 仅提供一个标签页标题（可能中英混合、包含品牌名、短语、符号、emoji、版本号）
【输出】
- 只输出严格 JSON：
{
  "label": "News|Tools|Learning|Shopping|Social|Entertainment|Other",
  "confidence": 0.50-1.00之间的小数,
  "reasoning": "不超过30字，说明判定依据"
}
禁止输出任何额外文本、markdown、代码块、前后缀说明。
--------------------------------------------------
一、类别定义（含扩展关键词与典型站点）
--------------------------------------------------
1) News（新闻）
定义：以“报道、时效资讯、媒体新闻”为主的页面。
关键词（中英）：
新闻、快讯、头条、今日、最新、报道、资讯、专题、深度、要闻、突发、财经快报、科技资讯、国际、国内、观察、社评、live、breaking、headline、news、report
典型站点：
人民网、新华网、央视网、澎湃新闻、财新、界面新闻、第一财经、观察者网、BBC、CNN、Reuters、Bloomberg、WSJ、FT、TechCrunch、The Verge、Wired、36氪、虎嗅、钛媒体、少数派资讯
2) Tools（工具）
定义：以“完成任务/提高效率/开发协作”为核心目的的工具页面。
关键词（中英）：
工具、在线工具、转换、压缩、OCR、翻译、表格、文档协作、网盘、云盘、下载器、格式化、正则、API、调试、控制台、仓库、CI/CD、部署、监控、dashboard、workspace、editor、converter、generator、utility
典型站点：
GitHub、GitLab、Gitee、Postman、Apifox、Swagger、Jira、Confluence、Notion、飞书文档、腾讯文档、石墨文档、语雀、Trello、Asana、Google Drive、Dropbox、OneDrive、Figma、Canva、Vercel、Netlify、Cloudflare、CodePen、JSFiddle、在线 JSON/SQL/Markdown 工具
3) Learning（学习）
定义：以“学习知识/教程课程/文档查阅/学术检索”为核心。
关键词（中英）：
教程、课程、文档、指南、手册、示例、原理、入门、进阶、实践、训练营、题解、笔记、学习路径、论文、文献、综述、lecture、course、tutorial、guide、docs、reference、paper、arxiv、scholar
典型站点：
Python Docs、MDN、PyTorch Docs、TensorFlow Docs、OpenCV Docs、Coursera、edX、Udemy、Khan Academy、中国大学MOOC、慕课网、网易云课堂、极客时间、CSDN（教程页）、掘金（技术文章页）、博客园、Stack Overflow（知识问答）、arXiv、Google Scholar、知网、ResearchGate、ACL Anthology
4) Shopping（购物）
定义：以“商品浏览、比价、下单、支付、订单管理”为核心。
关键词（中英）：
商品、店铺、购物车、下单、支付、优惠、券、促销、满减、拼团、秒杀、比价、评价、物流、订单、退换货、price、deal、coupon、checkout、order、cart、buy
典型站点：
淘宝、天猫、京东、拼多多、唯品会、苏宁易购、亚马逊、eBay、AliExpress、Shopee、Temu、Apple Store、小米商城、华为商城、网易严选、闲鱼、转转、得物、小红书商城、抖音商城、美团闪购
5) Social（社交）
定义：以“人与人互动、社区讨论、社交关系”为核心。
关键词（中英）：
动态、关注、粉丝、评论、点赞、转发、私信、话题、圈子、社区、帖子、回答、讨论、互动、group、community、thread、post、comment、chat、dm、follow
典型站点：
微博、知乎（社区互动页）、豆瓣小组、贴吧、小红书（社区内容页）、即刻、Twitter/X、Facebook、Instagram、Reddit、Discord、Telegram、QQ、微信网页版、V2EX、Quora、NGA、虎扑
6) Entertainment（娱乐）
定义：以“休闲消费内容（看/听/玩）”为核心。
关键词（中英）：
视频、短视频、直播、音乐、MV、歌单、电影、电视剧、综艺、番剧、动漫、游戏、攻略、赛事、直播间、play、watch、stream、music、movie、show、game
典型站点：
B站、YouTube、抖音、快手、腾讯视频、爱奇艺、优酷、Netflix、Disney+、HBO、网易云音乐、QQ音乐、Spotify、Apple Music、Steam、Epic、TapTap、斗鱼、虎牙、Twitch、豆瓣电影（观影内容页）
7) Other（其他）
定义：系统页面、本地页面、错误页面、功能不明或信息不足页面。
关键词（中英）：
设置、配置、帮助、支持、反馈、关于、登录、注册、权限、404、500、错误、无法访问、本地文件、新标签页、历史记录、书签、extension、settings、about、help、error、file://、chrome://、edge://
典型场景：
浏览器设置页、扩展管理页、系统控制台、空白页、新标签页、下载页、登录授权中转页、验证码页、文件预览页
--------------------------------------------------
二、判定规则（高一致性）
--------------------------------------------------
1. 按“当前页面核心用途”分类，不按网站品牌名机械分类。  
2. 出现多重特征时，按以下优先级决策（从高到低）：
   Shopping > Social > Entertainment > Learning > Tools > News > Other
3. 例外修正：
   - “科技媒体报道”优先 News（不是 Learning）
   - “官方文档/教程页”优先 Learning（即使站点是开发平台）
   - “代码仓库/项目管理/在线编辑器”优先 Tools
   - “纯登录/错误/设置/跳转页”优先 Other
4. 信息不足时归 Other，confidence 不高于 0.65。
5. 必须返回且只返回一个标签。
--------------------------------------------------
三、置信度标准
--------------------------------------------------
- 0.90-1.00：标题语义非常明确，几乎无歧义
- 0.75-0.89：基本明确，有轻微歧义
- 0.60-0.74：存在明显歧义，但可判断
- 0.50-0.59：信息较弱，仅弱判断
- confidence 必须在 [0.50, 1.00]
--------------------------------------------------
四、输出格式（严格）
--------------------------------------------------
仅输出 JSON 对象：
{
  "label": "News|Tools|Learning|Shopping|Social|Entertainment|Other",
  "confidence": 0.87,
  "reasoning": "核心用途是xxx"
}"""

        # 多模型模式
        if models_config:
            self.models_config = models_config
            self.multi_model_mode = True
            self.current_model_index = 0
            total_concurrent = sum(m.get('concurrent_requests', 5) for m in models_config)
            self.concurrent_requests = total_concurrent
            self.semaphore = asyncio.Semaphore(total_concurrent)
            logger.info(f"多模型模式：{len(models_config)} 个模型，总并发数: {total_concurrent}")
        else:
            # 单模型模式（向后兼容）
            self.models_config = [{
                'name': 'default',
                'api_key': api_key,
                'base_url': base_url.rstrip('/') if base_url else '',
                'model': model,
                'max_retries': max_retries,
                'concurrent_requests': concurrent_requests
            }]
            self.multi_model_mode = False
            self.current_model_index = 0
            self.concurrent_requests = concurrent_requests
            self.semaphore = asyncio.Semaphore(concurrent_requests)
            logger.info("单模型模式")

    def _get_next_model(self) -> Dict[str, Any]:
        """轮询获取下一个模型配置"""
        model_cfg = self.models_config[self.current_model_index]
        self.current_model_index = (self.current_model_index + 1) % len(self.models_config)
        return model_cfg

    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        """尽量稳健地从模型响应中提取JSON"""
        text = content.strip()

        # 去掉 markdown code block
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        # 优先直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试截取第一个 {...}
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
            raise

    async def classify_single(
        self,
        session: aiohttp.ClientSession,
        input_text: str
    ) -> Optional[Dict[str, Any]]:
        """
        分类单个标签页标题（循环重试版，避免递归重入）
        """
        model_config = self._get_next_model()
        api_key = model_config.get('api_key')
        base_url = (model_config.get('base_url') or '').rstrip('/')
        model = model_config.get('model')
        model_name = model_config.get('name', 'unknown')
        max_retries = int(model_config.get('max_retries', 3))

        if not api_key or not base_url or not model:
            logger.error(f"[{model_name}] 模型配置不完整，跳过: api_key/base_url/model")
            return None

        for attempt in range(max_retries + 1):
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"请分析并分类这个标签页标题：\n\n标题：{input_text}\n\n请给出分类、置信度和理由。"}
                    ],
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 200
                }

                async with self.semaphore:
                    async with session.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:

                        if response.status == 200:
                            result = await response.json()
                            content = result['choices'][0]['message']['content']
                            classification = self._extract_json(content)

                            label = classification.get('label')
                            if label not in self.VALID_LABELS:
                                logger.warning(f"[{model_name}] 无效分类标签: {label}, 标题: {input_text[:50]}")
                                return None

                            confidence = classification.get('confidence', 1.0)
                            try:
                                confidence = float(confidence)
                            except (ValueError, TypeError):
                                confidence = 1.0

                            logger.info(
                                f"✓ [{model_name}] 分类成功: {input_text[:40]}... -> {label} (置信度: {confidence:.2f})"
                            )
                            return {
                                "input": input_text,
                                "label": label,
                                "confidence": confidence
                            }

                        # 非200响应
                        error_text = await response.text()
                        if response.status == 429:
                            wait_time = 2 ** (attempt + 1)
                            logger.warning(f"[{model_name}] 触发速率限制，等待{wait_time}秒: {input_text[:40]}...")
                            await asyncio.sleep(wait_time)
                            continue

                        logger.error(f"[{model_name}] API错误 {response.status}: {error_text[:200]}")
                        if attempt < max_retries:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return None

            except asyncio.TimeoutError:
                logger.error(f"[{model_name}] 请求超时: {input_text[:40]}...")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None

            except json.JSONDecodeError as je:
                logger.error(f"[{model_name}] JSON解析失败: {input_text[:40]}... - {je}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None

            except Exception as e:
                logger.error(f"[{model_name}] 分类失败: {input_text[:40]}... - {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None

        return None

    async def classify_batch(self, inputs: List[str]) -> List[Dict[str, Any]]:
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

        # 过滤失败结果并统计
        valid_results = [r for r in results if r is not None]
        failed_count = len(inputs) - len(valid_results)

        if valid_results:
            label_dist: Dict[str, int] = {}
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
        new_data: List[Dict[str, Any]],
        training_data_path: str
    ) -> None:
        """
        将新数据追加到训练数据文件

        Args:
            new_data: 新的训练数据（每项至少包含 input,label）
            training_data_path: 训练数据文件路径
        """
        try:
            path = Path(training_data_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # 读取现有数据
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        logger.warning("已有训练数据文件格式异常（非list），将重置为空列表")
                        existing_data = []
            else:
                existing_data = []

            # 去重：按 input 去重
            existing_inputs = {item.get('input') for item in existing_data if isinstance(item, dict)}
            unique_new_data = [item for item in new_data if item.get('input') not in existing_inputs]

            if not unique_new_data:
                logger.warning("没有新的唯一数据需要添加")
                return

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

    # 加载配置文件（相对当前脚本目录）
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        logger.info("请创建 config.json")
        sys.exit(1)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"读取配置文件失败: {str(e)}")
        sys.exit(1)

    # 待分类标题
    if args.input:
        sample_inputs = args.input
    else:
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

    # 统一检查 training_data_path
    if 'training_data_path' not in config:
        logger.error("配置文件缺少必需字段: training_data_path")
        sys.exit(1)

    enable_multi_model = config.get('enable_multi_model', False)

    if enable_multi_model:
        # 多模型模式校验
        models = config.get('models')
        if not isinstance(models, list) or not models:
            logger.error("多模型模式下缺少 models 配置或为空")
            sys.exit(1)

        for idx, m in enumerate(models):
            for field in ['api_key', 'base_url', 'model']:
                if not m.get(field):
                    logger.error(f"models[{idx}] 缺少必需字段: {field}")
                    sys.exit(1)

        logger.info(f"启用多模型模式，共 {len(models)} 个模型")
        generator = TrainingDataGenerator(models_config=models)

    else:
        # 单模型模式校验
        required_fields = ['api_key', 'base_url', 'model', 'training_data_path']
        missing_fields = [field for field in required_fields if not config.get(field)]
        if missing_fields:
            logger.error(f"配置文件缺少必需字段: {', '.join(missing_fields)}")
            sys.exit(1)

        logger.info("使用单模型模式")
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

    # 格式化结果（移除confidence，仅保留input和label）
    formatted_results = [{"input": r["input"], "label": r["label"]} for r in results]

    # 追加训练数据
    if formatted_results:
        generator.append_to_training_data(formatted_results, config['training_data_path'])

        # 打印统计
        label_counts: Dict[str, int] = {}
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