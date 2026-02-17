import os
import sqlite3
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import urllib.parse as urlparse
import sys
import logging

def get_chrome_history_path():
    if sys.platform == 'win32':
        path = os.path.expanduser(r"~\AppData\Local\Google\Chrome\User Data\Default\History")
    elif sys.platform == 'darwin':
        path = os.path.expanduser(r"~/Library/Application Support/Google/Chrome/Default/History")
    elif sys.platform == 'linux':
        path = os.path.expanduser(r"~/.config/google-chrome/Default/History")
    else:
        return None

    if not os.path.exists(path):
        print("未能找到默认路径下的 History 文件: {}".format(path))
        return None
    return path

def safe_extract_history():
    logs_folder_path = "../../logs"
    if not os.path.exists(logs_folder_path):
        os.makedirs(logs_folder_path)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../../logs/fetch_data.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    # logger.info("Test1")
    # logger.error("Test2")
    # logger.warning("Test3")
    # logger.exception("Test4")

    logger.info("=" * 50)
    logger.info("开始提取历史文件数据")
    # 获取路径
    try:
        original_path = get_chrome_history_path()
        if original_path is None:
            logger.error("无法获取 Chrome 历史文件路径")
            return None
        logger.info(f"找到历史文件: {original_path}")
    except Exception as e:
        logger.exception(f"获取路径时发生错误: {e}")
        return None

    # 复制文件
    temp_dir = "..\\cache"
    try:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            logger.info(f"创建临时目录: {temp_dir}")
    except Exception as e:
        logger.exception(f"无法创建临时目录: {e}")
        return None

    temp_path = os.path.join(temp_dir, "chrome_temp_history.db")
    try:
        shutil.copy2(original_path, temp_path)
        logger.info(f"成功复制历史文件到: {temp_path}")
    except PermissionError:
        logger.error("无法复制文件: Chrome可能正在运行， 请关闭Chrome后重试")
        return None
    except OSError as e:
        logger.error(f"文件复制失败: {e}")
        return None
    except Exception as e:
        logger.exception(f"文件复制时发生未知错误: {e}")
        return None

    # 连接数据库，读取数据
    conn = None
    df = None

    try:
        conn = sqlite3.connect(temp_path)
        logger.info("成功连接到数据库")

        query = """
                SELECT urls.url, \
                       urls.title, \
                       urls.visit_count, \
                       urls.last_visit_time
                FROM urls
                WHERE urls.last_visit_time > 0 \
                """

        df = pd.read_sql_query(query, conn)
        logger.info(f"成功读取 {len(df)} 条历史记录")
    except sqlite3.Error as e:
        logger.error(f"数据库操作失败: {e}")
        df = None
    except pd.errors.DatabaseError as e:
        logger.error(f"数据库读取失败: {e}")
        df = None
    except Exception as e:
        logger.exception(f"查询数据时发生未知错误 {e}")
        df = None

    finally:
        if conn:
            conn.close()
            logger.info("数据库连接已关闭")

        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("临时文件已删除")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {e}")

    # 验证数据
    def validate_data(_df):
        if _df is None:
            logger.error("数据为 None")
            return False

        if _df.empty:
            logger.warning("数据为空，没有历史记录")
            return False

        required_columns = ['url', 'title', 'visit_count', 'last_visit_time']
        missing_columns = [col for col in required_columns if col not in _df.columns]

        if missing_columns:
            logger.error("缺少必要的列")
            return False

        if not pd.api.types.is_numeric_dtype(_df['last_visit_time']):
            logger.error("\'last_visit_time\' 列不是数字类型")
            return False

        logger.info("数据验证通过")
        return True

    if not validate_data(df):
        logger.error("数据验证失败")
        return None

    logger.info("历史数据提取完成")
    logger.info("=" * 50)
    return df

def process_history_time():
    def get_local_timezone():
        local_offset_seconds = -time.timezone
        if time.daylight and time.localtime().tm_isdst > 0:
            local_offset_seconds -= time.altzone
        return timezone(timedelta(seconds=local_offset_seconds))

    def convert_to_local_time(df, col_name='visit_time'):
        local_timezone = get_local_timezone()

        if df[col_name].dt.tz is None:
            df[col_name] = df[col_name].dt.tz_localize('UTC')
        else:
            df[col_name] = df[col_name].dt.tz_convert('UTC')

        df[col_name] = df[col_name].dt.tz_convert(local_timezone)

        return df

    def process_history(df):
        if df is None or df.empty:
            return None

        def parse_webkit_time(microseconds):
            return datetime(1601, 1, 1) + timedelta(microseconds=microseconds)

        df['visit_time'] = df['last_visit_time'].apply(parse_webkit_time)
        df = convert_to_local_time(df, 'visit_time')

        df['domain'] = df['url'].apply(lambda url: urlparse.urlparse(url).netloc)
        df['hour'] = df['visit_time'].dt.hour

        return df

def save_as_csv(df, output_path):
    folder_path = output_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    new_df = pd.DataFrame({
        'domain': df['domain'].tolist(),
        'visit_time': df['visit_time'].tolist(),
        'hour' : df['hour'].tolist()
    })

    file_path = os.path.join(folder_path, 'history_data.csv')
    new_df.to_csv(file_path, index=False)

def analyze_and_plot(df): #ai生成的: 分析数据并绘图
    if df is None:
        return

    print(f"共加载了 {len(df)} 条访问记录。\n")

    # --- 分析 1: 访问次数最多的前 10 个域名 ---
    top_domains = df['domain'].value_counts().head(10)
    print("=== 访问最多的网站 (Top 10) ===")
    print(top_domains)

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False

    # 绘图 1: Top 10 域名
    plt.figure(figsize=(12, 6))
    top_domains.plot(kind='bar', color='skyblue')
    plt.title('访问次数最多的 Top 10 网站')
    plt.xlabel('域名')
    plt.ylabel('访问次数')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # --- 分析 2: 一天中哪个时间段最活跃 ---
    hourly_activity = df['hour'].value_counts().sort_index()

    # 绘图 2: 每日活跃时间分布
    plt.figure(figsize=(12, 6))
    hourly_activity.plot(kind='line', marker='o', color='coral')
    plt.title('每日上网活跃时间分布 (24小时制)')
    plt.xlabel('小时 (0-23)')
    plt.ylabel('访问页面数')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # print("开始读取 Chrome 浏览记录...")
    # raw_df = extract_history()
    #
    # if raw_df is not None:
    #     clean_df = process_history(raw_df)
    #     save_as_csv(clean_df, "../../tests/files")
    #     analyze_and_plot(clean_df)
    safe_extract_history()