import os
import sqlite3
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import urllib.parse as urlparse
import sys

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

def extract_history():
    original_path = get_chrome_history_path()
    if not original_path:
        return None

    temp_path = "../cache/temp_history_db"
    try:
        shutil.copy2(original_path, temp_path)
    except PermissionError:
        print("无法复制文件")
        return None

    conn = sqlite3.connect(temp_path)
    cursor = conn.cursor()

    query = """
    SELECT 
        urls.url, 
        urls.title, 
        urls.visit_count, 
        urls.last_visit_time 
    FROM urls 
    WHERE urls.last_visit_time > 0
    """

    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"数据库读取错误:{e}")
        return None
    finally:
        conn.close()
        os.remove(temp_path)

    return df

def process_history(df):
    if df is None or df.empty:
        return None

    def parse_webkit_time(microseconds):
        return datetime(1601, 1, 1) + timedelta(microseconds=microseconds)

    df['visit_time'] = df['last_visit_time'].apply(parse_webkit_time)
    df['visit_time'] = df['visit_time'] + timedelta(hours=8)

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
    print("开始读取 Chrome 浏览记录...")
    raw_df = extract_history()

    if raw_df is not None:
        clean_df = process_history(raw_df)
        save_as_csv(clean_df, "../../tests/files")
        analyze_and_plot(clean_df)