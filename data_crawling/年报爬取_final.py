import pandas as pd
import re
import warnings
import pickle
warnings.filterwarnings("ignore", category=UserWarning)
##### 数据导入
df_600 = pd.read_excel('/hy-tmp/data/d上交所主板_年报链接全.xlsx')
print(f'上交所主板：数据框形状{df_600.shape}，股票个数{len(df_600["股票代码"].unique())}')
df_688 = pd.read_excel('/hy-tmp/data/d上交所科创板_年报链接全.xlsx')
print(f'上交所科创板：数据框形状{df_688.shape}，股票个数{len(df_688["股票代码"].unique())}')
df_000 = pd.read_excel('/hy-tmp/data/d深交所主板_年报链接全.xlsx')
print(f'深交所主板：数据框形状{df_000.shape}，股票个数{len(df_000["股票代码"].unique())}')
df_300 = pd.read_excel('/hy-tmp/data/d深交所创业板_年报链接全.xlsx')
print(f'深交所创业板：数据框形状{df_300.shape}，股票个数{len(df_300["股票代码"].unique())}')
# 读取目标数据框
df_targets = pd.read_excel('/hy-tmp/data/final_公司年报列表_dropST金融科研_去不平衡_28483条_1994家企业.xlsx', dtype={'股票代码': str})
# with open('/hy-tmp/gpt/targets.pickle', 'rb') as handle:
#     df_targets = pickle.load(handle)
print(f'目标数据框：数据框形状{df_targets.shape}，股票个数{len(df_targets["股票代码"].unique())}')
save_list = df_targets['股票代码'].unique().tolist()

##### 数据处理和筛选
def change_year(df):
    df['年份'] = df['报告名称'].str.extract(r'([O〇○零一二三四五六七八九0-9]{4})年')
    hanzi_to_num = {
        '零': '0', '〇': '0', '○': '0', 'O': '0',
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'
    }   # 汉字数字转阿拉伯数字
    df['年份'] = df['年份'].apply(lambda x: ''.join(hanzi_to_num[char] if char in hanzi_to_num else char for char in x) if pd.notna(x) else x)
    df['年份'] = pd.to_numeric(df['年份'], errors='coerce')
    df = df[(df['年份'] >= 2007) | (df['年份'].isna())]   # 删除'年份'在2007年之前的年报（但不删除空值）
    grouped = df.groupby('股票代码')
    def linear_interpolate(group):
        group['年份'] = group['年份'].interpolate(method='linear').round().astype('Int64')  # 线性插值后四舍五入到整数
        return group
    df = grouped.apply(linear_interpolate)
    df = df.reset_index(drop = True)
    return df

df_600 = change_year(df_600)
df_688 = change_year(df_688)
df_000 = change_year(df_000)
df_300 = change_year(df_300)

def select_reports(group):
    reports_with_bracket = group[group['报告名称'].str.contains(r'[()（）]')]  # 存在任意之一即保留
    if not reports_with_bracket.empty:
        return reports_with_bracket
    else:
        return group

def drop_duplicate(df):
    # 股票代码删除标点符号和英文字符，'(600000.SH)'转换为'600000'
    df['股票代码'] = df['股票代码'].str.extract(r'(\d{6})')
    print('删除重复年报前：', df.shape)
    new_df = pd.DataFrame()
    for year in range(2007, 2024):
        data = df[df['年份'] == year]
        print(f'{year}年年报总数：{data.shape[0]}')
        ID_LIST = data['股票代码'].unique().tolist()
        print(f'{year}年公司数量：{len(ID_LIST)}')
        duplicated_rows = data[data['股票代码'].duplicated(keep=False)]
        print(f'{year}年重复年报数量：{duplicated_rows.shape}')
        print(f"{year}年重复公司数量：{len(duplicated_rows['股票代码'].unique().tolist())}")
        # 第一个筛选逻辑，对于duplicated_rows数据框每家公司（按照'股票代码区分'）重复的股票年报，我们首先删除'报告名称'中存在'英文'、'财务'、'H股'字样的年报
        filtered_rows_1 = duplicated_rows[~duplicated_rows['报告名称'].str.contains('英文|财务|H股')]
        print(f'{year}年第一次筛选后年报数量：{filtered_rows_1.shape}')
        print(f"{year}年第一次筛选后公司数量：{len(filtered_rows_1['股票代码'].unique().tolist())}")
        # 第二个筛选逻辑，对于每家公司剩余的年报，我们优先保留存在'('和'（'字符的（代表修改后），如果不存在则保留剩余的年报
        filtered_rows_2 = filtered_rows_1.groupby('股票代码').apply(select_reports).reset_index(drop=True)
        print(f'{year}年第二次筛选后年报数量：{filtered_rows_2.shape}')
        print(f"{year}年第二次筛选后年报数量：{len(filtered_rows_2['股票代码'].unique().tolist())}")
        # 第三个筛选逻辑，在上一步操作后，还存在一些公司有重复的年报，我们对这些重复的年报再做一次筛选操作，按照'股票代码'分组，我们只保留每家公司的第一行数据
        final_rows = filtered_rows_2.groupby('股票代码').first().reset_index()
        print(f'{year}年第三次筛选后年报数量：{final_rows.shape}')
        print(f"{year}年第三次筛选后年报数量：{len(final_rows['股票代码'].unique().tolist())}")
        # 得到需要删除的行
        to_delete = pd.concat([duplicated_rows, final_rows]).drop_duplicates(keep=False)
        if len(to_delete) == len(duplicated_rows) - len(final_rows):
            print(f'需要删除的数据有{len(to_delete)}条')
        # 从 data 中删除这些行
        data = pd.concat([data, to_delete]).drop_duplicates(keep=False)
        print(data.shape)
        print(len(data['股票代码'].unique().tolist()))
        new_df = pd.concat([new_df, data])
    print('删除重复年报后：', new_df.shape)
    # 按照股票代码升序、年份降序
    new_df = new_df.sort_values(by=['股票代码', '年份'], ascending=[True, False])
    new_df = new_df.reset_index(drop=True)
    return new_df

df_600_drop = drop_duplicate(df_600)
df_688_drop = drop_duplicate(df_688)
df_000_drop = drop_duplicate(df_000)
df_300_drop = drop_duplicate(df_300)

# df_600_drop = targets_df[targets_df['股票代码'].str.startswith('60')].copy().reset_index(drop=True)
# df_600_drop = pd.merge(df_600_drop, df_600['股票代码', '年报', '年报链接'], on=['股票代码', '年份'], how='left')
# df_688_drop = targets_df[targets_df['股票代码'].str.startswith('68')].copy().reset_index(drop=True)
# df_000_drop = targets_df[targets_df['股票代码'].str.startswith('00')].copy().reset_index(drop=True)
# df_300_drop = targets_df[targets_df['股票代码'].str.startswith('30')].copy().reset_index(drop=True)
data_dict = {
    '上交所主板': df_600_drop,
    '上交所科创板': df_688_drop,
    '深交所主板': df_000_drop,
    '深交所创业板': df_300_drop
}

##### 爬虫
# 配置驱动
from selenium import webdriver
import selenium.webdriver.support.wait as WA
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
options = Options()
options.add_argument('--headless')  # 无头模式
options.add_argument('--disable-gpu')  # 禁用GPU加速
options.add_argument('--no-sandbox')  # 禁用沙盒（在Linux系统中非常重要）
options.add_argument("--disable-dev-shm-usage")  # 禁用/dev/shm使用
options.add_argument("--remote-debugging-port=9222")  # 远程调试端口
options.add_argument("--verbose")
options.add_argument("--log-level=0")
prefs = {
    # "download.default_directory": download_path,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "plugins.always_open_pdf_externally": True
}
options.add_experimental_option("prefs", prefs)

edge_driver_path = r'/usr/local/bin/msedgedriver'
service = Service(edge_driver_path)
driver = webdriver.Edge(service=service, options=options)

import os
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import requests
from fpdf import FPDF

pdf = FPDF()      # 初始化一个PDF对象
download_path = r"/hy-tmp/report/"      # 设置下载路径

def write_to_pdf(text, filename):
    pdf = FPDF()
    pdf.add_font('simhei', '', 'SIMHEI.TTF', True)
    pdf.set_font("simhei", size=12)
    pdf.add_page()
    encoded_text = text.encode('utf-8')    # 将文本编码为 UTF-8
    pdf.multi_cell(0, 10, txt=encoded_text.decode('utf-8'))    # 写入 PDF
    pdf.output(filename)

for name in data_dict.keys():    # 循环爬取数据
    start_time = time.time()
    data = data_dict[name]
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    print(f'正在下载{name}年报，共有{data.shape[0]}篇年报')
    total_size_kb = 0   # 初始化内存占用
    wait = WebDriverWait(driver, 10)   # 设置最长等待时间（例如，10秒）

    for i in range(len(data)):
        stock_id = data.loc[i, '股票代码']
        # 如果股票代码在剔除列表中则跳过
        if stock_id not in save_list:
            print(f"股票代码{stock_id}在剔除列表中，跳过")
            continue
        stock_id = re.sub(r'[^\w\s]', '', stock_id)
        stock_name = data.loc[i, 'A股简称']
        report_year = data.loc[i, '年份']
        if report_year != 2022:
            print(f"股票代码{stock_id}的{report_year}在剔除列表中跳过")
            continue
        report_path = f'{download_path}/{stock_id}/{report_year}/'
        if not os.path.exists(report_path):
            os.makedirs(report_path)
        report_url = data.loc[i, '报告链接']
        print(f'正在下载第{i + 1}篇年报：{stock_id}_{stock_name}_{report_year}年年度报告')

        local_filename = f'{report_path}/{report_year}.PDF'   # 检查文件是否已经下载
        if os.path.exists(local_filename):
            print(f"文件已下载，跳过：{stock_id}_{stock_name}_{report_year}年年度报告")
            continue

        driver.get(report_url)   # 访问下载页面

        try:
            # 从url进行下载
            element = wait.until(
                EC.visibility_of_element_located((By.XPATH, '//*[@id="allbulletin"]/thead/tr/th/font/a')))
            file_url = element.get_attribute('href')
            file_extension = os.path.splitext(file_url)[1]

            response = requests.get(file_url, stream=True)
            if response.status_code == 200:
                with open(local_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                file_size_kb = os.path.getsize(local_filename) / 1024
                total_size_kb += file_size_kb
                print(f"文件已成功下载，保存路径：{local_filename}")
                print(f"文件大小：{file_size_kb:.2f} KB")
            else:
                raise Exception(f"网址{file_url}下载失败，HTTP状态码：{response.status_code}")

        except (NoSuchElementException, TimeoutException) as e:
            # 直接读取文本// *[ @ id = "content"] / pre
            try:
                element = wait.until(
                    EC.visibility_of_element_located((By.XPATH, '//*[@id="content"]/pre')))
                text = element.text
                write_to_pdf(text, local_filename)
                file_size_kb = os.path.getsize(local_filename) / 1024
                total_size_kb += file_size_kb
                print(f"文件已成功下载，保存路径：{local_filename}")
                print(f"文件大小：{file_size_kb:.2f} KB")
            except Exception as e:
                print(f"处理错误：无法找到元素或请求超时 {e}")
                with open(f'{report_path}/未找到_{stock_id}_{stock_name}_{report_year}年年度报告.txt', 'w') as f:
                    f.write(f"未能找到下载链接或请求超时，错误信息：{str(e)}")
                continue

        except Exception as e:
            print(f"其他错误：{str(e)}")
            continue

    end_time = time.time()
    print(f'{name}年报下载完成')
    print(f'共下载{len(data)}篇年报')
    print(f'共计下载{total_size_kb:.2f} KB')
    # 写一个txt文件记录该板块的下载数量、占用内存大小和下载总用时
    with open(f'/hy-tmp/{name}年报下载情况.txt', 'w') as f:
        f.write(f'共下载{len(data)}篇年报\n')
        f.write(f'共计下载{total_size_kb:.2f} KB\n')
        f.write(f'共计用时{end_time - start_time:.2f}秒\n')

driver.quit()