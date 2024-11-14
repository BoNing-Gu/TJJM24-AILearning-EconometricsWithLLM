import PyPDF2
import os
from pdf2docx import parse
from tqdm import tqdm


def get_files_list(path):
    """
    获取传入路径中及其子目录下的所有pdf文件路径
    :param path: 要搜索的根路径
    :return: pdf文件路径列表
    """
    files_list = []
    for root, dirs, files in os.walk(path):  # 遍历目录
        for file in files:  # 遍历文件
            file_path = os.path.join(root, file)  # 拼接路径
            if file_path.endswith(".PDF"):  # 如果是pdf文件
                files_list.append(file_path)  # 添加到列表中
    return files_list


root_path = r'\hy-tmp\data\上交所主板年报'
files = get_files_list(root_path)


# print(files)

def find_contents_page(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)

        for page_number in range(total_pages):
            page_text = pdf_reader.pages[page_number].extract_text()
            if "目录" in page_text:  # 检查是否为目录页
                pdf_name = pdf_path.split('\\')[-1]
                content_page_number = int(page_number) + 1
                print(f'找到{pdf_name}的工作目录页：第{content_page_number}页')
                return page_text, content_page_number


def extract_contents(text):
    contents = {}
    lines = text.splitlines()
    content_page_number_from_page = None
    for line in lines:
        # 移除行中多余的空格和点
        line = line.replace('.', '').strip()
        if '/' in line or line.isdigit() or "目录" in line:
            continue
        # 拆分行并尝试提取页码
        parts = line.rsplit(maxsplit=1)  # 从右侧拆分，最多拆分一次
        if len(parts) == 2 and parts[1].isdigit():
            chapter_title = parts[0].strip()
            # 移除章节编号 "第x节"
            chapter_title = ' '.join(chapter_title.split()[1:])  # 假设标题格式为"第一节 标题"
            chapter_title = chapter_title.replace(' ', '')  # 删除chapter_title字符串中的空格
            page_number = parts[1]
            contents[chapter_title] = page_number
    return contents


def find_chapter_page(pdf_path, chapter_title, work_page_number, contents):
    next_chapter_title = list(contents.keys())[list(contents.keys()).index(chapter_title) + 1]
    start_page = None
    end_page = None
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)
        for page_number in range(total_pages):
            if page_number > work_page_number:
                page_text = pdf_reader.pages[page_number].extract_text()
                if (start_page is None) & (chapter_title in page_text) & ('第三节' in page_text):
                    start_page = page_number + 1
                    print(f'找到{chapter_title}的起始页：第{start_page}页')
                if (end_page is None) & (next_chapter_title in page_text) & ('第四节' in page_text):
                    end_page = page_number + 1
                    print(f'找到{chapter_title}的结束页：第{end_page}页')
                    break
    return start_page, end_page


for file in tqdm(files, desc="Processing Files"):
    contents_text, content_page_number = find_contents_page(file)
    # print(contents_text)
    contents_dict = extract_contents(contents_text)
    # print(contents_dict)
    # chapter_title = '管理层讨论与分析'
    # next_chapter_title = list(contents_dict.keys())[list(contents_dict.keys()).index(chapter_title) + 1]
    # print(next_chapter_title)
    start_page, end_page = find_chapter_page(file, '管理层讨论与分析', content_page_number, contents_dict)
    # 截取pdf的部分页面
    with open(file, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pdf_writer = PyPDF2.PdfWriter()
        for page_number in range(start_page - 1, end_page):
            pdf_writer.add_page(pdf_reader.pages[page_number])

        # 构造输出文件路径
        output_directory = '\\'.join(file.split('\\')[:-2]) + '\\上交所主板年报管理层讨论与分析'
        file_name_part = file.split("\\")[-1]
        output_file_name = f'管理层讨论与分析_{file_name_part}'
        output_path = f'{output_directory}\\{output_file_name}'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # 写入文件
        with open(output_path, 'wb') as output_pdf:
            pdf_writer.write(output_pdf)

    pdf_file = output_path
    docx_file = output_path.split('.')[:-1] + '.docx'
    # convert pdf to docx
    parse(pdf_file, docx_file)
    os.remove(pdf_file)
    print('完成转换')