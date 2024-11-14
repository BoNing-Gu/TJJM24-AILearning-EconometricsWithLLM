import os
import sys
sys.modules['sqlite3'] = __import__('pysqlite3')  # 覆盖sqlite3模块，chromadb依赖
import json
import pickle
import time
import argparse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, ServiceContext
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import chromadb

CHUNK_SIZE = 512                # 文本块大小
CHUNK_OVERLAP = 32          # 文本块重叠大小

def save_index(embeddings_path, embedding_model, symbol, ar_year, config_dict):
    db = chromadb.PersistentClient(path=os.path.join(embeddings_path, symbol, ar_year))
            # 创建持久化客户端，用于管理嵌入向量的存储和检索，
            # 客户端的地址是/hy-tmp/embedding_example/股票代码/年份，这样每个年报的嵌入向量都会被保存在以股票代码和年报年份命名的目录中
    chroma_collection = db.create_collection("ar_year")  # 创建'ar_year'集合，用于存储年报的嵌入向量
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            # 创建了一个 Chroma 向量存储对象，它用于管理嵌入向量的存储和检索
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # 创建了一个存储上下文对象，其中包含了用于存储嵌入向量的向量存储对象
    service_context = ServiceContext.from_defaults(embed_model=embedding_model,
                                                   chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=CHUNK_OVERLAP)
            # 创建了一个服务上下文对象，其中包含了用于生成嵌入向量的嵌入模型、以及用于处理文本的分块大小和重叠大小
    ar_filing_path = os.path.join(config_dict['annual_reports_pdf_save_directory'], symbol, ar_year)
            # 年报文件路径，这里是年报的 PDF 文件路径    /hy-tmp/report_example/股票代码/年份
    documents = SimpleDirectoryReader(ar_filing_path).load_data()
            # 加载指定路径中的数据，这里是年报的 PDF 文件
    # debug_file_path = os.path.join(ar_filing_path, f"{ar_year}.txt")            # 调试文件
    # with open(debug_file_path, 'w') as debug_file:
    #     for i, document in enumerate(documents):
    #         debug_file.write(f"DEBUG: Index: {i}\n")
    #         debug_file.write(f"DEBUG: Text: {document.text[:]}...\n")
    #         debug_file.write(f"DEBUG: Metadata: {document.metadata}\n")
    #         debug_file.write(f"DEBUG: Relationships: {document.relationships}\n")
    #         debug_file.write("\n")
    _ = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)
            # 使用加载的数据生成嵌入向量，并将其存储到指定的嵌入向量存储对象中。from_documents 方法将文档转换为嵌入向量，并将其存储在提供的存储上下文中。

def save_embeddings(df, embedding_model, save_directory, config_dict, runtime_log_path):
    for i in df.index:
        start_time = time.time()
        curr_series = df.loc[i]
        symbol = curr_series['股票代码']    # 股票代码
        ar_year = str(curr_series['年份'])  # 年报年份
        # ar_year = curr_series['年份'].date().strftime('%Y')     # 年报年份
        save_path = os.path.join(save_directory, symbol, ar_year)    # /hy-tmp/embedding_example/股票代码/年份
        if os.path.exists(save_path):
            continue
        save_index(save_directory, embedding_model, symbol, ar_year, config_dict)
                # 对每个年报生成嵌入向量，并将其存储到指定的目录中
        end_time = time.time()
        duration = end_time - start_time
        print("DEBUG: Completed: {}, {}, {} in {:.2f}s".format(i + 1, symbol, ar_year, duration))
        with open(runtime_log_path, 'a') as log_file:
            log_file.write(f"Completed: {i + 1}, {symbol}, {ar_year} in {duration:.2f}s\n")

def main(args):
    start_time = time.time()  # 记录程序开始运行时间
    try:  # 加载配置文件
        with open(args.config_path) as json_file:
            config_dict = json.load(json_file)
        print("配置文件加载成功。")
        runtime_log_path = config_dict['targets_df_path'].replace("targets", "embedding_log").replace(".pickle", ".txt")
        # for key, value in config_dict.items():
        #     print(f"DEBUG: {key}: {value}")  # 调试输出打印字典内容
        with open(runtime_log_path, 'a') as log_file:
            log_file.write(f"配置文件加载成功: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            for key, value in config_dict.items():
                log_file.write(f"{key}: {value}\n")
    except Exception as e:
        print(f"无法加载配置文件：{e}")
        return  # 提前退出函数

    try:   # 读取目标数据框
        with open(config_dict['targets_df_path'], 'rb') as handle:
            df_targets = pickle.load(handle)
            df_targets = df_targets.reset_index(drop=True)
        with open(runtime_log_path, 'a') as log_file:
            log_file.write(f"目标数据框加载成功: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"数据框形状：{df_targets.shape}\n")
            log_file.write(f"列名：{df_targets.columns.tolist()}\n")
        print("目标数据框加载成功。")
        print(f"DEBUG: 数据框形状：{df_targets.shape}")
        print(f"DEBUG: 列名：{df_targets.columns.tolist()}")
    except Exception as e:
        print(f"无法加载目标数据框：{e}")
        return  # 提前退出函数

    # local_dir = './models'
    # repo_id = "sentence-transformers/all-mpnet-base-v2"
    # model_path = os.path.join(local_dir, repo_id.split("/")[-1])  # 离线方案
    # embedding_model = LangchainEmbedding(
    #     HuggingFaceEmbeddings(model_path=model_path)
    # )       # 创建嵌入模型实例

    embedding_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )    # 需要能够连接到huggingface
    # 可选择的还包括DMetaSoul/sbert-chinese-general-v2

    save_embeddings(df_targets, embedding_model, config_dict['embeddings_directory'], config_dict, runtime_log_path)

    end_time = time.time()  # 记录程序结束时间
    duration = end_time - start_time  # 计算运行时长
    with open(runtime_log_path, 'a') as log_file:
        log_file.write(f"本次运行开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        log_file.write(f"本次运行结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        log_file.write(f"本次运行总时长: {duration} 秒\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest='config_path', type=str,
                        required=True,
                        help='''Full path of config.json''')
    # python embedding.py --config_path /hy-tmp/code/config.json
    main(args=parser.parse_args())
    sys.exit(0)
