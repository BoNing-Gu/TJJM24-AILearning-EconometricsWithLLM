import pandas as pd
import sys
sys.modules['sqlite3'] = __import__('pysqlite3')  # 覆盖sqlite3模块，chromadb依赖
import argparse
import os
import json
import pickle
import glob
import time
from datetime import datetime

import openai
import chromadb
from dotenv import load_dotenv   # 加载环境变量
from llama_index.core import VectorStoreIndex, ServiceContext   # 向量索引和上下文
from llama_index.embeddings.langchain.base import LangchainEmbedding     # 词嵌入模型
from langchain.embeddings.huggingface import HuggingFaceEmbeddings      # 词嵌入模型
from llama_index.llms.openai import OpenAI     # 语言模型
from llama_index.vector_stores.chroma import ChromaVectorStore       # 向量存储仓库
# prompt模板
# from llama_index.core import ChatPromptTemplate
from llama_index.core.prompts import LangchainPromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
# 创建查询引擎
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

CHUNK_SIZE = 512            # 文本块大小
CHUNK_OVERLAP = 32      # 文本块重叠大小

# def setEnv:
#     proxy = 'http://127.0.0.1:11000'
#     os.environ['http_proxy'] = proxy
#     os.environ['HTTP_PROXY'] = proxy
#     os.environ['https_proxy'] = proxy
#     os.environ['HTTPS PROXY'] = proxy

def load_target_dfs(config_dict):
    with open(config_dict['targets_df_path'], 'rb') as handle:
        df_targets = pickle.load(handle)
    print(f"数据框形状：{df_targets.shape}")
    print(f"列名：{df_targets.columns.tolist()}")
    # print(df_targets)   # 调试输出
    # df_targets['年份']= str(df_targets['年份'])  # 年报年份
    # df_targets['年份'] = df_targets['年份'].apply(lambda x: x.date().strftime('%Y'))   # 转换成年份格式
    df_targets = df_targets.reset_index(drop=True)
    return df_targets

def initialize_and_return_models(config_dict):
    os.environ["OPENAI_API_KEY"] = config_dict['openai_api_key']   # 设置环境变量
    load_dotenv("openai.env")       # 读取环境变量
    openai.api_key = os.getenv('OPENAI_API_KEY')   # 传给 openai.api_key
    llm = OpenAI(model='gpt-3.5-turbo', temperature=0.5)   # 创建语言模型实例
    embedding_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )   # 创建嵌入模型实例
    # DMetaSoul/sbert-chinese-general-v2
    return llm, embedding_model

def load_index(llm, embedding_model, base_embeddings_path, symbol, ar_year):
    # print("DEBUG: base_embeddings_path:", base_embeddings_path)  # 调试输出
    # print("DEBUG: symbol:", symbol)                              # 调试输出
    # print("DEBUG: ar_year:", ar_year)                            # 调试输出
    path = os.path.join(base_embeddings_path, symbol, str(ar_year))
    # 如果该目录为空，返回一个异常值
    if not os.path.exists(path):
        return None
    # print("DEBUG: Full path:", path)  # 调试输出
    db = chromadb.PersistentClient(path=path)  # 连接到服务器
    chroma_collection = db.get_collection("ar_year")     # 获取'ar_year'集合
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)      # 向量仓库
    service_context = ServiceContext.from_defaults(
                                                    embed_model=embedding_model,
                                                    llm=llm,
                                                    chunk_size=CHUNK_SIZE,
                                                    chunk_overlap=CHUNK_OVERLAP
    )        # 创建一个服务上下文对象
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        service_context=service_context,
    )        # 从向量存储创建一个索引对象
    return index

def get_systemprompt_template(config_dict):
    # 将langchan框架下的prompt转换到llama_index框架
    chat_text_qa_msgs = [
        SystemMessagePromptTemplate.from_template(
            config_dict['llm_system_prompt']
        ),
        HumanMessagePromptTemplate.from_template(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information, "
            "answer the question: {query_str}\n"
        ),
    ]
    chat_text_qa_msgs_lc = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    text_qa_template = LangchainPromptTemplate(chat_text_qa_msgs_lc)
    return text_qa_template   # 提供了一个可与大预言模型交互的模板

# def get_systemprompt_template(config_dict):
#     qa_prompt_str = (
#         "Context information is below.\n"
#         "---------------------\n"
#         "{context_str}\n"
#         "---------------------\n"
#         "Given the context information, "
#         "answer the question: {query_str}\n"
#     )
#     # 直接在llama_index下写prompt
#     # chat_text_qa_msgs = [
#     #     ("system",config_dict['llm_system_prompt'],),
#     #     ("user", qa_prompt_str),
#     # ]
#     # text_qa_template = PromptTemplate(qa_prompt_str)
#     chat_text_qa_msgs = [
#         ChatMessage(content=config_dict['llm_system_prompt'], role=MessageRole.SYSTEM),
#         ChatMessage(content=qa_prompt_str, role=MessageRole.USER),
#     ]
#     text_qa_template = ChatPromptTemplate(messages=chat_text_qa_msgs)
#     return text_qa_template

def load_query_engine(index, llm, text_qa_template):
    query_engine = index.as_query_engine(llm=llm, text_qa_template=text_qa_template, similarity_top_k=10)   # 通过高度模块化的api创建一个查询引擎
    return query_engine
    # retriever = VectorIndexRetriever(
    #     index=index, similarity_top_k=2,
    # )    # configure retriever
    # response_synthesizer = get_response_synthesizer(
    #     response_mode="tree_summarize",
    # )    # configure response synthesizer
    # query_engine = RetrieverQueryEngine(
    #     retriever=retriever, response_synthesizer=response_synthesizer,
    # )    # assemble query engine
    # return  query_engine

def get_gpt_generated_feature_dict(query_engine, questions_dict):
    response_dict = {}
    for feature_name, question in questions_dict.items():   # 特征名称和对应问题
        # 暂停0.2秒，避免超过OpenAI的速率限制，即在短时间内发送太多请求
        time.sleep(0.2)
        response = query_engine.query(question)
        # print("DEBUG: 输出:", response)             # 调试输出
        # print("DEBUG: 查看源节点:", response.source_nodes)            # 调试输出
        # print("DEBUG: 查看特定源节点的文本:", response.source_nodes[0].node.get_text())            # 调试输出
        # print("DEBUG: 查看特定源节点的文本:", response.source_nodes[1].node.get_text())  # 调试输出
        # print("DEBUG: 查看特定源节点的文本:", response.source_nodes[2].node.get_text())  # 调试输出
        response_dict[feature_name] = int(eval(response.response)['score'])
                # 假设GPT生成的响应包含一个表示特征值的 'score' 字段，并将其转换为整数类型
    return response_dict

def are_features_generated(base_path, symbol, ar_year):
            # 检查特征是否已经生成
    df_name = 'df_{}_{}.pickle'.format(symbol, ar_year)
    full_path = os.path.join(base_path, df_name)
    if os.path.exists(full_path):
        return True
    return False

def save_features(df, llm, embedding_model, config_dict, questions_dict,
                           embeddings_directory, features_save_directory):
    for i in df.index:
        start_time = time.time()
        curr_series = df.loc[i]
        symbol = curr_series['股票代码']          # 股票代码
        ar_year = str(curr_series['年份'])              # 报告年份
        print(f"Processing symbol: {symbol}, year: {ar_year}")
        if are_features_generated(features_save_directory, symbol, ar_year):
            continue   # 如果已经创建则跳过
        index = load_index(llm, embedding_model, embeddings_directory, symbol, ar_year)    # 加载索引
        # 如果index为None，跳过循环
        if index is None:
            continue
        text_qa_template = get_systemprompt_template(config_dict)       # 获取系统提示模板
        query_engine = load_query_engine(index, llm, text_qa_template)       # 创建查询引擎，传入索引和系统提示模板
        gpt_feature_dict = get_gpt_generated_feature_dict(query_engine, questions_dict)   # 使用 GPT 生成特征字典
        gpt_feature_df = pd.DataFrame.from_dict(gpt_feature_dict, orient='index').T    # 转换成数据框
        gpt_feature_df.columns = ['feature_{}'.format(c) for c in gpt_feature_df.columns]
        # print("DEBUG: gpt_feature_df columns:", gpt_feature_df.columns)
        gpt_feature_df['meta_股票代码'] = symbol
        gpt_feature_df['meta_年份'] = ar_year
        print("DEBUG: gpt_feature_df:", gpt_feature_df.head())
        with open(os.path.join(features_save_directory, 'df_{}_{}.pickle'.format(symbol, ar_year)), 'wb') as handle:
            pickle.dump(gpt_feature_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Completed: {} in {:.2f}s".format(i, time.time() - start_time))

def save_consolidated_df(config_dict, questions_dict, targets_df,
                         features_save_directory, final_df_save_path):
    df_paths_list = [file for file in glob.glob(os.path.join(features_save_directory, '*')) \
                     if os.path.isfile(file)]
    feature_df_full = pd.DataFrame()
    feature_cols = list(questions_dict.keys())
    feature_cols = ['feature_{}'.format(f) for f in feature_cols]
    # print("DEBUG: final_df Feature columns:", feature_cols)
    meta_cols = ['meta_股票代码', 'meta_年份']
    for df_path in df_paths_list:
        with open(df_path, 'rb') as handle:
            gpt_feature_df = pickle.load(handle)
        # print("DEBUG: gpt_feature_df head:", gpt_feature_df.head())
        gpt_feature_df = gpt_feature_df.loc[:, feature_cols + meta_cols].copy()
        feature_df_full = pd.concat([feature_df_full, gpt_feature_df], ignore_index=True)   # 合并成完整的gpt特征数据框
    feature_df_full['meta_年份'] = feature_df_full['meta_年份'].astype(str)
    targets_df['年份'] = targets_df['年份'].astype(str)
    merged_df = pd.merge(feature_df_full, targets_df, left_on=['meta_股票代码', 'meta_年份'], right_on=['股票代码', '年份'], how='inner')
                        # 合并到target_df，不过在我们的情况下，target_df只包含了'股票代码'和'年份'
    with open(final_df_save_path, 'wb') as handle:
        pickle.dump(merged_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    merged_df.to_excel(final_df_save_path.replace('.pickle', '.xlsx'), index=False)
    print('已完成打分', merged_df.shape)

def main(args):
    try:  # 加载配置文件
        with open(args.config_path) as json_file:
            config_dict = json.load(json_file)
        print("配置文件加载成功。")
        for key, value in config_dict.items():
            print(f"{key}: {value}")  # 打印字典内容
    except Exception as e:
        print(f"无法加载配置文件：{e}")
        return  # 提前退出函数

    try:  # 加载配置文件
        with open(args.questions_path) as json_file:
            questions_dict = json.load(json_file)
        print("用户prompt文件加载成功。")
        for key, value in questions_dict.items():
            print(f"{key}: {value}")  # 打印字典内容
    except Exception as e:
        print(f"无法加载用户prompt文件：{e}")
        return  # 提前退出函数

    df_targets  = load_target_dfs(config_dict)
    llm, embedding_model = initialize_and_return_models(config_dict)
    save_features(df_targets, llm, embedding_model, config_dict, questions_dict,
                            config_dict['embeddings_directory'], config_dict['features_save_directory'])
    save_consolidated_df(config_dict, questions_dict, df_targets,
                            config_dict['features_save_directory'], config_dict['final_df_save_path'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest='config_path', type=str,
                        required=True,
                        help='''Full path of config.json''')
    parser.add_argument('--questions_path', dest='questions_path', type=str,
                        required=True,
                        help='''Full path of questions.json which contains the questions 
                        for asking to the LLM''')
    # python gpt_score.py --config_path /hy-tmp/code/config.json --questions_path /hy-tmp/code/questions.json
    main(args=parser.parse_args())
    sys.exit(0)
