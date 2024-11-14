import pandas as pd
import pickle
import json
import sys
import argparse

# 修改：直接读取最终列表，数据处理在“列表数据处理.ipynb”中完成

def main(args):
    with open(args.config_path) as json_file:
        config_dict = json.load(json_file)

    targets_df = pd.read_excel('/hy-tmp/data/final_公司年报列表_dropST金融科研_去不平衡_28483条_1994家企业.xlsx', dtype={'股票代码': str})

    if config_dict['example'] == 'True':
        print('示例')
        targets_df = targets_df[targets_df['股票代码'].isin(['600004', '000002'])]
    if config_dict['example'] == '2022':
        print('2022')
        targets_df = targets_df[targets_df['年份'] == 2022]

    with open(config_dict['targets_df_path'], 'wb') as handle:
        pickle.dump(targets_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    targets_df.to_excel(config_dict['targets_df_path'].replace('.pickle', '.xlsx'), index=False)
    print('已保存', targets_df.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest='config_path', type=str,
                        required=True,
                        help='''Full path of config.json''')
    # python targets_df.py --config_path /hy-tmp/code/config.json
    main(args=parser.parse_args())
    sys.exit(0)
