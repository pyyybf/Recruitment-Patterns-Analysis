import pandas as pd
import os

def merge_csv(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    # 初始化一个空的 DataFrame 列表
    df_list = []
    
    # 遍历所有文件
    for file in files:
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, file)
        
        # 检查文件是否是一个 CSV 文件
        if file.endswith('.csv'):
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            
            # 检查并删除空行
            df = df.dropna(how='all')
            # df = df.dropna()
            
            # 将清理过的 DataFrame 添加到列表中
            df_list.append(df)
    
    # 检查是否有 CSV 文件被读取
    if not df_list:
        return "No CSV files found in the specified folder."
    
    # 合并所有的 DataFrame
    merged_df = pd.concat(df_list, ignore_index=True)
    
    return merged_df
