import json
from typing import List, Dict
import argparse

def process_single_log(log: List[Dict]) -> List[Dict]:
    conversations = []
    current_system = ""
    current_input = ""
    
    # 跳过初始连续assistant消息
    start_idx = 0
    while start_idx < len(log) and log[start_idx]['role'] == 'assistant':
        start_idx += 1

    for msg in log[start_idx:]:
        if msg['role'] == 'system':
            # 强制重置系统提示
            current_system = msg['content'].strip()
            current_input = ""  # 清空之前可能存在的用户输入
        
        elif msg['role'] == 'user':
            # 合并连续用户消息（带换行符）
            current_input += "\n" + msg['content'].strip() if current_input else msg['content'].strip()
        
        elif msg['role'] == 'assistant':
            # 构建完整对话轮次
            new_round = {
                "input": current_input.strip(),
                "output": msg['content'].strip()
            }
            
            # 首轮携带system信息
            if not conversations:
                new_round = {"system": current_system, **new_round}
            
            conversations.append(new_round)
            current_input = ""  # 重置用户输入

    return conversations

def convert_entry(entry: Dict) -> List[Dict]:
    """
    转换单个JSON条目到目标格式
    """
    return [
        {
            "conversation": process_single_log(entry["revise_log"]) 
        },
        {
            "conversation": process_single_log(entry["high_log"])
        }
    ]

def transform_jsonl(input_path: str, output_path: str):
    """
    完整转换流程
    """
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            entry = json.loads(line)
            converted = convert_entry(entry)
            for conv in converted:
                outfile.write(json.dumps(conv, ensure_ascii=False) + '\n')

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="转换JSONL格式数据")
    parser.add_argument("--input", 
                      type=str, 
                      required=True,
                      help="输入文件路径（例如：webshop_centric.jsonl）")
    parser.add_argument("--output",
                      type=str,
                      required=True,
                      help="输出文件路径（例如：formatted_data.jsonl）")
    return parser.parse_args()

# 修改后的使用示例
if __name__ == "__main__":
    args = parse_arguments()
    transform_jsonl(args.input, args.output)