# API配置示例
# 请复制此文件为 config.py 并填入你的实际API密钥

API_CONFIG = {
    "api_key": "your-api-key-here",  # 请替换为你的实际API密钥
    "base_url": "https://api.siliconflow.cn/v1/",
    "llm_model": "Qwen/Qwen2.5-7B-Instruct",
    "vlm_model": "Qwen/Qwen2.5-VL-32B-Instruct"
}

# 任务配置
TASK_CONFIG = {
    "max_workers": 5,  # 最大并发数
    "max_retries": 2,  # 最大重试次数
    "timeout": 30      # 超时时间（秒）
}

# 文件路径配置
FILE_CONFIG = {
    "input_file": "val_gt.jsonl",
    "output_file": "results.json"
} 