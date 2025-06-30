# 网页语料图像分类项目

这是一个基于大语言模型的图像分类和质量评估项目，使用VLM模型对网页截图进行内容分析和质量评估。

## 项目简介

本项目实现了以下功能：
- 🖼️ **图像描述生成**：使用VLM模型分析图片内容
- 📊 **内容分类**：将内容分类为数学、物理、化学、生命、地球、材料、其他
- ⭐ **质量评估**：评估内容质量（P0/P1/P2）
- ⚡ **并发处理**：支持多任务并发处理
- 🔄 **错误处理**：完善的错误处理和重试机制

## 项目结构

```
code/
├── baseline.py          # 主程序
├── Toolbox.py           # 工具箱类
├── AsyncTasks.py        # 异步任务处理
├── ApiKeyManager.py     # API密钥管理
├── config.py            # 配置文件（需要自己创建）
├── config.example.py    # 配置文件示例
├── requirements.txt     # 依赖包
├── val_gt.jsonl         # 输入数据
├── pic/                 # 图片文件夹
├── .gitignore          # Git忽略文件
└── README.md           # 说明文档
```

## 快速开始

### 1. 克隆项目

```bash
git clone <你的GitHub仓库地址>
cd <项目目录>
```

### 2. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置API密钥

```bash
# 复制配置文件示例
cp config.example.py config.py

# 编辑配置文件，填入你的API密钥
# 编辑 config.py 文件，将 your-api-key-here 替换为你的实际API密钥
```

### 4. 准备数据

确保以下文件存在：
- `val_gt.jsonl`：包含任务数据的JSONL文件
- `pic/` 文件夹：包含所有图片文件

### 5. 运行程序

```bash
python baseline.py
```

## 安装和配置

### 1. 安装依赖

```bash
# 激活虚拟环境
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥

编辑 `config.py` 文件，将 `your-api-key-here` 替换为你的实际API密钥：

```python
API_CONFIG = {
    "api_key": "sk-your-actual-api-key",  # 替换为你的API密钥
    "base_url": "https://api.siliconflow.cn/v1/",
    "llm_model": "Qwen/Qwen2.5-7B-Instruct",
    "vlm_model": "Qwen/Qwen2.5-VL-32B-Instruct"
}
```

### 3. 准备数据

确保以下文件存在：
- `val_gt.jsonl`：包含任务数据的JSONL文件
- `pic/` 文件夹：包含所有图片文件

## 运行程序

```bash
python baseline.py
```

## 功能说明

### 主要功能

1. **图像描述生成**：使用VLM模型分析图片内容
2. **内容分类**：将内容分类为数学、物理、化学、生命、地球、材料、其他
3. **质量评估**：评估内容质量（P0/P1/P2）
4. **并发处理**：支持多任务并发处理
5. **错误处理**：完善的错误处理和重试机制

### 配置选项

在 `config.py` 中可以调整以下参数：

- **API配置**：模型名称、API密钥、基础URL
- **任务配置**：并发数、重试次数、超时时间
- **文件配置**：输入输出文件路径

### 输出结果

程序会生成 `results.json` 文件，包含每个任务的处理结果：

```json
{
  "url_id": "任务ID",
  "subject": "分类结果",
  "quan": "质量评估",
  "subject_correct": true/false,
  "quan_correct": true/false,
  "both_correct": true/false
}
```

### 准确率统计

程序会输出三个准确率指标：

1. **类型分类准确率**：预测的内容类型与真实类型的一致性
2. **质量评估准确率**：预测的质量等级与真实质量等级的一致性  
3. **两项都正确率**：类型和质量都预测正确的比例

示例输出：
```
=== 准确率统计 ===
类型分类准确率：75.00% (15/20)
质量评估准确率：60.00% (12/20)
两项都正确率：45.00% (9/20)

=== 详细分析 ===
仅类型分类正确：6 个
仅质量评估正确：3 个
两项都正确：9 个
两项都错误：2 个
```

## 常见问题

### 1. API密钥无效

**错误信息**：`Error code: 401 - Api key is invalid`

**解决方法**：
- 检查 `config.py` 中的API密钥是否正确
- 确认API密钥是否有效且未过期
- 检查API服务是否正常

### 2. 文件找不到

**错误信息**：`FileNotFoundError`

**解决方法**：
- 确认 `val_gt.jsonl` 文件存在
- 确认图片文件路径正确
- 检查文件编码是否为UTF-8

## 性能优化

1. **调整并发数**：根据API限制调整 `max_workers`
2. **设置重试次数**：适当增加 `max_retries` 提高成功率
3. **监控API使用**：注意API调用频率限制

## 技术支持

如果遇到问题，请检查：
1. 虚拟环境是否正确激活
2. 依赖包是否完整安装
3. API密钥是否有效
4. 网络连接是否正常

## 贡献

欢迎提交Issue和Pull Request！

