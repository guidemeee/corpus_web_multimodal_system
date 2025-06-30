"""
本项目旨在实现一个基准代码，实现该项目
"""
import json
from Toolbox import Toolbox, MetaData
from AsyncTasks import AsyncTasks
import asyncio
from config import API_CONFIG, TASK_CONFIG, FILE_CONFIG

def process_task(task: MetaData, toolbox: Toolbox):
    """处理单条数据"""
    try:
        # 处理图片描述
        imgdesc = response = toolbox.imgdesc(task.ori_pic)
        response = imgdesc
        print(f"图片描述: {response}")
        
        # 分类markdown - 处理返回值变化
        classify_result = toolbox.classify_markdown(response, task.title)
        if isinstance(classify_result, tuple):
            type = classify_result[0]  # 取第一个元素作为分类结果
        else:
            type = classify_result
        
        # 质量评估
        evaluate = toolbox.evaluate_markdown(response)
        
        print(f"分类结果: {type}")
        print(f"质量评估: {evaluate}")
        print(f"实际分类: {task.subject}")
        print(f"实际质量: {task.quan}")
        
        # 返回处理结果
        return {
            "url_id": task.url_id,
            "subject": type,
            "quan": evaluate,
            "subject_correct": type == task.subject,
            "quan_correct": evaluate == task.quan,
            "both_correct": (type == task.subject) and (evaluate == task.quan)
        }
    except Exception as e:
        print(f"处理任务时出错: {e}")
        return {
            "url_id": task.url_id,
            "subject": "其他",
            "quan": "P2(质量差，信息量少)",
            "subject_correct": False,
            "quan_correct": False,
            "both_correct": False,
            "error": str(e)
        }

def main():
    # 检查API密钥配置
    if API_CONFIG["api_key"] == "your-api-key-here":
        print("错误：请在 config.py 中配置有效的API密钥")
        return
    
    # 创建引擎
    from Toolbox import LLMEngine, VLMEngine
    llm_engine = LLMEngine(
        model_name=API_CONFIG["llm_model"], 
        api_key=API_CONFIG["api_key"], 
        base_url=API_CONFIG["base_url"]
    )
    vlm_engine = VLMEngine(
        model_name=API_CONFIG["vlm_model"], 
        api_key=API_CONFIG["api_key"], 
        base_url=API_CONFIG["base_url"]
    )
    
    # 创建工具箱
    toolbox = Toolbox(llm_engine, vlm_engine)
    
    # 读取数据 - 适配新的dealwith_jsonl方法
    try:
        metadatas = toolbox.dealwith_jsonl(FILE_CONFIG["input_file"])
        print(f"读取到 {len(metadatas)} 条数据")
    except FileNotFoundError:
        print(f"错误：找不到文件 {FILE_CONFIG['input_file']}")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 创建异步任务处理器
    tasks = AsyncTasks()
    
    # 准备任务参数
    submit_args = [(metadata, toolbox) for metadata in metadatas]
    
    # 执行任务
    results = tasks.submit(
        process_task, 
        submit_args, 
        name="process_task", 
        max_workers=TASK_CONFIG["max_workers"],
        max_retries=TASK_CONFIG["max_retries"]
    )
    
    # 统计结果
    count_subject = 0  # 类型分类正确数
    count_quan = 0     # 质量评估正确数
    count_both = 0     # 两项都正确数
    total = 0          # 成功处理总数
    errors = 0         # 处理失败数
    
    for result in results:
        if result is not None:
            if "error" in result:
                errors += 1
            else:
                total += 1
                if result["subject_correct"]:
                    count_subject += 1
                if result["quan_correct"]:
                    count_quan += 1
                if result["both_correct"]:
                    count_both += 1
    
    # 输出统计结果
    print(f"\n=== 处理结果统计 ===")
    print(f"总任务数: {len(metadatas)}")
    print(f"成功处理: {total}")
    print(f"处理失败: {errors}")
    
    # 避免除零错误并输出准确率
    if total > 0:
        subject_accuracy = count_subject / total
        quan_accuracy = count_quan / total
        both_accuracy = count_both / total
        
        print(f"\n=== 准确率统计 ===")
        print(f"类型分类准确率：{subject_accuracy:.2%} ({count_subject}/{total})")
        print(f"质量评估准确率：{quan_accuracy:.2%} ({count_quan}/{total})")
        print(f"两项都正确率：{both_accuracy:.2%} ({count_both}/{total})")
        
        # 详细分析
        print(f"\n=== 详细分析 ===")
        print(f"仅类型分类正确：{count_subject - count_both} 个")
        print(f"仅质量评估正确：{count_quan - count_both} 个")
        print(f"两项都正确：{count_both} 个")
        print(f"两项都错误：{total - count_subject - count_quan + count_both} 个")
    else:
        print("没有成功处理任何任务，请检查API密钥配置")
    
    # 保存结果
    try:
        with open(FILE_CONFIG["output_file"], "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到 {FILE_CONFIG['output_file']}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

if __name__ == "__main__":
    main()

