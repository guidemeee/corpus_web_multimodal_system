"""
本工具旨在基于LLM实现和集成一些图像和文本处理工具，方便后续的开发和调用。
"""
import os
import time
import re
from openai import OpenAI
import fitz
from gptpdf import parse_pdf
import html2text
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from openai import OpenAI
from PIL import Image
import io
import base64
from AsyncTasks import AsyncTasks

def convert_image_to_webp_base64(input_image_path):
    try:
        with Image.open(input_image_path) as img:
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='webp')
            byte_arr = byte_arr.getvalue()
            base64_str = base64.b64encode(byte_arr).decode('utf-8')
            return base64_str
    except IOError:
        print(f"Error: Unable to open or convert the image {input_image_path}")
        return None
    

class MetaData:
    def __init__(self, data, data_path = None):
        # 读取json里的每个key，value，并保存到self里
        self.data_path = data_path
        self.data = data
        for key, value in data.items():
            setattr(self, key, value)
    def __str__(self):
        return f"MetaData(url_id={self.url_id}, url={self.url}, title={self.title}, domain={self.domain})"
        
class LLMEngine:
    def __init__(self, model_name, api_key, base_url):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

class VLMEngine:
    def __init__(self, model_name, api_key, base_url):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)


class Toolbox:
    def __init__(self, llm_engine, vlm_engine):
        self.llm_engine = llm_engine
        self.vlm_engine = vlm_engine
    
    def img2pdf(self, image_path):
        img_name = image_path.split('.')[0]
        doc = fitz.open()
        # 使用fitz打开图片
        imgdoc = fitz.open(image_path)
        # 转为pdf
        pdfbytes = imgdoc.convert_to_pdf()
        imgpdf = fitz.open("pdf", pdfbytes)
        doc.insert_pdf(imgpdf)
        doc.save(img_name + '.pdf')
        doc.close()
        print(f"图片{image_path}已转换为pdf，保存路径为：{img_name}.pdf")
    
    def pdf2markdown(self, pdf_path):
        try:
            content, image_paths = parse_pdf(pdf_path, api_key=self.vlm_engine.api_key, base_url=self.vlm_engine.base_url, model=self.vlm_engine.model_name)
        except Exception as e:
            print(e)
            return ""
        # 删除图片，保存markdown
        for image_path in image_paths:
            os.remove(image_path)
        print(content)
        return content

    def img2markdown(self, image_path):
        # 先转pdf
        self.img2pdf(image_path)
        # 再转markdown
        # print(image_path.replace('.png', '.pdf'))
        for _ in range(10):
            markdown = self.pdf2markdown(image_path.replace('.png', '.pdf'))
            print(markdown)
            # print(markdown)
            if len(markdown) > 50:
                break
        # 删除pdf
        os.remove(image_path.replace('.png', '.pdf'))
        # 返回markdown路径
        return markdown
    
    def html2markdown(self, html_content):
        # with open(html_path, 'r', encoding='utf-8') as f:
        #     html_content = f.read()
        markdown_content = html2text.html2text(html_content)
        # print("-"*100)
        # print(markdown_content)
        # print("-"*100)
        return markdown_content
    
    def dealwith_jsonl(self, jsonl_path):
        metadatas = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                metadata = MetaData(data, jsonl_path)
                metadatas.append(metadata)
        return metadatas
    
    def get_response(self, engine_type, request):
        if engine_type == "vlm":
            try:
                response = self.vlm_engine.client.chat.completions.create(
                    model=self.vlm_engine.model_name,
                    messages=request,
                    # max_tokens=4096,
                    temperature=0.0
                )
            except Exception as e:
                print(e)
                time.sleep(1)
                return self.get_response(engine_type, request)
    
            return response
        else:
            response = self.llm_engine.client.chat.completions.create(
                model=self.llm_engine.model_name,
                messages=request,
                temperature=0.0
            )
            return response
    
    def merge_mardkown(self, markdown_a, markdown_b):
        prompt = f"""
{markdown_a}
---------------------------------
{markdown_b}
---------------------------------
逐步推理，给出内容分析和评估。直接返回markdown内容，禁止任何其他内容。
"""
        response = self.get_response("vlm", [{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    

    def extract_boxed_answer(self, solution):
        # 使用非贪婪匹配和平衡括号的方式提取boxed内容
        result = []
        
        # 匹配带转义符号的 \boxed{} 和不带转义符号的 boxed{}
        pattern1 = r'boxed\{((?:[^{}]|(?:\{[^{}]*\}))*)\}'
        pattern2 = r'fbox\{((?:[^{}]|(?:\{[^{}]*\}))*)\}'
        
        matches1 = re.findall(pattern1, solution)
        matches2 = re.findall(pattern2, solution)
        matches = matches1 + matches2
        
        # 如果没有匹配到，尝试使用更复杂的递归方式处理嵌套括号
        if not matches and ('\\boxed{' in solution or 'boxed{' in solution):
            # 查找所有 \boxed{ 和 boxed{ 的位置
            start_indices = []
            start_indices.extend([m.start() for m in re.finditer(r'\\boxed\{', solution)])
            start_indices.extend([m.start() for m in re.finditer(r'boxed\{', solution)])
            
            for start_idx in start_indices:
                # 确定起始位置
                if solution[start_idx:start_idx+2] == '\\b':  # 是 \boxed{
                    content_start = start_idx + 7  # len('\\boxed{') = 7
                else:  # 是 boxed{
                    content_start = start_idx + 6  # len('boxed{') = 6
                
                # 从起始位置开始
                idx = content_start
                bracket_count = 1
                
                # 遍历寻找匹配的右括号
                while idx < len(solution) and bracket_count > 0:
                    if solution[idx] == '{':
                        bracket_count += 1
                    elif solution[idx] == '}':
                        bracket_count -= 1
                    idx += 1
                
                if bracket_count == 0:  # 找到了匹配的右括号
                    result.append(solution[content_start:idx-1])
        else:
            result = matches
            
        return result[-1] if result else ""

    def classify_markdown(self, markdown, title):
        if len(markdown) > 10000:
            markdown = markdown[:10000]
        prompt = f"""
下面有一个网页的markdown形式：
--------------------------------
{markdown}
--------------------------------
现有以下主题：
数学：数学公式，数学推理，证明，猜想发现，应用，建模。（验证码不算数学主题）
物理：物质的物理性质（例如铁的磁性），物理公式，发现，实验，推导以及物理g实际应用如科技产品的细节，物理工程，科技，航天，军事。
化学：物质的化学性质（例如氧气的化学性质），化学公式，化学反应，合成，实验，发现。
生命：生命科学，生物研究调研（动物，植物，微生物），医学（疾病，药物，治疗）实验，发现，测试，调研。
地球：天文，大气科学，地理，地质学，海洋学，地球物理学，地震，地质勘探。
材料：材料的发现和研究。（不包含物理、化学、生物材料）
其他：不包含上述任何主题信息的登陆，请求界面，验证码等。
根据以上类型，对网页的主题从[数学, 物理, 化学, 生命, 地球, 材料, 其他]中选择一个类别，并返回。
Please think step by step, and put the final answer within \\boxed{{}}.
"""
        async_tasks = AsyncTasks()
        responses = async_tasks.submit(self.get_response, [["llm", [{"role": "user", "content": prompt}]] for _ in range(3)])

        types = ["数学", "物理", "化学", "生命", "地球", "材料", "其他"]
        vote = dict()
        for response in responses:
            for type in types:
                if type in self.extract_boxed_answer(response.choices[0].message.content):
                    vote[type] = vote.get(type, 0) + 1
        
        if vote:
            max_type = max(vote, key=vote.get)
            return max_type, responses
        else:
            return "其他", responses

    def evaluate_markdown(self, markdown):
        prompt = f"""
下面有一篇论文的描述：
--------------------------------
{markdown}
--------------------------------
请对论文从信息密度、⼴告⼲扰程度、内容原创性等考察，最终表现为⼀个综合等级。质量等级类型如下：
- P0(质量很好)
- P1(质量⼀般)
- P2(质量差，并且信息量少)
从P0,P1,P2中选择一个等级。
Please think step by step, and put the final answer within \\boxed{{}}.
"""
        response = self.get_response("vlm", [{"role": "user", "content": prompt}])
        types = ["P0", "P1", "P2"]
        all_types = ["P0(质量很好)", "P1(质量一般)", "P2(质量差，信息量少)"]
        for type in types:
            if type in response.choices[0].message.content:
                return all_types[types.index(type)]
        return "P2(质量差，信息量少)"
    
    def imgdesc(self, image_path):
        base64_image = convert_image_to_webp_base64(image_path)
        messages=[
            {
                "role": "user",
                "content":[
                    {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail":"low"
                }
            },
            {
                "type": "text",
                "text": "逐步思考，挖掘图片中的所有重要且有意义的信息。"
            }
        ]
        }
        ]
        response = self.get_response("vlm", messages)
        return response.choices[0].message.content
    
    def imgeval(self, image_path):
        base64_image = convert_image_to_webp_base64(image_path)
        messages=[
            {
                "role": "user",
                "content":[
                    {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail":"low"
                }
            },
            {
                "type": "text",
                "text": """
请对该网页图片从信息密度、⼴告⼲扰程度、内容原创性等考察(大片空白，且无实际意义信息的必定被打低分，数理题目需要分析其内容价值和和意义)，最终表现为⼀个综合分数(0-1之间的三位小数)。
Please think step by step, and put the final answer within \\boxed{{}}.
"""
            }
        ]
        }
        ]
        async_tasks = AsyncTasks()
        responses = async_tasks.submit(self.get_response, [["vlm", messages] for _ in range(5)])
        vote = dict()
        types = ["P0(质量很好)", "P1(质量一般)", "P2(质量差，信息量少)"]
        sums = 0
        scores = []
        for response in responses:
            boxed_answer = self.extract_boxed_answer(response.choices[0].message.content)
            scores.append(float(boxed_answer))
        
        scores.sort()
        # scores = scores[1:-1]  # 去掉最高分和最低分
        sums = sum(scores)
        if sums/len(scores) >= 0.65:
            return "P0(质量很好)",sums/len(scores),responses
        elif sums/len(scores) >= 0.5:
            return "P1(质量一般)",sums/len(scores),responses
        else:
            return "P2(质量差，信息量少)",sums/len(scores),responses


if __name__ == "__main__":
    llm_engine = LLMEngine(model_name="Qwen/Qwen2.5-VL-32B-Instruct", api_key="sk-uinjzgrmudclvnbszzyfqguoaatawrvuxahaxiakzelxfrzu", base_url="https://api.siliconflow.cn/v1/")
    vlm_engine = VLMEngine(model_name="Qwen/Qwen2.5-VL-32B-Instruct", api_key="sk-uinjzgrmudclvnbszzyfqguoaatawrvuxahaxiakzelxfrzu", base_url="https://api.siliconflow.cn/v1/")
    toolbox = Toolbox(llm_engine, vlm_engine)
    # toolbox.img2pdf("1747998537949.jpg")
    # toolbox.pdf2markdown("1747998537949.pdf")
    
    # 处理jsonl文件中的HTML
    # metadatas = toolbox.dealwith_jsonl("data/val_gt.jsonl")
    # print(metadatas)
    toolbox.html2markdown("test.html")