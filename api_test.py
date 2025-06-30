from openai import OpenAI
from PIL import Image
import io
import base64

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
input_image_path="1747998537949.jpg"
base64_image=convert_image_to_webp_base64(input_image_path)
client = OpenAI(api_key="sk-dbycajuugroovfkibuopglsfbgqpdepsuddvcwjglhenojsq", 
                base_url="https://api.siliconflow.cn/v1/")


# 测试 llm 模型
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Hello, world!"}],
    max_tokens=4096,
    extra_body={
        "thinking_budget": 128,
    },
    stream=True
)
content = ""
reasoning_content = ""
for chunk in response:
    if chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)


# 测试 VL 模型
# response = client.chat.completions.create(
#     model="Pro/Qwen/Qwen2.5-VL-7B-Instruct",
#     messages=[
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": "https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/dog.png"
#                 }
#             },
#             {
#                 "type": "text",
#                 "text": "Describe the image."
#             }
#         ]
#     }],
# )
# print(response.choices[0].message.content)


# 测试图片转base64
response = client.chat.completions.create(
    model="Pro/Qwen/Qwen2.5-VL-7B-Instruct",
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
            "text": "text-prompt here"
        }
    ]
}
])
print(response.choices[0].message.content)