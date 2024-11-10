'''
使用了通义千问api（因为可以处理图片类型的输入）
若要使用，需要自行到官网获取api key并配置环境变量
'''
from openai import OpenAI
import os
import base64

class Qwen():
    def __init__(self, image_path):
        self.image_path = image_path
        self.base64_image = self.encode_image(self.image_path)
        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.completion = self.client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{self.base64_image}"},
                        },
                        {"type": "text", "text": "图片中的数字是0-9的哪些(若有小数点，将小数点也作为输出结果)"},
                    ],
                }
            ],
        )
            
    #  base 64 编码格式
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    # def encode_image(self, img):
    #     return base64.b64encode(img).decode("utf-8")

    def outputResult(self):
        print(self.completion.choices[0].message.content)
        return self.completion.choices[0].message.content