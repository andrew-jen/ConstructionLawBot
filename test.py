import openai
import textwrap
from dotenv import load_dotenv
import os

# 載入環境變數
load_dotenv()

# 使用環境變數存取API金鑰
client = openai.OpenAI(
    api_key=os.getenv("sk-A2EONyn0217UHDlwzXnRQg"),
    base_url="https://chatapi.akash.network/api/v1"
)

try:
    response = client.chat.completions.create(
        model="DeepSeek-R1",
        messages=[
            {
                "role": "user",
                "content": "你是誰（請用繁體中文回答）?"
            }
        ],
    )

    print(textwrap.fill(
        response.choices[0].message.content,
        50
    ))
except Exception as e:
    print(f"發生錯誤: {str(e)}")