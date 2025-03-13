import os
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import textwrap
from dotenv import load_dotenv
import openai
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

# 載入環境變數
load_dotenv()

# 使用 Akash API 設定
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("未找到 OPENAI_API_KEY，請檢查 .env 文件或環境變數")

client = openai.OpenAI(
    api_key=api_key,
    base_url="https://chatapi.akash.network/api/v1"
)

# 讀取圖像描述
with open('yolo-workspace-main/results.json', "r", encoding="utf-8") as f:
    file_info = json.load(f)

image_descriptions = [f" {image['file_name']},  {image['event_type']}, {image['date']}" for image in file_info]

# 讀取法規資料
with open("law_data.json", "r", encoding="utf-8") as json_file:
    data_list = json.load(json_file)

# 初始化知識庫
knowledge_base = [item["val"] for item in data_list]

# 初始化嵌入模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
knowledge_embeddings = embedding_model.encode(knowledge_base)

# 輸入問題
query = input(f'請輸入法規相關問題： ')
query_embedding = embedding_model.encode([query])

# 計算相似度並選擇相關資料
similarities = cosine_similarity(query_embedding, knowledge_embeddings)[0]
relevant_indices = np.argsort(similarities)[-5:]  # 取最相关的5条
relevant_knowledge = [knowledge_base[i] for i in relevant_indices]
problem = f'請根據{relevant_knowledge}回答{query}的問題'

# 使用檢索內容進行生成
examples = [
  {
    "question": f'請根據{image_descriptions}裡面的image和{relevant_knowledge}裡面的法規資料回答 你認為 image1.jpg 中的違規觸犯了哪些法規？ 的問題',
    "answer": "分析結果: 第 11-1 條 雇主對於進入營繕工程工作場所作業人員，應提供適當安全帽，並使其正確戴用。"
  },
  {
    "question": f'請根據{image_descriptions}裡面的image和{relevant_knowledge}裡面的法規資料回答 image2 有哪些違規？ 的問題',
    "answer": "分析結果: 第 173 條 第三項 雇主對於工作場所之急救設施，除依一般工作場所之急救設施規定外，應防止昆蟲、老鼠等孳生並予以撲滅。"
  },
  {
    "question": f'請根據{image_descriptions}裡面的image和{relevant_knowledge}裡面的法規資料回答 請問image3 是違反了哪些法規 的問題',
    "answer": "分析結果: 第 172 條 第二項 雇主對於臨時房舍應有適當之通風及照明。"
  },
]

example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

# 向 Akash API 發送請求
try:
    response = client.chat.completions.create(
        model="DeepSeek-R1",
        messages=[
            {
                "role": "user",
                "content": prompt.format(input=problem)
            }
        ],
    )

    print(textwrap.fill(response.choices[0].message.content, 50))

except Exception as e:
    print(f"發生錯誤: {str(e)}")
