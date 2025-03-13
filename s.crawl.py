import json
import requests
from bs4 import BeautifulSoup

# 发送请求获取网页内容
response = requests.get("https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0060014")
soup = BeautifulSoup(response.text, "html.parser")

# 解析数据
data_list = []
n = 1
result = soup.find_all('div', class_='row')

for tag in result:
    data_list.append({"name": n, "val": tag.text.strip()})
    n += 1

# 将数据存成 JSON 文件
json_path = "law_data.json"
with open(json_path, "w", encoding="utf-8") as json_file:
    json.dump(data_list, json_file, indent=4, ensure_ascii=False)

print(f"✅ 数据已保存到 {json_path}")
