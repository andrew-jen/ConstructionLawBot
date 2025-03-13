# ConstructionLawBot
Use AI to answer questions about pictures of construction.
![image](https://github.com/user-attachments/assets/93d02464-fb73-4cb8-b98e-fb22c254b320)

透過Akash模型，以RAG、embedding以及Yolo訓練出來的深度學習模型(目前僅能判斷有無安全帽)，可以判斷放在yolo-workspace-main\images-file資料夾內的照片，
並對於用戶提出關於圍觀照片的問題做出回復。

程式執行:
執行s.crawl.py會爬蟲工地法規網站，並將資料存為law_data.json檔。
執行myyolo.py會檢視images-file資料夾內的所有照片，並存為result.json檔。
執行ragtest.py可以與AI進行對話詢問。
