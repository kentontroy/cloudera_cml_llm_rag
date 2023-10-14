import requests

prompt = "Name the top place to visit in SF"

server = "http://127.0.0.1:8000"
streaming = False
if streaming:
  endpoint = server + "/llama/streaming" 
else:
  endpoint = server + "/llama"

res = requests.post(endpoint, json={"prompt": prompt})
res.raise_for_status()

for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
  chunk = chunk.replace("data:", "")
  print(chunk, end="")
  
