
server = "http://127.0.0.1:8000"
streaming = False
if streaming:
  endpoint = server + "/llm/streaming"
else:
  endpoint = server + "/llm"
