from collections.abc import Generator
from datetime import date, datetime
from handler import MyQueueHandler
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from model import MyModel
from queue import Queue, Empty
from threading import Thread
import json
import os
import uuid

class MyRagQuestionAndAnswer:
  def __init__(self, model: MyModel):
    EMBEDDINGS_MODEL_PATH = os.getenv("EMBEDDINGS_MODEL_PATH")
    with open(os.getenv("PROMPT_TEMPLATE_NAME"), "r") as f:
      PROMPT_TEMPLATE = f.read()
    promptTemplate = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "message"])
    self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_PATH, model_kwargs={"device": "cpu"})
    self.model = model
    self.chain = LLMChain(llm = self.model.llm, prompt=promptTemplate)
    
# TODO: pre-loading fixed index stores for demo purposes
    DB_FAISS_PATH = os.getenv("DB_FAISS_PATH")
    self.vectorStore = { 
      "game_of_thrones":      FAISS.load_local(os.path.join(DB_FAISS_PATH, "game_of_thrones"), self.embeddings),
      "cloudera_ecs":         FAISS.load_local(os.path.join(DB_FAISS_PATH, "cloudera_ecs"), self.embeddings),
      "cloudera_pse_support": FAISS.load_local(os.path.join(DB_FAISS_PATH, "cloudera_pse_support"), self.embeddings) 
    }

  def track(self, input: {}, output: str) -> None:
    MODEL_TRACKING_ENABLED = os.getenv("MODEL_TRACKING_ENABLED")
    if not MODEL_TRACKING_ENABLED.lower() in ("yes", "true", "t", "1"):
      return
    MODEL_TRACKING_DIRECTORY = os.getenv("MODEL_TRACKING_DIRECTORY")

    randStr = uuid.uuid4().hex
    fileName = str(date.today()) + "_" + randStr + "_message.json"
    record = json.dumps(input, indent = 4) 
    with open(os.path.join(MODEL_TRACKING_DIRECTORY, fileName), "w") as f:
      f.write(record) 
  
    fileName = str(date.today()) + "_" + randStr + "_response.json"
    with open(os.path.join(MODEL_TRACKING_DIRECTORY, fileName), "w") as f:
      f.write(output) 

    modelParams = { 
      "TIMESTAMP":                  str(datetime.utcnow()),
      "RETRIEVER_INDEX_NAME":       os.getenv("RETRIEVER_INDEX_NAME"),
      "MODEL_NAME":                 "LLaMa2",
      "MODEL_PATH":                 self.model.llm.model_path,
      "MODEL_PARAM_CONTEXT_LEN":    self.model.llm.n_ctx,
      "MODEL_PARAM_BATCH_SIZE":     self.model.llm.n_batch,
      "MODEL_PARAM_TEMPERATURE":    self.model.llm.temperature,
      "MODEL_PARAM_MAX_TOKENS":     self.model.llm.max_tokens,
      "MODEL_PARAM_GPU_LAYERS":     self.model.llm.n_gpu_layers,
      "MODEL_PARAM_TOP_K":          self.model.llm.top_k,
      "MODEL_PARAM_TOP_P":          self.model.llm.top_p,
      "MODEL_PARAM_N_TOKENS_SIZE":  self.model.llm.last_n_tokens_size,
      "MODEL_PARAM_REPEAT_PENALTY": self.model.llm.repeat_penalty,
      "MODEL_PARAM_THREADS":        self.model.llm.n_threads,
      "MODEL_PARAM_MLOCK":          self.model.llm.use_mlock
    }
    record = json.dumps(modelParams, indent = 4) 
    fileName = str(date.today()) + "_" + randStr + "_metadata.json"
    with open(os.path.join(MODEL_TRACKING_DIRECTORY, fileName), "w") as f:
      f.write(record) 
    

#############################################################################################
# Paraphrased from langchain docs 
# The vector store split the documents into chunks. Summarization is done in a recursive 
# manner. The process summarizes each chunk by itself, then groups the summaries into chunks 
# and summarizes each chunk of summaries. The process continues until only one chunk is left.
##############################################################################################
  def write(self, message: str, k=4) -> Generator:
    q = Queue()
    handler = MyQueueHandler(q)
    jobComplete = object()

# TODO: for quick demo purposes, assumes the use of Stuffing vs. MR and k <= 4 in 1024 token chunks
#       yielding a context window = 1024 * k
    RETRIEVER_INDEX_NAME = os.getenv("RETRIEVER_INDEX_NAME")
    if RETRIEVER_INDEX_NAME in self.vectorStore:
      print("Retrieval augmentation using index: {0}\n".format(RETRIEVER_INDEX_NAME))
      index = self.vectorStore[RETRIEVER_INDEX_NAME]
      docs = index.similarity_search(message, k=k)
      context = "\n".join([doc.page_content for doc in docs])
      input = {"context": context, "message": message}
    else:
      print("Unable to find an index, skipping retrieval augmentation\n")
      input = {"context": "", "message": message}

    def task():
      self.chain.apply([input], callbacks=[handler])
      q.put(jobComplete)

    t = Thread(target=task)
    t.start()
    output = ""
    while True:
      try:
        nextToken = q.get(True, timeout=1)
        if nextToken is jobComplete:
          break
        output += nextToken
        yield nextToken
      except Empty:
        continue

    self.track(input, output)
