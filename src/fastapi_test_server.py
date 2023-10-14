import copy
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from llama_cpp import Llama
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.INFO)
logger = logging.getLogger("llm_rag_app")
app = FastAPI()
load_dotenv()

class LLMRequest(BaseModel):
  prompt: str
  temperature: float = os.getenv("MODEL_PARAM_TEMPERATURE")
  max_tokens: int = os.getenv("MODEL_PARAM_MAX_TOKENS")
  top_p: float = os.getenv("MODEL_PARAM_TOP_P")
  top_k: float = os.getenv("MODEL_PARAM_TOP_K")
  repeat_penalty: float = os.getenv("MODEL_PARAM_REPEAT_PENALTY")
  n_tokens_size: int = os.getenv("MODEL_PARAM_N_TOKENS_SIZE")

PATH = os.path.join(os.getenv("LLM_MODEL_PATH"), os.getenv("LLM_MODEL_FILE"))
MODEL = Llama(
  model_path = PATH, 
  n_ctx = int(os.getenv("MODEL_PARAM_CONTEXT_LEN")),
  n_batch = int(os.getenv("MODEL_PARAM_BATCH_SIZE")),
  use_mlock = os.getenv("MODEL_PARAM_MLOCK"),
  n_threads = int(os.getenv("MODEL_PARAM_THREADS")),
  n_gpu_layers = 0,
  f16_kv = True,
  verbose = False
)

@app.post("/llm")
async def query(req: LLMRequest) -> str:
  logger.info(f'Got prompt: "{req.prompt}"')
  res = MODEL(
    prompt = req.prompt,
    max_tokens = int(req.max_tokens),
    temperature = float(req.temperature),
    top_p = float(req.top_p),
    top_k = int(req.top_k),
    repeat_penalty = float(req.repeat_penalty),
    stream = False
  )
  text = res["choices"][0]["text"].strip()
  return text

@app.post("/llm/streaming")
async def stream(req: LLMRequest, request: Request):
  logger.info(f'Got prompt: "{req.prompt}"')
  stream = MODEL(
    prompt = req.prompt,
    max_tokens = int(req.max_tokens),
    temperature = float(req.temperature),
    top_p = float(req.top_p),
    top_k = int(req.top_k),
    repeat_penalty = float(req.repeat_penalty),
    stream = True
  )
  async def generateStream():
    for entry in stream:
      yield entry

  async def sendEvents():
    async for entry in generateStream():
      if await request.is_disconnected():
        break
      res = copy.deepcopy(entry)
      text = res["choices"][0]["text"].strip()
      yield text 
 
  return EventSourceResponse(sendEvents())


