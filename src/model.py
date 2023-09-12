from enum import Enum, auto
from langchain.llms import LlamaCpp, GPT4All
import os

class ModelName(Enum):
  LLAMA2 = auto()
  GPT4All = auto()

class MyModel:
  def __init__(self, verbosity=False, modelName: ModelName = ModelName.LLAMA2):
    LLAMA2_MODEL_PATH = os.getenv("LLAMA2_MODEL_PATH")
    LLAMA2_MODEL_FILE = os.getenv("LLAMA2_MODEL_FILE")
    GPT4ALL_MODEL_PATH = os.getenv("GPT4ALL_MODEL_PATH")
    MODEL_PARAM_CONTEXT_LEN = os.getenv("MODEL_PARAM_CONTEXT_LEN")
    MODEL_PARAM_BATCH_SIZE = os.getenv("MODEL_PARAM_BATCH_SIZE")
    MODEL_PARAM_TEMPERATURE = os.getenv("MODEL_PARAM_TEMPERATURE")
    MODEL_PARAM_MAX_TOKENS = os.getenv("MODEL_PARAM_MAX_TOKENS")
    MODEL_PARAM_TOP_K = os.getenv("MODEL_PARAM_TOP_K")
    MODEL_PARAM_TOP_P = os.getenv("MODEL_PARAM_TOP_P")
    MODEL_PARAM_MLOCK = os.getenv("MODEL_PARAM_MLOCK")
    MODEL_PARAM_THREADS = os.getenv("MODEL_PARAM_THREADS")
    MODEL_PARAM_REPEAT_PENALTY = os.getenv("MODEL_PARAM_REPEAT_PENALTY") 
    MODEL_PARAM_N_TOKENS_SIZE = os.getenv("MODEL_PARAM_N_TOKENS_SIZE")
    MODEL_PARAM_THREADS = os.getenv("MODEL_PARAM_THREADS")

# TODO: programmatically control whether or not the GPU is used
    gpuLayers = 1

# Only Python 3.10+ supports switch statements and pattern matching
    if modelName == ModelName.LLAMA2:
      modelPath = os.path.join(LLAMA2_MODEL_PATH, LLAMA2_MODEL_FILE)
      print("Using Llama2 model")
      print("Model path: {0}".format(modelPath))
      self.llm = LlamaCpp(
        model_path = modelPath,
        n_ctx = MODEL_PARAM_CONTEXT_LEN,
        n_batch = MODEL_PARAM_BATCH_SIZE,
        temperature = MODEL_PARAM_TEMPERATURE,
        max_tokens = MODEL_PARAM_MAX_TOKENS,
        n_gpu_layers = gpuLayers,
        top_k = MODEL_PARAM_TOP_K,
        top_p = MODEL_PARAM_TOP_P,
        last_n_tokens_size = MODEL_PARAM_N_TOKENS_SIZE,
        repeat_penalty = MODEL_PARAM_REPEAT_PENALTY,
        n_threads = MODEL_PARAM_THREADS,
        use_mlock = MODEL_PARAM_MLOCK,
        f16_kv = True,
        verbose = verbosity,
        streaming = True)
    else:
      print("Using GPT4All model")
      self.llm = GPT4All(
        model = GPT4ALL_MODEL_PATH,
        n_batch = MODEL_PARAM_BATCH_SIZE,
        temp = MODEL_PARAM_TEMPERATURE,
        max_tokens = MODEL_PARAM_MAX_TOKENS,
        top_k = MODEL_PARAM_TOP_K,
        top_p = MODEL_PARAM_TOP_P,
        last_n_tokens_size = MODEL_PARAM_N_TOKENS_SIZE,
        repeat_penalty = MODEL_PARAM_REPEAT_PENALTY,
        n_threads = MODEL_PARAM_THREADS,
        use_mlock = MODEL_PARAM_MLOCK,
        f16_kv = True,
        verbose = verbosity,
        streaming = True)
