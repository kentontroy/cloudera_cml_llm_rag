from dotenv import load_dotenv
from collections.abc import Generator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from model import MyModel
from chain import MyRagQuestionAndAnswer
import gradio as gr
import os

#########################################################################
# Load global variables
#########################################################################
load_dotenv()
MODEL = MyModel(verbosity=False)
RAG = MyRagQuestionAndAnswer(model=MODEL)
TOKENS_PER_STREAM_OUTPUT = int(os.getenv("TOKENS_PER_STREAM_OUTPUT"))
CSS = """
#chatbot .user {
    text-align: right
}
"""

#########################################################################
# Load vector store indexes and prompt templates
#########################################################################
INDEXES = ["None", "cloudera_pse_support", "cloudera_ecs", "game_of_thrones"]
MODELS = ["mistral-7b-v0.1.Q4_K_M.gguf", "mistral-7b-instruct-v0.1.Q4_K_M.gguf"]
PROMPTS = [".blog", ".sentiment", ".troubleshooting"] 

#########################################################################
# Sync UI control values with environment variable settings and
# reload the model
#########################################################################
def reloadModel() -> None:
  global RAG, MODEL
  del RAG
  del MODEL
  MODEL = MyModel(verbosity=False)
  RAG = MyRagQuestionAndAnswer(model=MODEL)

#########################################################################
# Generates a chat response as a stream
#########################################################################
def getResponse(message, history) -> Generator:
  i = 0               
  streamedResponse = ""
  try:
    for nextToken in RAG.write(message=message):
      streamedResponse += nextToken
      i = i + 1       
      if i % TOKENS_PER_STREAM_OUTPUT == 0:
        yield streamedResponse
  except StopIteration:
    return
  except StopAsyncIteration:
    return

#########################################################################
# Event handlers that modify model or chain parameters
# TODO: add validation and deal with -1 flag behavior
#########################################################################
def setIndex(indexName) -> None:
  os.environ["RETRIEVER_INDEX_NAME"] = indexName

def setTemp(temperature) -> None:
  os.environ["MODEL_PARAM_TEMPERATURE"] = str(temperature)

def setContextLength(length) -> None:
  os.environ["MODEL_PARAM_CONTEXT_LEN"] = str(length)

def setMaxTokens(n) -> None:
  os.environ["MODEL_PARAM_MAX_TOKENS"] = str(n)

def setTopK(topK) -> None:
  os.environ["MODEL_PARAM_TOP_K"] = str(topK)

def setTopP(topP) -> None:
  os.environ["MODEL_PARAM_TOP_P"] = str(topP)

def setRepeatLastN(n) -> None:
# TODO: Has a -1 flag when set, last_n_tokens_size = context length
  os.environ["MODEL_PARAM_N_TOKENS_SIZE"] = str(n)

def setRepeatPenalty(p) -> None:
  os.environ["MODEL_PARAM_REPEAT_PENALTY"] = str(p)

def setBatchSize(n) -> None:
  os.environ["MODEL_PARAM_BATCH_SIZE"] = str(n)

def setNumOfThreads(n) -> None:
# TODO: Has a -1 flag when set, n_threads is automatically determined
  os.environ["MODEL_PARAM_THREADS"] = str(n)

def setModel(name) -> None:
  os.environ["LLM_MODEL_FILE"] = name

def enableTracking(v) -> None:
  os.environ["MODEL_TRACKING_ENABLED"] = str(v)

def search(indexName, query, neighbors) -> str:
  EMBEDDINGS_MODEL_PATH = os.getenv("EMBEDDINGS_MODEL_PATH")
  embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_PATH, model_kwargs={"device": "cpu"})
  DB_FAISS_PATH = os.getenv("DB_FAISS_PATH")
  db = FAISS.load_local(os.path.join(DB_FAISS_PATH, indexName), embeddings)
  results = db.similarity_search_with_score(query, k=neighbors)
  output = ""
  for doc, score in results:
    output += "===============================\n"
    output += f"Distance:\n"
    output += "===============================\n"
    output += f"{score}\n\n"
    output += f"Content:\n"
    output += "===============================\n"
    output += f"{doc.page_content}\n\n"
    output += f"Metadata:\n"
    output += "===============================\n"
    output += f"{doc.metadata}\n\n"
  return output

def setPromptTemplate(template) -> str:
  os.environ["PROMPT_TEMPLATE_NAME"] = template 
  with open(template, "r") as f:  
    prompt = f.read()
  return prompt

#########################################################################
# Main UI construction
#########################################################################
with gr.Blocks(css=CSS, theme=gr.themes.Glass()) as demo:
  with gr.Row():
    with gr.Column(scale=1, min_width=200):
      indexDropdown = gr.Dropdown(
        INDEXES,
        multiselect = False,  
        value = os.getenv("RETRIEVER_INDEX_NAME"),
        label = "Vector store index",
        scale = 0
      )
      indexDropdown.change(fn=setIndex, inputs=indexDropdown)

      with gr.Row():
        tempSlider = gr.Slider(
          minimum=0, maximum=1, value=os.getenv("MODEL_PARAM_TEMPERATURE"), label="Temperature", 
          step=0.1, interactive=True)
        tempSlider.change(fn=setTemp, inputs=tempSlider, outputs=[]) 
      with gr.Row():
        contextSlider = gr.Slider(
          minimum=0, maximum=4096, value=os.getenv("MODEL_PARAM_CONTEXT_LEN"), label="Context Length", 
          step=4, interactive=True)
        contextSlider.change(fn=setContextLength, inputs=contextSlider, outputs=[]) 
      with gr.Row():
        tokensSlider = gr.Slider(
          minimum=10, maximum=4096, value=os.getenv("MODEL_PARAM_MAX_TOKENS"), label="Max tokens to generate", 
          step=4, interactive=True
        )
        tokensSlider.change(fn=setMaxTokens, inputs=tokensSlider, outputs=[]) 
      with gr.Row():
        topKSlider = gr.Slider(
          minimum=5, maximum=50, value=os.getenv("MODEL_PARAM_TOP_K"), label="top_k", 
          step=1, interactive=True, container=True
        )
        topKSlider.change(fn=setTopK, inputs=topKSlider, outputs=[]) 
        topPSlider = gr.Slider(
          minimum=0.10, maximum=0.95, value=os.getenv("MODEL_PARAM_TOP_P"), label="top_p", 
          step=0.05, interactive=True, container=True
        )
        topPSlider.change(fn=setTopP, inputs=topPSlider, outputs=[]) 
      with gr.Row():
        repeatLastNSlider = gr.Slider(
          minimum=-1, maximum=150, value=os.getenv("MODEL_PARAM_N_TOKENS_SIZE"), label="repeat_last_n", 
          step=1, interactive=True, container=True
        )
        repeatLastNSlider.change(fn=setRepeatLastN, inputs=repeatLastNSlider, outputs=[]) 
        repeatPenaltySlider = gr.Slider(
          minimum=0.5, maximum=2, value=os.getenv("MODEL_PARAM_REPEAT_PENALTY"), label="repeat_penalty", 
          step=0.1, interactive=True, container=True
        )
        repeatPenaltySlider.change(fn=setRepeatPenalty, inputs=repeatPenaltySlider, outputs=[]) 
      with gr.Row():
        batchSlider = gr.Slider(
          minimum=8, maximum=512, value=os.getenv("MODEL_PARAM_BATCH_SIZE"), label="n_batch", 
          info="Should be <= context length", step=1, interactive=True, container=True
        )
        batchSlider.change(fn=setBatchSize, inputs=batchSlider, outputs=[]) 
        threadsSlider = gr.Slider(
           minimum=-1, maximum=64, value=os.getenv("MODEL_PARAM_THREADS"), label="n_threads", 
           info="If -1 then automatically set", step=1, interactive=True, container=True)
        threadsSlider.change(fn=setNumOfThreads, inputs=threadsSlider, outputs=[]) 
      with gr.Row():
        modelDropdown = gr.Dropdown(
          MODELS,
          multiselect = False,  
          value = os.getenv("LLM_MODEL_FILE"),
          label = "Quantized model version",
          scale = 1
        )
        modelDropdown.change(fn=setModel, inputs=modelDropdown)
      with gr.Row():
# TODO: Gradio docs say icons must be within the working directory of the Gradio app or an external URL.
        btnReload = gr.Button(value="Reload model", icon="refresh.png")
        btnReload.click(reloadModel, inputs=[], outputs=[])
        chkTracking = gr.Checkbox(value=True, label="Enable model tracking", info="Track prompts and responses")
        chkTracking.change(fn=enableTracking, inputs=chkTracking)

    with gr.Column(scale=2, min_width=400):
      bot = gr.Chatbot(
        render=False, 
        elem_id="chatbot"
      )
      chat = gr.ChatInterface(
        fn=getResponse,
        chatbot=bot, 
        examples=["What does the error in the stack trace - No module named 'utils' mean?"],
      )

vectorSearchDropdown = gr.Dropdown(
# Skip the first entry in the Index which is "None"
  INDEXES[1:],
  multiselect = False,  
  value = os.getenv("RETRIEVER_INDEX_NAME"),
  label = "Vector store index",
  scale = 1,
  interactive = True
)
query = gr.Textbox(
  label="Query / message",
  info="Consider what you would type in a ChatGPT dialogue",
  lines=3,
  scale=2
)
txtSearch = gr.TextArea(label="", lines=10)
neighborsSlider = gr.Slider(
  minimum=1, maximum=10, value=1, label="Closest Documents", 
  info="Similarity search against vector store", step=1, 
  interactive=True, container=True
)
with gr.Blocks(theme=gr.themes.Glass()) as vectorSearch:
  gr.Interface(
    fn=search,
    inputs=[ vectorSearchDropdown, query, neighborsSlider ],
    outputs=txtSearch,
    allow_flagging="never" 
  )

with gr.Blocks(theme=gr.themes.Glass()) as promptTemplate:
  promptTemplateDropdown = gr.Dropdown(
    PROMPTS,
    multiselect = False,  
    value = os.getenv("PROMPT_TEMPLATE_NAME"),
    label = "Prompt Template",
    scale = 1,
    interactive = True
  )
  with open(os.getenv("PROMPT_TEMPLATE_NAME"), "r") as f:
    defaultTemplate = f.read()
  txtPrompt = gr.TextArea(
    label="", 
    lines=10, 
    value=defaultTemplate,
    interactive=False
  )
  promptTemplateDropdown.change(fn=setPromptTemplate, inputs=promptTemplateDropdown, outputs=txtPrompt)
  btnRefresh = gr.Button(value="Reload model", icon="refresh.png")
  btnRefresh.click(
    reloadModel, 
    inputs=[],
    outputs=[]
  )

tabbed = gr.TabbedInterface(
  [demo, vectorSearch, promptTemplate], 
  ["Chatbot", "Vector search", "Prompt template"]
)
tabbed.queue()

if __name__ == "__main__":
  tabbed.launch(share=True,
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
  
