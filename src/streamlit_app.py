import requests
import streamlit as st
import streamlit_app_model_endpoint as endpoint
from streamlit_app_sidebar_params import SidebarModelParams

params = SidebarModelParams()

# Initialize session state
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you?"):
# Add user message to chat history
  st.session_state.messages.append({"role": "user", "content": prompt})
# Display user message
  with st.chat_message("user"):
    st.markdown(prompt)
# Display assistant response
  with st.chat_message("assistant"):
    placeholder = st.empty()
    fullResponse = ""

  with open(".query", "r") as f:
    template = f.read()

# Include previous response from the assistant in the current prompt
  previous = ""
  for msg in reversed(st.session_state.messages):
    if msg["role"] == "assistant":
      previous = msg["content"] 
      break
  query = ""
  if previous != "":
    query = template.format(previous, prompt)
  else:
    query = prompt
  print(query)

# Interact with the chat server
  input = {
    "prompt": query,
    "max_tokens": params.max_tokens,
    "temperature": params.temperature,
    "top_p": params.top_p,
    "top_k": params.top_k,
    "repeat_penalty": params.repeat_penalty,
    "n_tokens_size": params.n_tokens_size
  }
  res = requests.post(endpoint.endpoint, json=input)
  res.raise_for_status()
  for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
    chunk = chunk.replace("data:", "")
    chunk = chunk.replace("\r\n", "")
    fullResponse += chunk
    placeholder.markdown(fullResponse + "▌")

  placeholder.markdown(fullResponse)
  st.session_state.messages.append({"role": "assistant", "content": fullResponse})

# Create Reset and Clear button layout
colHist1, colHist2, dummy1, dummy2, dummy3 = st.columns(5)

with colHist1:
  resetBtnKey = "reset_button"
  resetBtn = st.button("Reset Context", key=resetBtnKey)

with colHist2:
  clearBtnKey = "clear_button"
  clearBtn = st.button("Clear Chat", key=clearBtnKey)

if resetBtn:
  st.session_state.messages.clear()

if clearBtn:
  st.session_state.messages.clear()
  st.rerun()
