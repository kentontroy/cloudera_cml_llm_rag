import streamlit as st

class SidebarModelParams:
  def __init__(self):
    self.__include_sidebar__() 

  def __include_sidebar__(self) -> None:
    st.subheader("Cloudera PSE Team", divider="rainbow")
    st.subheader("GenAI with Intel is :blue[cool] :sunglasses:")
    with st.sidebar:
      self.max_tokens = st.slider(
        label="max_tokens",
        min_value=1,
        max_value=4096,
        value=3000,
        step=4,
        help="Maximun number of tokens to generate from LLM call"
      )
      self.temperature = st.slider(
        label="temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Higher values create more diversity in the response and can stray away from context"
      )
      self.top_k = st.slider(
        label="top_k",
        min_value=0,
        max_value=100,
        value=40,
        step=1,
        help="Instead of using the full distribution, the model only samples from the K most probable tokens"
      )
      self.top_p = st.slider(
        label="top_p",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="""The minimum number of tokens such that the sum of the probabilities of occurence for each 
                token totals to the top_p setting"""
      )
      self.repeat_penalty = st.slider(
        label="repeat_penalty",
        min_value=1.0,
        max_value=3.0,
        value=1.1,
        step=0.1,
        help="The penalty to apply to repeated tokens"
      )
      self.n_tokens_size = st.slider(
        label="n_tokens_size",
        min_value=0,
        max_value=200,
        value=64,
        step=1,
        help="The number of tokens to look back when applying the repeat_penalty"
      )
