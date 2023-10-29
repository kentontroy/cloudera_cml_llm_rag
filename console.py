from chain import MyRagQuestionAndAnswer
from dotenv import load_dotenv
from model import MyModel
import argparse
import os

if __name__ == "__main__":
  load_dotenv()
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--index", type=str, required=False, help="Specify an index in the vector database")
  parser.add_argument("-q", "--query", type=str, required=False, help="Input a query")
  parser.add_argument("-v", "--verbose", action="store_true", required=False, help="Set verbose trace level")
  args = parser.parse_args()

  if (args.index and args.query):
    os.environ["RETRIEVER_INDEX_NAME"] = args.index
  elif args.query:
    os.environ["RETRIEVER_INDEX_NAME"] = "None"
  else:
    print("Incorrect usage: python llm.py [-h] to get help on command options")
    exit()

  TOKENS_PER_STREAM_OUTPUT = int(os.getenv("TOKENS_PER_STREAM_OUTPUT"))

  model = MyModel(verbosity=args.verbose)
  
  print(args.query)
  print("===================================\n")
  i = 0               
  streamedResponse = ""
  rag = MyRagQuestionAndAnswer(model=model)
  for nextToken in rag.write(message=args.query):
    streamedResponse += nextToken
    i = i + 1       
    if i % TOKENS_PER_STREAM_OUTPUT == 0:
      print(streamedResponse, end="")
      streamedResponse = ""
  print("\n")
                            
  del rag.chain
  del model.llm   

