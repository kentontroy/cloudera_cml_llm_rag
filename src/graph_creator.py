from characters import characters
from dotenv import load_dotenv
from langchain.graphs.networkx_graph import NetworkxEntityGraph
import argparse
import df_creator as dfCreate
import os
import pandas as pd
import pandasql as ps


if __name__ == "__main__":
  tripleFiles = [
    "game_of_thrones_pages_10_109.txt",
    "game_of_thrones_pages_110_159.txt",
    "game_of_thrones_pages_160_209.txt",
    "game_of_thrones_pages_210_259.txt",
    "game_of_thrones_pages_260_309.txt",
    "game_of_thrones_pages_310_359.txt",
    "game_of_thrones_pages_360_409.txt",
    "game_of_thrones_pages_410_459.txt",
    "game_of_thrones_pages_460_509.txt",
    "game_of_thrones_pages_510_552.txt"
  ] 

  load_dotenv()
  dfTriples = pd.DataFrame()
  for t in tripleFiles:
    path = os.path.join(os.getenv("TRIPLES_DIR_FILES"), t)
    df = dfCreate.readTriplesFromFile(filePath = path) 

    for label in characters:
      tempDf = dfCreate.getTriples(df = df, label = label)
      dfTriples = pd.concat([dfTriples, tempDf], ignore_index=True, axis=0)

  dfTriples.drop_duplicates(inplace = True)
  print(dfTriples)
  
