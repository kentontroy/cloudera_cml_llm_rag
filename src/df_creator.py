from dotenv import load_dotenv
from langchain.graphs.networkx_graph import NetworkxEntityGraph
import argparse
import ast
import os
import numpy as np
import pandas as pd
import spacy

#####################################################################################
# Read triples from file into a dataframe
#####################################################################################
def readTriplesFromFile(filePath: str) -> pd.DataFrame:
  data = []
  with open(filePath, "r") as f:
    for l in f.readlines():
      line = l.split(":", 1)
      page = line[0].strip()
      triples = ast.literal_eval(line[1].strip())
      subject = triples[0][0].strip()
      object = triples[0][1].strip()
      predicate = triples[0][2].strip()
      data.append([page, subject, object, predicate])
  df = pd.DataFrame(data, columns=["Page", "Subject", "Object", "Predicate"])   
  return df

#####################################################################################
# Get unique subjects and objects
#####################################################################################
def getUniqueSubjectsAndObjects(df: pd.DataFrame) -> pd.DataFrame:
  subjects = df["Subject"].map(str)
  objects = df["Object"].map(str)
  values = np.unique(np.concatenate((subjects, objects)))
  df = pd.DataFrame(values, columns=["Raw_Entity"])
  return df

#####################################################################################
# Get named entities
#####################################################################################
def getNamedEntities(rawDf: pd.DataFrame) -> pd.DataFrame:
  ner = spacy.load("en_core_web_sm")
  text = ner(", ".join(rawDf["Raw_Entity"]))
  for w in text.ents:
    print(w.text, w.label_)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--file", type=str, required=True, help="Specify filename")
  args = parser.parse_args()

  df = readTriplesFromFile(filePath = args.file)
  raw = getUniqueSubjectsAndObjects(df = df)
  getNamedEntities(rawDf = raw)
  
  print(raw)
