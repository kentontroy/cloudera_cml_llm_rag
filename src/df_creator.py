from dotenv import load_dotenv
from langchain.graphs.networkx_graph import NetworkxEntityGraph
import argparse
import ast
import os
import numpy as np
import pandas as pd
import pandasql as ps

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
def getTriples(df: pd.DataFrame, label: str) -> pd.DataFrame:
  subjects = df["Subject"].map(str)
  objects = df["Object"].map(str)
  query = f"""SELECT Page, Subject, Object, Predicate 
              FROM df 
              WHERE Subject LIKE '{label}%' OR Object LIKE '{label}%'
           """
  return ps.sqldf(query, locals())


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--file", type=str, required=True, help="Specify the filename containing the triples")
  parser.add_argument("-l", "--label", type=str, required=True, help="Specify the label of interest")
  args = parser.parse_args()

  df = readTriplesFromFile(filePath = args.file)
  dfQuery = getTriples(df = df, label = args.label)
  print(dfQuery) 
