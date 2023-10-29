from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import argparse
import os

#####################################################################################
# Facebook AI Similarity Search (FAISS)
#####################################################################################
def createIndexFromDocs(embeddings, dataPath, indexName, isPDF=True):
    DB_FAISS_PATH = os.getenv("DB_FAISS_PATH") 
    if isPDF:
      loader = DirectoryLoader(dataPath, glob="*.pdf", loader_cls=PyPDFLoader)
    else:
      loader = DirectoryLoader(dataPath, glob="*.txt")

    documents = loader.load()
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                                  chunk_overlap=100)
    texts = textSplitter.split_documents(documents)
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(os.path.join(DB_FAISS_PATH, indexName))

#####################################################################################
# The top responses from FAISS are sorted in ascending order of similarity score
# FAISS uses L2-Norm (Euclidean distance) such that lower scores represent a closer 
# distance to the query of interest
#####################################################################################
def queryIndex(embeddings, indexName, query, k=3):
    DB_FAISS_PATH = os.getenv("DB_FAISS_PATH") 
    index = FAISS.load_local(os.path.join(DB_FAISS_PATH, indexName), embeddings)
    results = index.similarity_search_with_score(query, k=k)
    for doc, score in results:
        print(f"Score: {score}\n")
        print(f"Metadata: {doc.metadata}") 
        print(f"Content: {doc.page_content}") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--create", type=str, required=False, help="Create the vector database, specify index name")
    parser.add_argument("-p", "--pdfs", type=str, required=False, help="Specify a directory containing PDF files")
    parser.add_argument("-t", "--text", type=str, required=False, help="Specify a directory containing text files")
    parser.add_argument("-q", "--query", type=str, required=False, help="Query the vector database, pass search string")
    parser.add_argument("-i", "--index", type=str, required=False, help="Specify an index in the vector database")
    args = parser.parse_args()

    load_dotenv()
    EMBEDDINGS_MODEL_PATH = os.getenv("EMBEDDINGS_MODEL_PATH")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_PATH, model_kwargs={"device": "cpu"})

    if args.create and args.pdfs and not args.text:
        createIndexFromDocs(embeddings=embeddings, dataPath=args.pdfs, indexName=args.create)
    elif args.create and args.text and not args.pdfs:
        createIndexFromDocs(embeddings=embeddings, dataPath=args.text, indexName=args.create, isPDF=False)
    elif args.query and args.index:
        queryIndex(embeddings=embeddings, indexName=args.index, query=args.query)
    else:
        print("Incorrect usage: python vectorStore.py [-h] to get help on command options")
