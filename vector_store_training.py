# vectorizing
from langchain_openai import OpenAIEmbeddings
import os

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large") # now use text-embedding-3-large instead of text-embedding-ada-002

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader

DB_FAISS_PATH = "resources/vectorstore/db_faiss"

loader = CSVLoader(file_path="resources/Finished_Data_in_769audio_vn.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
print("finish loading data")
# Split the text into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        ],
    chunk_size=65,
    chunk_overlap=2**4
    )

text_chunks = text_splitter.split_documents(data)
print("finish splitting")
# Converting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
print("start creating docsearch")
docsearch = FAISS.from_documents(text_chunks, embeddings)
print("start saving locally docsearch")
docsearch.save_local(DB_FAISS_PATH)
print("done mission")