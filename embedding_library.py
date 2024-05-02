from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader # will load text from document so no need python `with open , doc.read()`
from langchain_community.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from langfuse.callback import CallbackHandler
from io import StringIO
import json
import time
from PyPDF2 import PdfReader
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser



load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

langfuse = Langfuse(
  secret_key=os.getenv("LANG_SECRET_KEY"),
  public_key=os.getenv("LANG_PUBLIC_KEY"),
  host=os.getenv("LANG_HOST")
)

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANG_SECRET_KEY"),
    public_key=os.getenv("LANG_PUBLIC_KEY"),
    host=os.getenv("LANG_HOST"), # just put localhost here probably
)


### LANGCHAIN EMBEDDING AND RETRIVAL PART

# VAR; get doc/text and split in chunks cardano_meme_coin.txt, best_meme_coins_2024.txt, history_of_coins.txt
docs_folder = "./docs"
list_documents_txt = []
for doc in os.listdir(docs_folder):
  list_documents_txt.append(doc)


# Use ollama to create embeddings
embeddings = OllamaEmbeddings(temperature=0)

# define connection to pgvector database
CONNECTION_STRING = PGVector.connection_string_from_db_params(
     driver=os.getenv("DRIVER"),
     host=os.getenv("HOST"),
     port=int(os.getenv("PORT")),
     database=os.getenv("DATABASE"),
     user=os.getenv("USER"),
     password=os.getenv("PASSWORD"),
)
# define collection name
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

def load_docs(path: str, file_name: str) -> str:
    all_text = ""
    #print("file name end: ", file_name.split(".")[1])
    if file_name.split(".")[1] == "pdf":
      print(f"File at path: {path}/{file_name}")
      pdf_reader = PdfReader(f"{path}/{file_name}")
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        all_text += text
    elif file_name.split(".")[1] == "txt":
      stringio = StringIO(file_uploaded.getvalue().decode("utf-8"))
      text = stringio.read()
      all_text += text
    #print(f"All Text [0, 100] for '{file_name}': \n{all_text[0:100]}\n")
    return all_text

# HELPER functions , create collection, retrieve from collection, chunk documents
def chunk_doc(path: str, files: list) -> list:
  list_docs = []
  for f in files:
    # loader has the text of the file
    loader = load_docs(path, f) # TextLoader(f"{path}/{file}")
    # documents = loader.load()
    documents = [Document(page_content=loader, metadata={"source": f"{f}"})]
    # using CharaterTextSplitter
    # text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=200, chunk_overlap=20)
    # using RecursiveCharacterTextSplitter (maybe better)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    docs = text_splitter.split_documents(documents)
    list_docs.append(docs)
    #print(f"Doc: {docs}\nLenght list_docs: {len(list_docs)}")
  return list_docs


# using PGVector
def vector_db_create(doc, collection, connection):
  db_create = PGVector.from_documents(
    embedding=embeddings,
    documents=doc,
    collection_name=collection, # must be unique
    connection_string=connection,
    distance_strategy = DistanceStrategy.COSINE,
  )
  return db_create

# PGVector retriever
def vector_db_retrieve(collection, connection, embedding):
  db_retrieve = PGVector(
    collection_name=collection,
    connection_string=connection,
    embedding_function=embedding,
  )
  return db_retrieve

# PGVector update collection
def vector_db_override(doc, embedding, collection, connection):
  changed_db = PGVector.from_documents(
    documents=doc,
    embedding=embedding,
    collection_name=collection,
    connection_string=connection,
    distance_strategy = DistanceStrategy.COSINE,
    pre_delete_collection=True,
  )
  return changed_db


### USE OF EMBEDDING HELPER FOR BUSINESS LOGIC
## Creation of the collection
all_docs = chunk_doc(docs_folder, list_documents_txt) # list_documents_txt
def create_embedding_collection(all_docs: list, COLLECTION_NAME: str, CONNECTION_STRING: str) -> str:
  collection_name = COLLECTION_NAME
  connection_string = CONNECTION_STRING
  count = 1
  for doc in all_docs:
    print(f"Doc number: {count} with lenght: {len(doc)}")
    vector_db_create(doc, collection_name, connection_string) # this to create/ maybe override also
    # vector_db_override(doc, embeddings, collection_name, connection_string) # to override
    count += 1
  return f"Collection created with documents: {count}"
# print(create_embedding_collection(all_docs))

##  similarity query
#question = "what is the capital city of Australia"

def similarity_search(question):
  db = vector_db_retrieve(COLLECTION_NAME, CONNECTION_STRING, embeddings)
  docs_and_similarity_score = db.similarity_search_with_score(question)
  for doc, score in docs_and_similarity_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
#print("********** SIMILARITY *************\n", similarity_search(question))

## MMR (Maximal Marginal Relevance) query
def MMR_search(question):
  db = vector_db_retrieve(COLLECTION_NAME, CONNECTION_STRING, embeddings)
  docs_and_MMR_score = db.max_marginal_relevance_search_with_score(question)
  for doc, score in docs_and_MMR_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
#print("********** MMR *************\n", MMR_search(question))

## OR use ollama query embedding
text = "How many needs are they in Chikara houses?"
def ollama_embedding(text):
  query_result = embeddings.embed_query(text)
  return query_result

def answer_retriever(query, collection, connection, embedding):
  db = vector_db_retrieve(collection, connection, embedding)
  llm = ChatOllama(model="mistral:7b")
  retriever = db.as_retriever(
    search_kwargs={"k": 3} # 3 best responses
  )
  retrieve_answer = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
  )
  query = f"{query}"
  response = retrieve_answer.invoke(query, config={"callbacks": [langfuse_handler]})
  # return display(Markdown(response))
  print("RETRIEVAL RESPONSE Query: ", response["query"])
  print("RETRIEVAL RESPONSE Result: ", response["result"])
  return {
    "Question": query,
    "Response": response["result"],
  }

#print("**** STRATING EMBEDDINGS OF DOCUMENTS ****")
#create_embedding_collection(all_docs, COLLECTION_NAME, CONNECTION_STRING)
#print("**** DONE STARTING RETRIEVAL FROM QUERY ****")
#text_query = "What country and capital is at the number 153?"
#print(answer_retriever(text_query, COLLECTION_NAME, CONNECTION_STRING, embeddings))
#db = vector_db_retrieve(COLLECTION_NAME, CONNECTION_STRING, embeddings)
#print("***********DB************: -->>\n", db.page_content)



###### BUSINESS LOGIC #####

# define collection name

#user_doc = input("Please enter document path: ")
#print("user doc . ...: ", user_doc[-3:])

# Get collection name from document name
#if user_doc[-3:] == "txt":
#  COLLECTION_NAME = user_doc.split("Please enter document path: ")[0].split("/")[-1].split(".txt")[0].strip()
#elif user_doc[-3:] == "pdf":
#  COLLECTION_NAME = user_doc.split("Please enter document path: ")[0].split("/")[-1].split(".pdf")[0].strip()

# get user query
#user_query = input("What do you want us to do from this document? ")
#topic = user_query.split("What do you want us to do from this document? ")[0].strip()

import random

# use random user for the trace
users = [
  {
    "name": "Yola_Docs",
    "user_id": "34",
  },
  {
  "name": "Brol_Docs",
  "id":"86",
  },
    {
    "name": "May_Docs",
    "id": "45"
  },
  {
  "name": "Dinu_Docs",
  "id":"320",
  }
  
]

#print(random.choice(users))
#doc_path = f"{('/').join(user_doc.split('Please enter document path: ')[0].split('/')[:-1])}"
#print("doc path: ", doc_path)
#doc_name = user_doc.split("Please enter document's path: ")[0].split("/")[-1]
#print("doc name: ", doc_name)
#text_doc = load_docs(doc_path, doc_name)


# Chat with an intelligent assistant in your terminal
from openai import OpenAI

# Point to the local server
LM_OPENAI_API_BASE=os.getenv("LM_OPENAI_API_BASE")
LM_OPENAI_MODEL_NAME=os.getenv("LM_OPENAI_MODEL_NAME")
LM_OPENAI_API_KEY=os.getenv("LM_OPENAI_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
groq_client=Groq()
groq_llm_mixtral_7b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_MIXTRAL_7B"),
max_tokens=int(os.getenv("GROQ_MAX_TOKEN")))
groq_llm_llama3_70b = ChatGroq(temperature=float(os.getenv("GROQ_TEMPERATURE")), groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("MODEL_LLAMA3_70B"), max_tokens=int(os.getenv("GROQ_MAX_TOKEN")))
# client = OpenAI(base_url="http://localhost:1235/v1", api_key="lm-studio")
lmstudio_llm = OpenAI(base_url=LM_OPENAI_API_BASE, api_key=LM_OPENAI_MODEL_NAME)

"""
history = [
    {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
    {"role": "user", "content": f"here is the text: {text_doc}.\n Help me with that: {topic}"},
]

def answer_retriever(query, collection=COLLECTION_NAME, connection=CONNECTION_STRING, embedding=embeddings):
  db = vector_db_retrieve(collection, connection, embedding)
  llm = ChatOllama(model="mistral:7b")
  retriever = db.as_retriever(
    search_kwargs={"k": 3} # 3 best responses
  )
  retrieve_answer = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
  )
  query = f"{query}"
  response = retrieve_answer.invoke(query, config={"callbacks": [langfuse_handler]})
  # return display(Markdown(response))
  print("RETRIEVAL RESPONSE Query: ", response["query"])
  print("RETRIEVAL RESPONSE Result: ", response["result"])
  return {
    "Question": query,
    "Response": response["result"],
  }

print(answer_retriever(f"{topic}"))
"""
"""
def chat_long_context_retriever():
  while True:
    completion = lmstudio_llm.chat.completions.create(
      model="TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/openhermes-2.5-mistral-7b.Q3_K_M.gguf",
      messages=history,
      temperature=0.2,
      stream=False,
    )

    new_message = {"role": "assistant", "content": ""}
    print("completion:", completion)
    print(completion.choice[0].message.content)
    new_message["content"] += completion.choice[0].message.content

    history.append(new_message)
    
    # Uncomment to see chat history
    # import json
    # gray_color = "\033[90m"
    # reset_color = "\033[0m"
    # print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
    # print(json.dumps(history, indent=2))
    # print(f"\n{'-'*55}\n{reset_color}")

    print()
    history.append({"role": "user", "content": input("> ")})
    for elem in history:
      if elem["content"] == "stop now":
        print("The application will stop now..... see yaaaaaaaaaaaaaaa!")
        break

chat_long_context_retriever()
"""
# can also control if want to capture input/output or not as default is 'True' @observe(capture_input=False, capture_output=False)
@observe(as_type="observation")
def chat_with_groq(**kwargs):
  # can also extract the value from the dictionary **kwargs and use those separately in this function or use them all if many just using '**kwargs' that will be unfolded
  kwargs_clone = kwargs.copy()
  print("Kwargs clone: ", kwargs_clone)
  user_question = kwargs_clone.pop('text', None)
  print("User question: ", user_question)
  langfuse_context.update_current_observation(
      input=user_question,
      metadata=kwargs_clone
  )
  system_prompt = "You are a helpful assistant."
  human_message = "{text}"
  prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_message)])
  # pass in the prompt template the variable to create the final prompt
  chain = prompt | groq_llm_mixtral_7b | StrOutputParser()
  # here we use the **kwargs unfolding technique but we could al<so use the variable 'user_question' and put it in a dictionary to use it like 'chain.invoke({"text": "user_question"})'
  response = chain.invoke({"text": "user_question"})
  #response = chain.invoke(**kwargs) # TypeError: RunnableSequence.invoke() got an unexpected keyword argument 'text'
  
  return {
    "response content": response.content,
    "reponse total time duration in/out": response.response_metadata["token_usage"]["total_time"],
    "response id": response.id
  }

#print(chat_with_groq("What is the capital city of Senegal?"))
#Outputs:
"""
content="The capital city of Senegal is Dakar. Dakar is located on the westernmost point of Africa and is the largest city in Senegal. It's known for its vibrant culture, rich history, and beautiful beaches. The city is also an important economic and political center in West Africa." 
response_metadata={
  'token_usage': {
    'completion_time': 0.113,
    'completion_tokens': 64,
    'prompt_time': 0.011,
    'prompt_tokens': 25,
    'queue_time': None,
    'total_time': 0.124,
    'total_tokens': 89
  }, 
  'model_name': 'mixtral-8x7b-32768',
  'system_fingerprint': 'fp_c5f20b5bb1',
  'finish_reason': 'stop',
  'logprobs': None
}
id='run-de4928db-3834-42c3-8244-bfd1292165ea-0'
"""

question = "What is the biggest city in Asia?"
@observe()
def main(question):
  return chat_with_groq(text=question) # dictionary to be passed in

print(main(question))





