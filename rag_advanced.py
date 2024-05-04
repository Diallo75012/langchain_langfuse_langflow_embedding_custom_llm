
### Step 1: Convert PDF or Text to Markdown
import pdfplumber
import markdownify

def pdf_to_markdown(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages])
    return markdownify.markdownify(text, heading_style="ATX")

### Step 2: Chunk the Markdown Document
from langchain.text_splitter import MarkdownHeaderTextSplitter

def chunk_markdown(markdown_text):
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=["#", "##", "###"],
        chunk_size=1024,
        chunk_overlap=0
    )
    chunks = header_splitter.create_documents([markdown_text])
    return chunks

### Step 3: Generate Metadata Using LLM
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

def generate_metadata(chunk, llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", f"Summarize the meaning of this text in one sentence: {chunk}"),
            ("human", f"Create three potential questions that can be answered by this text: {chunk}")
        ]
    )
    response = llm(prompt)
    summary, questions = response[:response.index("\n")], response[response.index("\n")+1:]
    questions = questions.split("\n")[:3]
    metadata = {
        'content_meaning': summary,
        'potential_answered_questions': questions
    }
    return metadata

# alternative requesting special format 
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

def generate_metadata(chunk, llm):
  prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", f"Please provide a summary and three potential questions for this text. Format the response as follows:
   \n\n<summary>\n<question1>\n<question2>\n<question3>\n\nText: {chunk}")
    ]
  )
  response_lines = response.splitlines()
  summary = response_lines[0]
  questions = response_lines[1:4]
  metadata = {
    'content_meaning': summary,
    'potential_answered_questions': questions
  }
  return metadata

def create_chunks_with_metadata(chunks, llm, document_name):
    result = []
    for chunk in chunks:
        metadata = generate_metadata(chunk.page_content, llm)
        metadata['document_name'] = document_name
        chunk.metadata = metadata
        result.append(chunk)
    return result

### Step 4: Embed the Chunks (Change FAISS for PGVector)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def embed_chunks(chunks):
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# alternative PGVector
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import OpenAIEmbeddings
from sqlalchemy import create_engine

def embed_chunks_pgvector(chunks):
    embeddings = OpenAIEmbeddings()
    connection_string = "postgresql://username:password@localhost:5432/dbname"
    engine = create_engine(connection_string)
    vector_db = PGVector.from_documents(chunks, embeddings, engine)
    return vector_db


### Step 5: Perform Retrieval
def retrieve_chunks(vector_db, user_query):
    results = vector_db.similarity_search(user_query, k=5)
    return results

# Step 6: Judge Relevance Using LLM
def judge_relevance(llm, user_query, retrieved_chunks):
    relevant_chunks = []
    for chunk in retrieved_chunks:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("human", f"User query: {user_query}\n\nRetrieved chunk: {chunk.page_content}\n\nIs this chunk relevant to the user query?")
            ]
        )
        response = llm(prompt)
        if "yes" in response.lower():
            relevant_chunks.append(chunk)
    return relevant_chunks

### Step 7: Construct the Final Answer
def construct_final_answer(llm, user_query, relevant_chunks):
    combined_text = "\n".join([chunk.page_content for chunk in relevant_chunks])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", f"User query: {user_query}\n\nRelevant chunks: {combined_text}\n\nAnswer the user's query using the relevant information.")
        ]
    )
    response = llm(prompt)
    return response

### Full workflow
def process_document(file_path, user_query):
    # Step 1: Convert to Markdown
    markdown_text = pdf_to_markdown(file_path)
    
    # Step 2: Chunk the Document
    chunks = chunk_markdown(markdown_text)
    
    # Step 3: Generate Metadata
    llm = ChatOpenAI(temperature=0)
    chunks_with_metadata = create_chunks_with_metadata(chunks, llm, file_path)
    
    # Step 4: Embed the Chunks
    vector_db = embed_chunks(chunks_with_metadata)
    
    # Step 5: Perform Retrieval
    retrieved_chunks = retrieve_chunks(vector_db, user_query)
    
    # Step 6: Judge Relevance
    relevant_chunks = judge_relevance(llm, user_query, retrieved_chunks)
    
    # Step 7: Construct Final Answer
    final_answer = construct_final_answer(llm, user_query, relevant_chunks)
    
    return final_answer


