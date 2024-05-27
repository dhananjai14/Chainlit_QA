import os
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI

from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

openai_api_key = os.getenv('openai_api_key')
os.environ['openai_api_key'] = openai_api_key

# Whenever RAG is being created, the first thing req is data.
# Step 1: Data need to be uploded and then chunks are created.
# Step 2: Embeddings are created. 

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)

# 'async' is used to run the concurrency ie at same time it is required to run the multiple thread, then async is used.
# 'async' is used for multiprocessing 

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for used to upload the files
    while files == None:
        files = await cl.AskFileMessage(
            content= "please upload the file"
            , accept=['text/plain']
            , max_size_mb=20
            , timeout=180
        ).send()

    file = files[0]

    # Open the file 
    msg = cl.Message(content=f"Processing {file.name} ...", disable_feedback= True)
    await msg.send()

    with open(file.path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # Text splitting into chunks 
    texts = text_splitter.split_text(text)

    # creating metadata for each chunk 
    metadatas = [{'source': f'{i}-pl'} for i in range(len(texts))]
    # Creating embedding 
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embedding=embeddings, metadatas=metadatas
    )
    
    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key= 'chat_history'
        , output_key='answer'
        , chat_memory=message_history
        , return_messages=True
    )


    # Creating the chain

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature = 0, streaming = True)
        , chain_type = 'stuff'
        , retriever=docsearch.as_retriever()
        , memory=memory
        , return_source_documents=True
    ) 

    # Let the user know that the system is ready. User can now ask question
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


# Now user will ask the querry 

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler() 
    
    # Result is generated 
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"] 
    
    
    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
    





