import streamlit as st
from htmlTemplates import css, bot_template, user_template

#from langchain.chains import ConversationChain
#from langchain.memory import ConversationBufferMemory
#from langchain.chat_models import ChatOpenAI
#from langchain_community.chains import ConversationChain
#from langchain_openai import ChatOpenAI
#from langchain_core.prompts import PromptTemplate
#from langchain.memory import ConversationBufferMemory

#from langchain.chains import ConversationChain
#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
#from langchain_groq import ChatGroq
#from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
import tiktoken
from langchain_huggingface import HuggingFaceEmbeddings
##from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

load_dotenv(dotenv_path='.env')
open_api_token = os.getenv("OPENAI_API_TOKEN")

def ler_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        all_text = file.read()
    return all_text
    
tokenizer = tiktoken.get_encoding("cl100k_base")
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def load_doc():
    #text1 = ler_txt('sinonimias.txt')
    text2 = ler_txt('vocabulario_controlado.txt')

    ##text_splitter = RecursiveCharacterTextSplitter(  
    ##    chunk_size=512,
    ##    chunk_overlap=24,
    ##    length_function=count_tokens,
    ##    separators=["#"] 
    ##)
    
    text_splitter = text2.split("#")
    chunks2 = [Document(page_content=chunk, metadata={"source": "vocabulario_controlado.txt", "row": i})
              for i, chunk in enumerate(text_splitter)]
    
    #chunks1 = []
    ##chunks2 = []
    #metadata = {"source": 'sinonimias.txt', "row": 0}
    #chunks1 = text_splitter.create_documents([text1], metadatas=[metadata])
    ##metadata = {"source": 'vocabulario_controlado.txt', "row": 0}
    ##chunks2 = text_splitter.create_documents([text2], metadatas=[metadata])
    combined_chunks = chunks2

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(combined_chunks, embeddings)

    # Persist the vectors locally on disk
    vectorstore.save_local("faiss_index_datamodel_law")

    # Load from local storage
    persisted_vectorstore = FAISS.load_local("faiss_index_datamodel_law", embeddings,
                                             allow_dangerous_deserialization=True)
    return persisted_vectorstore

def load_model():
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Modelo usado
        temperature=0,        # Controle de criatividade das respostas
        openai_api_key=open_api_token  # Substitua pela sua chave de API
    )
    system_instruction = """ 
    Você é um assistente virtual que busca responder perguntas dos usuários. Responda as perguntas de acordo com as notas de definição dos termos disponíveis no contexto. Se a você for solicitado as referências bibliográficas das definições, responda de acordo com a garantia literária que é informada no contexto. Nele estão disponíveis também sinônimos de alguns termos, assuma que esses termos que possuem sinônimos, têm as mesmas definições e referências bibliográficas, portanto, entregue respostas sem ambiguidades. Somente entregue respostas que constem nesse contexto.
    """
    template = """
    Pergunta: {input}
    Histórico da conversa: {history}
    Contexto: {context}
    """
    concatenated_template = system_instruction + "\n" + template
    
    prompt = ChatPromptTemplate.from_template(template=concatenated_template)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    return chain

def ask_pdf(user_input, history):
    #st.markdown("Iniciando...")
    db = load_doc()
    #st.markdown("Dados carregados no Vector store...")
    # similar_response = db.similarity_search(query,k=3)
    similar_response = db.similarity_search_with_score(user_input, k=1)
    st.markdown("Teste de similaridade concluído...")
    # Exibindo os resultados com suas pontuações
    docs = []
    pontuacoes = []
    contexto = ''
    for doc, score in similar_response:
        docs.append(doc)
        pontuacoes.append(score)
        st.markdown(f"Documento: {doc}")
        st.markdown(f"Pontuação: {score}")
        contexto = contexto + doc.page_content

    ##similar_response = clean_references(docs, pontuacoes)
    similar_response = ''
    
    context = similar_response
    # context = [doc.page_content + doc.metadata['source'] for doc in similar_response]
    # print(context)

    chain = load_model()
    #st.markdown(contexto)

    # Execute the chain and get the result
    result = chain.invoke({
        "context": contexto,
        "input": user_input,
        "history": history
    })
    # chat_history.append((query, result.content))
    #print(result)

    return result


def main():
    st.set_page_config(page_title="ChatbotPI",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Bem-vindo ao ChatbotPI!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibir histórico de mensagens
    ##for message in st.session_state.messages:
    ##    if message['role'] == "user":
    ##        with st.chat_message("user"):
    ##            st.markdown(message['content'])
    ##    else:
    ##        with st.chat_message("assistant"):
    ##            #st.markdown(message['content']['response'])
    ##            st.markdown(message['content'])
                
    for i, message in enumerate(st.session_state.messages):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message['content']), unsafe_allow_html=True)


    if "exit" not in st.session_state:
        st.session_state.exit = False

    # Verifica se o chatbot foi encerrado
    if st.session_state.exit:
        st.write("Chatbot: Até mais!")
        return
        
    chain = load_model()
    
    user_input = st.chat_input("Você: ",key='input1')
    if user_input is not None and user_input != '':
        user_question = {"role": "user", "content": user_input}
        st.session_state.messages.append(user_question)
        st.write(user_template.replace("{{MSG}}", user_input), unsafe_allow_html=True)
    
        if user_input.lower() == "sair": 
            st.session_state.exit = True

        context = ''
        ##response = chain.invoke({"input":user_input,"history":st.session_state.messages,"context":context})
        response = ask_pdf(user_input, st.session_state.messages)
        chatbot_response = {"role": "assistant", "content": response}
        st.session_state.messages.append(chatbot_response)
        st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
