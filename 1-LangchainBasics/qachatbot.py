import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import os

##page config
st.set_page_config(page_title="Simple Langchain Chatbot with Groq", page_icon="t&T")

#title
st.title("t&T Simple Langchain Chat with Groq")
st.markdown("Learn Langchain basics with Groq's ultra-fast inference")


with st.sidebar:
    st.header("Settings")

    ##API KEY
    api_key=st.text_input("GROQ_API_KEY", type="password", help="GET Free API Key at console.groq.com")

    #MOFDEL SELECTION

model_name= st.selectbox(
    "Model",
    ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
    index=0
)

#clear button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


#initialize LLM
@st.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        return None

    #initialize the GROQ model
    llm = ChatGroq(groq_api_key=api_key,
             model_name=model_name,
             temperature=0.7,
             streaming=True)


    #create PromptTemplkate
    prompt= ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assitant powered by Groq. Answer questions clearly and concisely"),
        ("user", "{question}")
    ])


    #create chain
    chain= prompt | llm | StrOutputParser()

    return chain

#get chain
chain=get_chain(api_key, model_name)

if not chain:
    st.warning("^ Please enter your Groq API key in the sidebar to start chatting!")
    st.markdown("[Get your free API key here](https://console.groq.com)")

else:
    #display the chat message

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    #chat input
    if question:= st.chat_input("Ask me anything"):
        #add user message to session state
        st.session_state.messages.append({"role":"user", "content":question})
        with st.chat_message("user"):
            st.write(question)

        #generate response
        with st.chat_message("assistant"):
            message_placeholder= st.empty()
            full_response=""

            try:
                #stream response from Groq
                for chunk in chain.stream({"question": question}):
                    full_response+= chunk
                    message_placeholder.markdown(full_response + "▌")
            
                message_placeholder.markdown(full_response)    

                #add to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            except Exception as e:
                st.error(f"Error: {str(e)}")


#Examples
st.markdown("---")
st.markdown("### Try these examples:")
col1, col2 =  st.columns(2)
with col1:
    st.markdown("- What is LangChain?")
    st.markdown("- Explain Groq's LPU technology")
with col2:
    st.markdown("- How do I learn programming?")
    st.markdown("- Write a haiku about AI")

#Footer
st.markdown("---")
st.markdown("Built with LangChain &Groq | Explain the speed!")    