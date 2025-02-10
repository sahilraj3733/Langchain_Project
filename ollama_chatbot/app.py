import streamlit as st
import openai
# from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="SIMPLE Q&A Ollama"

#PROMPT TEMPLATE
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful massistant . Please  repsonse to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,engine,temperature,max_tokens):
    llm=Ollama(model=engine)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({"question":question})
    return answer

## Title of the app
st.title("Enhanced Q&A chatbot with Ollama")

# siderbar
st.sidebar.title("settings")

# Select the openAI model
engine=st.sidebar.selectbox("Select open AI model",["gemma:2b"])

## Adjust response Parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=.7)
max_tokens=st.sidebar.slider("Max_Tokens",min_value=50,max_value=300,value=150)

## main interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input :
    response=generate_response(user_input,engine,temperature,max_tokens)
    st.write(response)

else:
    st.write("plase provide the user input")