import streamlit as st
import os
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

# Load Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=groq_api_key,model="Gemma2-9b-It") 

# Transcription loader
def transcript_load(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        if not transcript_list:
            st.error("Transcript is empty or unavailable.")
            return None
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        st.error("No captions available for this video.")
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

# Create vector store from transcript
def create_vector_embedding(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.create_documents([transcript])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vector_store

def combine_doc(docs):
    context_text='\n\n'.join(doc.page_content for doc in docs)
    return context_text


# Prompt template
prompt_template = PromptTemplate(
    template="""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {question}
""",
    input_variables=['context', 'question']
)

# Streamlit UI
st.title("🎥 YouTube Video Chatbot")
video_id = st.text_input("Enter YouTube video ID:").strip()

# Load transcript/vector store only once per video
if video_id:
    if "current_video" not in st.session_state or st.session_state.current_video != video_id:
        with st.spinner("Fetching transcript and creating knowledge base..."):
            transcript = transcript_load(video_id)
            if transcript:
                vector_store = create_vector_embedding(transcript)
                st.session_state.current_video = video_id
                st.session_state.transcript = transcript
                st.session_state.vector_store = vector_store
            else:
                st.stop()
    else:
        transcript = st.session_state.transcript
        vector_store = st.session_state.vector_store

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Set up QA chain
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(combine_doc),
        'question': RunnablePassthrough()
    })
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt_template | llm | parser

    user_question = st.text_input("Ask a question about the video:")

    if user_question:
        with st.spinner("Generating answer..."):
            retrieved_docs = retriever.invoke(user_question)
            result = main_chain.invoke(user_question)

            st.markdown("### 💬 Answer:")
            st.write(result)

            with st.expander("🔍 Relevant Context from Transcript (click to view)"):
                for i, doc in enumerate(retrieved_docs):
                    st.write(f"**Chunk {i + 1}:**")
                    st.write(doc.page_content)
                    st.write("---")
