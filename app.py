from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from urllib.parse import urlparse, parse_qs
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("📺 Ask Questions About a YouTube Video ")

# --- Extract YouTube video ID from full URL ---
def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    elif parsed.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed.query).get("v", [None])[0]
    return None

# --- Cached function to process video transcript and embedding ---
@st.cache_resource(show_spinner="Processing video...")
def process_video(video_id: str):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(chunks, embeddings)

        return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    except Exception:
        return "error"

# --- UI inputs ---
video_url = st.text_input("Enter YouTube Video URL:", value="https://www.youtube.com/watch?v=Gfr50f6ZBvo")
question = st.text_area("Enter your question:", height=100)

video_id = extract_video_id(video_url)

if st.button("Submit") and video_id and question:
    retriever = process_video(video_id)
    with st.spinner("Processing your request..."):
        

        if retriever == "error":
            st.error("⚠️ Video should be in English or have English subtitles.")
        else:
            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables=['context', 'question']
            )

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",
                temperature=0.7
            )

            retrieved_docs = retriever.invoke(question)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            final_prompt = prompt.invoke({"context": context_text, "question": question})
            answer = llm.invoke(final_prompt)

            st.subheader("Answer:")
            st.write(answer.content)
