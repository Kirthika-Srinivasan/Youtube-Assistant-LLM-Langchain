# YouTube-Assistant-LLM-Langchain

A simple YouTube video assistant built using LangChain that answers user questions based on a video’s transcript.

## Tech Stack

- Python
- LangChain
- OpenAI API
- Streamlit
- YouTube Transcript API
- FAISS (vector store)

## Steps Followed

1. Clone the repository
  ```
git clone https://github.com/Kirthika-Srinivasan/Youtube-Assistant-LLM-Langchain
cd Youtube-Assistant-LLM-Langchain
  ```

2. Create and activate a virtual environment
  ```
python -m venv venv
.venv/Scripts/Activate.ps1 
  ```

3. Install the required dependencies
  ```
pip install -r requirements.txt
  ```

4. Create a .env file and add the OpenAI API key
  ```
OPENAI_API_KEY=your_api_key_here
  ```

5. Load a YouTube video transcript using the YouTube transcript loader

6. Split the transcript into smaller text chunks

7. Generate embeddings for each chunk using an embedding model

8. Store the embeddings in a FAISS vector database

9. Build a retrieval-based QA pipeline using LangChain to fetch relevant chunks for a user query

10. Pass the retrieved context to the LLM to generate answers

11. Build a simple user interface using Streamlit to:
  - accept a YouTube video URL
  - accept a user question
  - display the generated answer

12. Run the application
    ```
    streamlit run main.py
    ```

## Streamlit App screenshot

Input video - https://www.youtube.com/watch?v=-Osca2Zax4Y

<img width="1920" height="883" alt="image" src="https://github.com/user-attachments/assets/5d497fc9-da6c-4c7e-8914-5bac3a45ae77" />
