import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

# loading unstructured files of all sorts as LangChain Documents
def load_document(file):
    from langchain.document_loaders import UnstructuredFileLoader
    loader = UnstructuredFileLoader(file)
    data = loader.load()
    return data

# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-4o', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer

# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004

def check_openai_api_key_exist():
    if 'OPENAI_API_KEY' not in os.environ:
        st.error('Please provide your OpenAI API key in the sidebar.')
        st.stop()

def is_api_key_valid(api_key):
    import openai
    openai.api_key = api_key
    try:
        openai.Model.list()
    except openai.error.AuthenticationError as e:
        return False
    else:
        return True

def clear_text_input():
    st.session_state.text_input = ''

def start_over_with_new_document():
    st.session_state.text_input = ''
    # delete the vector store from the session state
    del st.session_state.vs
    # display message to user
    st.info('Please upload new documents to continue after clearing or updating the current ones.')

def create_linkedin_post(answer):
    # This function creates a simple LinkedIn post from the LLM's answer
    max_length = 1300  # LinkedIn's character limit
    post = f"Check out this interesting insight I learned today:\n\n{answer[:max_length]}..."
    if len(answer) > max_length:
        post += "\n\n(Post truncated due to LinkedIn's character limit)"
    return post

def copy_to_clipboard(text):
    st.session_state.clipboard = text

if __name__ == "__main__":
    # two images in sidebar next to each other
    col1, col2 = st.sidebar.columns(2)
    col1.image('images/OpenAI_logo.png')
    col2.image('images/langchain-chroma-light.png')

    st.header('LLM Question-Answering Application')
    with st.sidebar:
        # text_input for the OpenAI API key
        api_key = st.text_input('Your OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        # check if the API key is not valid
        if api_key and not is_api_key_valid(api_key):
            st.error('Invalid OpenAI API key. Please provide a valid key.')
            st.stop()

        # file uploader widget
        uploaded_files = st.file_uploader('Upload any file format with text to analyze:', accept_multiple_files=True)

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=8192, value=512)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3)

        # initialize add_data
        add_data = False

        # add data button widget
        if is_api_key_valid(api_key):
            add_data = st.button('Add Data', key='add_data')
        else:
            st.info('No OpenAI API key. Please provide a valid key.')

        if uploaded_files and add_data: # if the user uploaded files and clicked the add data button
            check_openai_api_key_exist()
            with st.spinner('Reading, chunking and embedding data ...'):

                # create ./docs/ folder if it doesn't exist
                if not os.path.exists('./docs/'):
                    os.mkdir('./docs/')

                # list to store all the chunks
                all_chunks = []
                
                for uploaded_file in uploaded_files:

                    # writing the file from RAM to the current directory on disk
                    bytes_data = uploaded_file.read()
                    file_name = os.path.join('./docs/', uploaded_file.name)
                    with open(file_name, 'wb') as f:
                        f.write(bytes_data)

                    data = load_document(file_name)
                    chunks = chunk_data(data, chunk_size=chunk_size)
                    st.write(f'File name: {os.path.basename(file_name)}, Chunk size: {chunk_size}, Chunks: {len(chunks)}')
                    all_chunks.extend(chunks)

                tokens, embedding_cost = calculate_embedding_cost(all_chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(all_chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('Uploaded, chunked and embedded successfully.')

                # deleting files from the docs folder after they have been chunked and embedded
                for file in os.listdir('./docs/'):
                    os.remove(os.path.join('./docs/', file))

                # deleting the docs folder
                os.rmdir('./docs/')

    if uploaded_files and 'vs' in st.session_state:
        # user's question text input widget
        q = st.text_input('Ask one or more questions about the content of the uploaded data:', key='text_input')
        if q: # if the user entered a question and hit enter
            if 'vs' in st.session_state: # if vector store exists in the session state
                vector_store = st.session_state.vs
                answer = ask_and_get_answer(vector_store, q, k)

                # text area widget for the LLM answer with flexible height
                st.text_area('LLM Answer: ', value=answer, height=200)

                # Add LinkedIn post creation button
                if st.button('Create LinkedIn Post'):
                    linkedin_post = create_linkedin_post(answer)
                    st.text_area('LinkedIn Post:', value=linkedin_post, height=150)
                    st.button('Copy to Clipboard', on_click=copy_to_clipboard, args=(linkedin_post,))
                    
                    if 'clipboard' in st.session_state:
                        st.success('Post copied to clipboard!')

        # Button for new question, on click clear text input
        if st.session_state.text_input:
            st.button('New question for same context', on_click=clear_text_input, key='new_question_same_context')
            st.button('New question for new context', on_click=start_over_with_new_document, key='new_question_new_context')

    else:
        st.info('Please upload one or more files to continue.')