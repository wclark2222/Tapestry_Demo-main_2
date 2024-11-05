import traceback
import requests
import streamlit as st
import os
import warnings
from crewai import Agent, Task, Crew, Process

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit UI
st.title("Research Article Generator")

# File uploader
uploaded_file = st.file_uploader("Upload your transcript file", type="txt")
st.write(uploaded_file)

# API Key input
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Button to start processing
if st.button("Generate Research Article"):
    if not uploaded_file:
        st.error("Please upload a transcript file.")
    elif not openai_api_key:
        st.error("Please enter your OpenAI API Key.")
    else:
        # Set up environment variables
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'

        # Read the uploaded file
        transcripts = uploaded_file.read().decode("utf-8")

        try:
            # Test API connection
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {openai_api_key}"}
            )
            response.raise_for_status()
            st.success("API connection successful!")

            # Define agents
            planner = Agent(
                role="Content Planner",
                goal="Plan engaging and factually accurate content on the given topic",
                backstory="You're working on planning a research report about a given topic. You collect information that helps the audience learn something and make informed decisions. Your work is the basis for the Content Writer to write an article on this topic.",
                allow_delegation=False,
                verbose=True
            )

            writer = Agent(
                role="Content Writer",
                goal="Write insightful and factually accurate research report about the given topic",
                backstory="You're working on writing a new opinion piece about a given topic. You base your writing on the work of the Content Planner, who provides an outline and relevant context about the topic. You follow the main objectives and direction of the outline, as provide by the Content Planner. You also provide objective and impartial insights and back them up with information and quotes from the participants. Include factual data and numbers wherever possible. Insert key anonymized quotes wherever possible. You acknowledge in your opinion piece when your statements are opinions as opposed to objective statements. Ensure you are Analytical and Insightful through the use of Expert Opinions and Case Studies. Focus on Practical Recommendations and Emphasize on Current and Emerging Issues",
                allow_delegation=False,
                verbose=True
            )

            editor = Agent(
                role="Editor",
                goal="Edit a given blog post to align with the writing style of the organization.",
                backstory="You are an editor who receives a research article from the Content Writer. Your goal is to review the blog post to ensure that it follows journalistic best practices, provides balanced viewpoints when providing opinions or assertions, and also avoids major controversial topics or opinions when possible. Maintain a Formal and Professional Tone and ensure content has a Structured and Thematic Organization.",
                allow_delegation=False,
                verbose=True
            )

            # Define tasks
            plan = Task(
                description=f"Plan content for the topic: {transcripts}",
                agent=planner,
            )

            write = Task(
                description="Write a research article based on the content plan",
                agent=writer,
            )

            edit = Task(
                description="Edit and finalize the research article",
                agent=editor
            )

            # Create crew
            crew = Crew(
                agents=[planner, writer, editor],
                tasks=[plan, write, edit],
                verbose=True
            )

            # Process the transcript
            with st.spinner("Generating research article... This may take a few minutes."):
                result = crew.kickoff()

            # Display the result
            st.success("Research article generated successfully!")
            st.markdown(result)

        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            if hasattr(e, 'response'):
                st.error(f"Response Status Code: {e.response.status_code}")
                st.error(f"Response Content: {e.response.text}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")

st.markdown("---")
st.markdown("Tapestry Networks")