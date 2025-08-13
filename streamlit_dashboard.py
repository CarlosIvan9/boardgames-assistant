

"""
# My first app
Here's our first attempt at using data to create a table:
"""

# In python (both options work):
#streamlit run your_script.py [-- script args] --server.port 8501 #change port if needed
#python -m streamlit run your_script.py



#########################
### Loading libraries ###
#########################

# Add api/utils/ to path, to load package functions
import sys
# Useful to set working directory
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "api" / "utils"))  #For scripts
#sys.path.append(str(Path().resolve() / "api" / "utils"))  #For notebooks

# Load rag graph
from rag import graph

# To feed the graph
from langchain_core.messages import HumanMessage

# For UI
import streamlit as st

# Create id of thread (Memory of the chatbot will be reset every different day)
from datetime import datetime

import logging
logging.basicConfig(level=logging.DEBUG)




#####################
### Streamlit app ###
#####################

st.markdown('# Boardgames Assistant')
st.markdown(
    "<span style='color:red;'>**Note:**</span> <span style='color:black;'>more than 10 questions per minute will trigger an error</span>.",
    unsafe_allow_html=True
)
user_name = st.text_input("Whats your name?", key="name")

# Specify an ID for the thread
config = {"configurable": {"thread_id": user_name}}

if user_name:
    input_message = st.text_input(f'### What is your boardgame and your question {user_name} ?')

    if input_message:

        an_input_state={'messages':[HumanMessage(content=input_message)]}

        answer = graph.invoke( an_input_state , config=config)
        
        # Format output to be able to be returned to user
        response = answer['messages'][-1].content

        st.markdown(response)


