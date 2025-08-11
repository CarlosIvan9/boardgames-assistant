


# Useful to set working directory
from pathlib import Path

# Useful to load api keys for Cohere embeddings and generative model
import os
from dotenv import load_dotenv
load_dotenv()  # Loads from .env
COHERE_API_KEY=os.getenv("COHERE_TOKEN")
os.environ["COHERE_API_KEY"]=COHERE_API_KEY 
from langchain_cohere import CohereEmbeddings
from langchain.chat_models import init_chat_model

# For connecting to the embeddings vector store
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# For declaring functions as tools to be used by the generative model
from langchain_core.tools import tool

# For declaring nodes
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

#For creating the graph
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph import END
from langgraph.prebuilt import  tools_condition

#For adding memory to the graph
from langgraph.checkpoint.memory import MemorySaver

#For logging
import logging

logger = logging.getLogger(__name__)

    

print('We started running rag script')


######################


# Set working directory
try:
    # Works in regular Python scripts
    base_dir = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Fallback for Jupyter notebooks and interactive shells
    base_dir = Path().resolve().parent.parent


########################


# Declare both embeddings and generative models
embeddings = CohereEmbeddings(model="embed-v4.0")
model = init_chat_model("command-a-03-2025", model_provider="cohere") # Cohere api key needs to be as env variable with the name COHERE_API_KEY



########################



# Connect to Qdrant vector store
path_qdrant_vector_store = base_dir / "data" /  "qdrant"
client = QdrantClient(path=path_qdrant_vector_store)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="document_embeddings",
    embedding=embeddings,
)


######################


# Approach via Chains
#Our graph will consist of three nodes:

    #1- A node that fields the user input, either generating a query for the retriever or responding directly; (this is cool and we did not have it before)
    #2- A node for the retriever tool that executes the retrieval step
    #3- A node that generates the final response using the retrieved context.


### Tools used in the nodes of the graph: ###

#Tool to retrieve data from embeddings store
#An agent decides which tool to use based on the tool's names, descriptions, and argument schemas
@tool(response_format="content_and_artifact") # This will make it possible for a chat model to call this function
def retrieve(query: str): # This receives a string as input and not a state bc it is not a node. This tool will be added into a node, that needs as input a State
    """Retrieve information related to a query."""
    logger.debug("Start of retrieval tool")
    nr_vector_store_documents=client.count(collection_name=vector_store.collection_name).count  # Works for Qdrant vector_store
    logger.debug(f'Size of vector store until now: {nr_vector_store_documents}')
    logger.debug(f"Starts vector similarity search")
    retrieved_docs = vector_store.similarity_search(query, k=1, search_params={"ef": 4})
    logger.debug("Vector similarity done")
    serialized = "\n\n".join(  #The concatenation of docs as strings was done in the generate function before, and it did not include metadata
        (f"Content: {doc.page_content}") 
        for doc in retrieved_docs 
    )
    logger.debug("End of retrieval tool")
    return serialized, retrieved_docs  # if you use @tool(response_format="content_and_artifact"), you must return exactly a tuple of two elements
#First element → This is what will be used as the main textual output (like a summary or answer).
#Second element → This is any additional data (e.g. a document, list of sources, JSON, etc.) that might be useful for tracking, inspection, or chaining.


### Nodes used in the graph: ###

# Node 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    logger.debug("Start of query_or_respond node")
    llm_with_tools = model.bind_tools([retrieve]) # Tells the chat model it can use the retrieve tool
    response = llm_with_tools.invoke(state["messages"]) 
    # "response" is an AIMessage object with the content of the message and some metadata
    # This metadata also says if you must call a Tool or not, and which tool to call
    # MessagesState appends messages to state instead of overwriting
    output = {"messages": [response]}
    logger.debug("End of query_or_respond node")
    logger.debug(f"Output of query_or_respond node: {output}")
    return output

# Node 2: Execute the retrieval.
#ToolNode executes the tool and adds the result as a ToolMessage to the state. (This is important info)
#I think this addition only happens when running it inside a graph for a particular state, not if you run it independently
tools = ToolNode([retrieve])

# Node 3: Generate a response using the retrieved content. This will only be run if we retrieved data from the vector_store
def generate(state: MessagesState): #This state contains the original query as a human message + the context from the retriever as an ai message
    """Generate answer."""
    # Get generated ToolMessages
    logger.debug("Start of node generate")
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        #print(message.content) #Useful to see what is appended in the message
        if message.type == "tool":
            recent_tool_messages.append(message.content) #appends only tool messages'content (not metadata!) (outputs from the retriever)
        else:
            break
    tool_messages = recent_tool_messages[::-1] #orders them from first to last 

    # Format into prompt
    docs_content = "\n\n".join(doc for doc in tool_messages) # I dont see the need of this if we already joined contexts in tools. 
    #maybe try using tools and see if it works or not?
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know, don't try to make up an answer."
        "Use between three to 5 sentences maximum and keep the answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls) #excludes ai messages from the retriever (outputs of the retriever are already added in system_message_content)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    logger.debug(f'Prompt for generate function: {prompt}' )

    # Run
    response = model.invoke(prompt)
    logger.debug("End of generate node")
    return {"messages": [response]} # MessagesState appends messages to state instead of overwriting



### Declaration of the graph ###


graph_builder = StateGraph(MessagesState) #Input of the graph is of the type MessagesState
#declaration of nodes
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)
#set starting node
graph_builder.set_entry_point("query_or_respond") 
#set conditional node
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
#link rest of nodes
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
#declaration of memory
memory = MemorySaver()
#graph compilation
graph = graph_builder.compile(checkpointer=memory)


print('We finished running rag script')



