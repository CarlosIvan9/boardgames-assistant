

possible_games = ['axis-allies-rules-1942-2nd-edition',
 'Axis_&_Allies_1942_2nd_Reference_v1',
 'Grand_Austria_Hotel_game_aid_-_2_pages_v1',
 'Grand_Austria_Hotel_Plain_and_Simple_Guide',
 'Grand_Hotel_Austria_rules_EN_web',
 'In_The_Footsteps_Of_Marco_Polo_-_English_Overview_Cards',
 'LlamaLand_Player_Aid_(English)_version_2',
 'LlamaLand_Rules_EN',
 'Memoir_44_-_a_beginners_reference',
 'memoir_44_rules_part1_en',
 'Memoir_44_Unofficial_FAQ_v12',
 'munchkindisneyrules',
 'MunchkinDisney_Instr',
 'MYSTERIUM_PARK_RULES_EN_BD',
 'Mysterium_Park_Rules_Summary_unnofficial',
 'Paleo_FAQ_11-12-20_ENG',
 'Paleo_Rulebook_EN_compressed',
 'pandemic season 1 rules',
 'partners',
 'Partners_rules',
 'Power Grid Expansion Benelux Central Europe',
 'Power Grid Expansion India Australia',
 'Power Grid Expansion UK Ireland Northern Europe',
 'Power-Grid-Recharged-Rules',
 'skull_king_rulebook_optimized',
 'splendor',
 'Splendor_Brief_by_Liumas_2014-05b',
 'Take-5-or-6-Nimmt-Rules',
 'Take_5_6nimmt30_rules_english',
 'The_Crew_Demo_Sheet',
 'The_Crew_Manual',
 'Tureluurs official rules EN',
 'Tureluurs official rules',
 'Tureluurs unofficial rules',
 'Voyages Of Marco Polo - English Rules V3',
 'Voyages_Of_Marco_Polo_-_FAQ_and_Errata_V2']

print('We started running rag script')



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

# For filtering a vector store
from qdrant_client import models

# For declaring functions as tools to be used by the generative model
from langchain_core.tools import tool

#For declaring custom graph states
from typing_extensions import  TypedDict, List, Annotated
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# For declaring nodes
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.graph import MessagesState, START
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

    

logger.debug('We started running rag script')


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
#embeddings = CohereEmbeddings(model="embed-v4.0") # Langchain wrapper does not allow reducing size to 256
embeddings = CohereEmbeddings(model="embed-multilingual-light-v3.0")  # This dimension is 384. Default of v4.0: 1536
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

######################


# Approach via Chains
#Our graph will consist of four nodes:

    #1- A node that extracts the possible document names that could help with the answer
    #2- A node for the retriever tool that executes the retrieval step
    #3- A node that generates the final response using the retrieved context.

class StateWithDocsList(TypedDict): 
    messages: Annotated[Sequence[BaseMessage], add_messages] #add_messages appends new messages in the state automatically
    list_useful_docs: List[str]
    concatenated_retrieved_context: str


# Node 1: Extract possible document names.
def get_useful_document_names(state: StateWithDocsList):
    """Finds subset of useful boardgame document names to be used in retrieval"""

    prompt = (
        "You are an assistant that tries to find which boardgame booklets are related to a particular sentence. "
        "First detect the name of the boardgame the user is asking about."
        "Then select from the list of documents the ones that could be related to that boardgame."
        "Then ONLY return the selected documents as a comma-separated list. Nothing else."
        "If you are not sure about including a document, select it just in case"
        "Do not select more than 5 names."
        "If none are relevant, return 'None'. "
        "The list of boardgame booklets is the following: "
        f"{possible_games}"

        "The sentence of the user is:"
        f"{state['messages'][-1].content}"
    )

    response = model.invoke(prompt)
    str_list_useful_docs = response.content #Responses are of type AIMessage

    # Convert string of lists of documents, into a list of strings of documents: 'doc1, doc2' -> ['doc1','doc2']
    list_useful_docs=str_list_useful_docs.split(',')
    list_useful_docs=[astring.strip() for astring in list_useful_docs]
    
    state['list_useful_docs']= list_useful_docs

    return state


# Node 2: Retrieve related content from docs.
def retriever(state: StateWithDocsList): # This receives a string as input and not a state bc it is not a node. This tool will be added into a node, that needs as input a State
    """Retrieve information related to a query."""
    nr_vector_store_documents=client.count(collection_name=vector_store.collection_name).count  # Works for Qdrant vector_store
    #print(nr_vector_store_documents)
    logger.debug(f'Size of vector store until now: {nr_vector_store_documents}')

    #Get last question of the human
    all_human_messages = [amessage for amessage in state['messages'] if amessage.type=='human']
    last_human_message = all_human_messages[-1]

    filter_docs=models.Filter( #Filter to be used in the similarity search
        should=[
            models.FieldCondition(
                key="metadata.game",
                match=models.MatchAny(any=state['list_useful_docs'])    
            ),
        ]
    )
    logger.debug(f"Starts vector similarity search")
    # This is were the app crashes due to lack of memory
    retrieved_docs = vector_store.similarity_search(
        last_human_message.content, 
        k= 8, #4,
        filter=filter_docs #This is what filters to only documents with specific metadata
        ) # Tried this to make it lighter but never worked(query, k=1, search_params={"ef": 4})
    logger.debug("Vector similarity done")
    concat_context = "\n\n".join(  #The concatenation of docs as strings was done in the generate function before, and it did not include metadata
        (f"Content: {doc.page_content}") 
        for doc in retrieved_docs 
    )
    state['concatenated_retrieved_context'] = concat_context
    logger.debug("End of retrieval tool")

    return state


# Node 3: Generate a response using the retrieved content. This will only be run if we retrieved data from the vector_store
def generate(state: StateWithDocsList): #This state contains the original query as a human message + the context from the retriever as an ai message
    """Generate answer."""
    # Get generated ToolMessages
    logger.debug("Start of node generate")


    #maybe try using tools and see if it works or not?
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know, don't try to make up an answer."
        "Use between three to 5 sentences maximum and keep the answer concise."
        "\n\n"
        f"{state['concatenated_retrieved_context']}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls) #excludes ai messages from the retriever (I think not doing this will output 1 extra message per doc extracted)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages #This is done to let know the chatbot of the question
    logger.debug(f'Prompt for generate function: {prompt}' )

    # Run
    response = model.invoke(prompt)
    logger.debug("End of generate node")

    state['messages'] = state['messages'] + [response] 

    return state



### Declaration of the graph ###


#graph_builder = StateGraph(MessagesState) #Input of the graph is of the type MessagesState
graph_builder=StateGraph(state_schema= StateWithDocsList)

#declaration of nodes
graph_builder.add_node(get_useful_document_names)
graph_builder.add_node(retriever)
graph_builder.add_node(generate)

#set starting nodes
graph_builder.add_edge(START, "get_useful_document_names")
graph_builder.add_edge("get_useful_document_names", "retriever")
graph_builder.add_edge("retriever", "generate")
graph_builder.add_edge("generate", END)


#declaration of memory
memory = MemorySaver()
#graph compilation
graph = graph_builder.compile(checkpointer=memory)


print('We finished running rag script')


