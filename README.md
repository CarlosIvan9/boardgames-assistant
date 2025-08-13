# boardgames-assistant
Assistant that explains rules about the boardgames I have at home.
The assistant is a RAG. We split the logic in 2 parts. The first part is in charge of uploading the documents to be used, chunks them and adds them to a Qdrant vector store.
The second part is the chatbot: it receives the question of the user, and retrieves an answer using as context the chunks from the vector store that are more similar to the question.
The first part is just a jupyter notebook called 'upload-files-to-qdrant-vector-store.ipynb'. The second part is a flask api which uses the script 'flask_api.py' and the rag module in the 'api/utils' folder.

Both branches work very well locally using flask, but fail to work on Render due to lack of memory in the free compute provided by them.

# Project Status
The project currently has three branches: first_rag, light_rag, and light_rag_st_deployment.

The first two branches shared the same objective. Both of them share the same logic in the first part of the project (creating chunks and embeddings in vector store), deviating only in the model used for the embeddings.
However, there are many differences between these branches in the second part of the project (creating the chatbot).

## Branch first_rag
First rag is the first model we did. It is very similar to the one shown in the tutorial for rag part 2 in Langchain, with the differences of using a Qdrant vector datastore for storing the embeddings. This rag worked well locally. We tried to deploy the project on Render.com; however, we hit memory issues.

## Branch light_rag
This branch had as objective to overcome the issues of the first rag by reducing the memory it needs. For that we aimed at 2 things: reducing the embeddings size and preventing the vector similarity to look at all chunks by applying a filter.

### Reducing embeddings size
The embeddings model we used in the first branch (Cohere's embeddingsv 4.0) allows for changing dimension sizes from the default (1536) to even 256. We tried this; however, the Langchain wrapper to use embeddings models did not allow for the change in embedding's size. We then had to choose between either not using the Langchain wrapper or choosing a different model with small default embedding size. We chose the later since it was the easiest solution (altough probably the less performant since the model we chose is older and 4.0 seems to do well even with small sizes due to the way it was trained).

We use the light version of embeddings v3.0, which has a 384 dimension

### Applying filter
The objective of the filter was to instead of searching for chunks of all possible boardgames, to only search for chunks coming from documents related to the specific boardgame the user is asking about. For this, we added a node in the graph. This node, which will be the first one to be called, receives the question of the user and the list of document names as input. The output of it is a list of document names that are related to the boardgame the user is inquiring about. We then used this list as input of the retriever to be used as filter, using the metadata value 'game' that all chunks possess.

However, the function we used to retrieve the data, which was a Langchain tool, did not allowed for extra inputs to be given (the list of relevant documents to be used in the filter). Hence, we changed this node from being a tool to being a regular node.

### Other changes
Besides that, I adapted the graph to only run the retriever and the node to detect useful documents if the question is related to a boardgame.

## Branch light_rag_st_deployment
Given the lack of success in deploying any of the previous 2 branches in Render, we focused on deploying them completely in Streamlit. We decided to only do this for the light_rag branch since it is the one I felt proudest the most. However, in order to deploy the app via the free Streamlit Community Cloud, the whole branch needs to be copied, and there does not exist a file to exclude files like .gitignore. This is a problem since the repo contains the pdf's of the rules  of the boardgames, which take up most of the memory of the repo. 

Furthermore, streamlit works only with 'requirements.txt' files for python packages (no other naming conventions allowed), and some extra changes need to be made since environment variables will be used instead of a .env file loaded via dotenv(). Because all of this, I decided the simplest solution was to create an additional branch which will only contain files needed for streamlit deployment.
