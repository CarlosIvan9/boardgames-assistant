


"""
flask-api.py

Creates an api to answer questions about boardgames using a rag system.

Usage locally or in a server.
Usage locally:
    python flask-app.py 
"""

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

# To work witth api
from flask import Flask, request

# To format outputs of api
from flask import  jsonify  # To return api answer as json in .py scripts
import json # To return api answer as json in .ipynb scripts

# Create id of thread (Memory of the chatbot will be reset every different day)
from datetime import datetime


#####################
### Set thread id ###
#####################

today = datetime.today().strftime("%Y%m%d")

# Specify an ID for the thread
config = {"configurable": {"thread_id": today}}


###########
### api ###
###########

app = Flask(__name__)

@app.route('/predict', methods=['POST'])  # Only POST
def predict():
    # Get question
    data = request.json
    input_message = data.get("question", "")
    
    print('Comment before invoking the graph')
    # Get output from rag
    answer = graph.invoke({"messages": [{"role": "user", "content": input_message}]}, config=config)
    print('Comment after invoking graph')
    
    # Format output to be able to be returned to user
    response = jsonify({"question": input_message, "answer": answer['messages'][-1].content}) # for scripts
    #response = json.dumps({"question": input_message, "answer": answer['messages'][-1].content}) #For notebooks

    return response





if __name__ == '__main__':
    app.run(debug=False) # debug=True reruns everything like every minute. This caused multiple calls of qdrant client, which led to an error


# To run app you have to do this in the terminal (powershell)
#1- $env:FLASK_APP = "my_app.py"
#2- flask run

# Alternatively you can run your app directly with Python (especially useful for larger apps):
# python my_app.py
# But for that to work, you need this at the bottom of my_app.py:
# if __name__ == "__main__":
#    app.run(debug=True)


# To test a post (in terminal. Git bash works well, powershell does not):
#curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"question": "How do you win at Memoir?"}'
#curl -X POST https://boardgames-assistant.onrender.com/predict -H "Content-Type: application/json" -d '{"question": "How do you win at Memoir?"}'



#curl -X POST https://sentiment-usecase.onrender.com/predict -H "Content-Type: application/json" -d '{"review": ["Me again", "Love", "I hated it"]}'



# Notes for deploying on render:
# * build commands: pip install -r requirements.txt
# * start command: gunicorn flask-app:app
#