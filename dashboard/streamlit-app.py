


"""
flask-app.py

Creates an app and an api to calculate if a movie review is positive or negative.

Usage locally or in a server.
Usage locally:
    python flask-app.py 
"""



from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import os
import ast
from huggingface_hub import InferenceClient
#from dotenv import load_dotenv
#load_dotenv()  # Loads from .env

my_token=os.getenv("HUGGING_FACE_TOKEN")



app = Flask(__name__)


def get_sentiment(reviews, positive_label, negative_label):
    """
    Calculates sentiment of a review or a list of reviews. 

    Args:
        reviews (str or list): a single review (str) or a list of reviews (list)
        positive_label (str): label indicating positive review
        negative_label (str): label indicating negative review

    Returns:
        outputs_list (list): list with the sentiment of the reviews (positive or negative).
    """
    # Make sure input is a list (output of hf is 3 classes if a string is given, or just the top class if a list is given)
    if not isinstance(reviews, list):
        reviews = [reviews]
    client=InferenceClient(
        model='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
        token=my_token
        )
    
    # Inference for all reviews
    all_rows = []
    for i, review in enumerate(reviews):
        review = review[:4010] # truncation of long reviews to the maximum we have seen
        output = client.zero_shot_classification(review, candidate_labels = [positive_label, negative_label])
        for item in output:
            all_rows.append({
                "review_index": i,
                "review" : review, 
                "label": item["label"],
                "score": item["score"]
            })

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # Extract positive score only
    df = df[df["label"] == positive_label].reset_index(drop=True)

    # Format output df 
    df.drop(columns=['label'], inplace=True)
    df.rename(columns={'score':'positive_score'}, inplace=True)

    outputs_list = ['positive' if score > 0.5 else 'negative' for score in df['positive_score']]
    return outputs_list



# HTML form template
form_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment App</title>
</head>
<body>
    <h2>Sentiment App</h2>
    <h3>Enter a review</h3>
    <form method="POST" action="/predict">
        <input type="text" name="review" required>
        <button type="submit">Sentiment</button>
    </form>
    {% if sentiment %}
        <h4> {{ review }}: {{ sentiment }}</h4>
    {% endif %}
</body>
</html>
'''

positive_label='a positive movie review (regardless if the movie context is negative)'
negative_label='a negative movie review (regardless if the movie context is positive)'


@app.route('/', methods=['GET'])
def index():
    """
    Redirects to the template to add the reviews. 

    Returns:
        Template to add the reviews
    """
    return render_template_string(form_html, sentiment=None)

@app.route('/predict', methods=['POST'])
def predict_sentiment(): 
    """
    Calculates and returns sentiments of a review or a list of reviews. Works both as an api or a webapp.

    Returns:
        Template with the reviews and their sentiment.
    """
    if request.is_json:  # Applies for curl requests
        body = request.get_json()
        review = body.get("review", "")
        if not review:
            return jsonify({"error": "Missing review"}), 400
        sentiments = get_sentiment(review, positive_label, negative_label)
        return jsonify(sentiments), 200

    else: #Applies for request from html template
        review = request.form.get('review', '')  

        try: # In case a list was received, to be able to treat it as such
            review = ast.literal_eval(review)
        except:
            review = review

        sentiments = get_sentiment(review, positive_label, negative_label)
        if len(sentiments)==1:  #Cleaner output in case a single review is given
            sentiments=sentiments[0]
        return render_template_string(form_html, sentiment=sentiments, review=review)
    


if __name__ == '__main__':
    app.run(debug=True)


# To run app you have to do this in the termal (powershell)
#1- $env:FLASK_APP = "my_app.py"
#2- flask run

# Alternatively you can run your app directly with Python (especially useful for larger apps):
# python my_app.py
# But for that to work, you need this at the bottom of my_app.py:
# if __name__ == "__main__":
#    app.run(debug=True)


# To test a post (in terminal. Git bash works well, powershell does not):
#curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"review": "I am alright and you?"}'
# In a list, double members inside it should appear in double quotes
#curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"review": ["Me again", "Trump is blonde", "I hated it"]}'


#curl -X POST https://sentiment-usecase.onrender.com/predict -H "Content-Type: application/json" -d '{"review": ["Me again", "Love", "I hated it"]}'

# Test with a big review
#"One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."

