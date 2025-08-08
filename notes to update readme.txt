


embedding does not work for different languages, in the sense that 2 sentences that are similar but
in a different language, will not have similar encodings...

so we have to translate files to english before calculating embeddings, and also have to check if user 
question is in english. if not we have to translate it to english and then apply the embeddings retriever.

i can use tureluurs specific questions to test that since rulebook is in dutch (not general questions since
we also uploaded an unofficial summary of the game in english.)