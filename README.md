### Hong Kong Hotel Recommender App
This recommender app I built will be able to take in tourist attraction names in Hong Kong, and recommends hotels based on these places. I first made use of SpaCy to remove stopwords from the previous hotel reviews and grouped all reviews that belonged to the same hotel together. I then made use of cosine similarity to output recommendations based on the input features. Each recommendation also has its score, the higher the better. 

To add some funsies, I also created a Wordcloud plot for each recommendation to visualize the most frequently mentioned terms in its reviews.

Check out the streamlit app on the following link to start your Hong Kong hotel search: https://share.streamlit.io/yxliu926/hw3/main/Codes.py
