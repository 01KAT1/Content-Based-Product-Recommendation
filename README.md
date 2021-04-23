# Content-Based-Product-Recommendation
"""
File - Content_Based_Product_Recommender_System
Aim - To recommend top 10 fashion products based on the last product bought by the user.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Use the following code snippet to see data of all columns in pycharm (it actually displays '...' after 2 columns if
# there are more than 5 columns to display)
desired_width = 1000
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

# Read csv files
df = pd.read_csv('sample-data.csv')

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a', 'an', 'and' etc.
tfidf = TfidfVectorizer(stop_words='english', min_df=0)

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['description'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Some data processing
df['title'] = df['description'].astype(str).str.split(' - ').str[0]
indices = pd.Series(df.index, index=df['title']).drop_duplicates()


# A function which recommends products to the user, given the last product they bought
def Predict(title):
    # Get the index of the product that matches the product name
    idx = indices[title]

    # Convert the cosine matrix to a list of tuples where the 1st ele is its pos, and the 2nd is the similarity score.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar products
    sim_scores = sim_scores[1:11]

    # Get the product indices
    result = [i[0] for i in sim_scores]

    # Return the top 10 most similar products
    return df['title'].iloc[result].drop_duplicates()


print()
df_new = Predict(input('Enter the last item you bought: '))
df_new.index = np.arange(1, len(df_new) + 1)
print()
print("Top 10 items recommended for you based on the the last item you bought: ")
print()
print(df_new)

