import random

# List of product names
product_names = ["SuperSavers", "KidsSavers", "SavingsPro", "MoneyMarketPro", "ForexPro", "BumperBusiness"]

# List of product descriptions
product_descriptions = [
    "A high-yield savings account with no minimum balance requirement.",
    "A savings account designed for kids with features like parental controls and educational resources.",
    "A high-interest savings account with a variety of features, including ATM access and online bill pay.",
    "A money market account with competitive interest rates and a variety of features, including check writing and FDIC insurance.",
    "A foreign exchange account that allows you to trade currencies online.",
    "A business checking account with features like free ATMs and online bill pay."
]

# Generate synthetic data
data = []
for i in range(100):
    product_name = random.choice(product_names)
    product_description = random.choice(product_descriptions)
    data.append([product_name, product_description])

# Write synthetic data to CSV file
import csv
with open('banking_products.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['product_name', 'product_description'])
    writer.writerows(data)



# Import the necessary libraries
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('banking_products.csv')

# Create a function to calculate the cosine similarity between two vectors
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2)

# Create a vector for each product
product_vectors = []
for i in range(len(df)):
    product_vectors.append(df.iloc[i, 1].split())

# Calculate the cosine similarity between all pairs of products
product_similarities = np.zeros((len(df), len(df)))
for i in range(len(df)):
    for j in range(len(df)):
        product_similarities[i, j] = cosine_similarity(product_vectors[i], product_vectors[j])

# Create a function to recommend products to a customer
def recommend_products(user_id):
    # Get the products that the user has used or rated
    user_ratings = df[df['user_id'] == user_id]

    # Get the products that the user has not used or rated
    unrated_products = df[~df['user_id'].isin([user_id])]

    # Calculate the cosine similarity between the user's ratings and all of the unrated products
    product_recommendations = np.zeros(len(unrated_products))
    for i in range(len(unrated_products)):
        product_recommendations[i] = np.dot(user_ratings, unrated_products.iloc[i, 1].split())

    # Sort the recommendations by similarity
    product_recommendations = np.argsort(product_recommendations)[::-1]

    # Return the top 10 recommendations
    return unrated_products.iloc[product_recommendations[:10], :].reset_index(drop=True)

# Test the recommender system
user_id = 1
print("Recommendations for user {}".format(user_id))
print(recommend_products(user_id))


import streamlit as st
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('banking_products.csv')

# Create a function to calculate the cosine similarity between two vectors
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2)

# Create a vector for each product
product_vectors = []
for i in range(len(df)):
    product_vectors.append(df.iloc[i, 1].split())

# Calculate the cosine similarity between all pairs of products
product_similarities = np.zeros((len(df), len(df)))
for i in range(len(df)):
    for j in range(len(df)):
        product_similarities[i, j] = cosine_similarity(product_vectors[i], product_vectors[j])

# Create a function to recommend products to a customer
def recommend_products(user_id):
    # Get the products that the user has used or rated
    user_ratings = df[df['user_id'] == user_id]

    # Get the products that the user has not used or rated
    unrated_products = df[~df['user_id'].isin([user_id])]

    # Calculate the cosine similarity between the user's ratings and all of the unrated products
    product_recommendations = np.zeros(len(unrated_products))
    for i in range(len(unrated_products)):
        product_recommendations[i] = np.dot(user_ratings, unrated_products.iloc[i, 1].split())

    # Sort the recommendations by similarity
    product_recommendations = np.argsort(product_recommendations)[::-1]

    # Return the top 2 recommendations
    return unrated_products.iloc[product_recommendations[:2], :].reset_index(drop=True)

# Create a Streamlit app
st.title('Banking Product Recommender')

# Get the user's ID
user_id = st.text_input('Enter your user ID')

# Recommend products to the user
recommendations = recommend_products(user_id)

# Display the recommendations to the user
st.write('Here are the top 2 recommended products for you:')
st.table(recommendations)

