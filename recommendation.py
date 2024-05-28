import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Create a sample user-item matrix where rows represent users and columns represent movies
data = {
    'user': ['Ravi', 'Ravi', 'Ravi', 'Neeta', 'Neeta', 'Aakash', 'Aakash', 'Aakash'],
    'movie': ['The Marvels', 'Spiderman', 'Black panther', 'The Marvels', 'Avengers', 'Spiderman', 'Black panther', 'Ant man'],
    'rating': [5, 3, 4, 4, 2, 3, 4, 5]
}

df = pd.DataFrame(data)

# Pivot the dataframe to create a user-item matrix
user_item_matrix = df.pivot(index='user', columns='movie', values='rating').fillna(0)

# Normalize the user-item matrix
scaler = StandardScaler(with_mean=False)
user_item_matrix_normalized = scaler.fit_transform(user_item_matrix)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix_normalized)

# Create a DataFrame from the similarity matrix
user_similarity_df = pd.DataFrame(
    user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
)

def get_user_recommendations(user_id, top_n=2):
    # Get the top similar users to the given user_id
    similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False)

    # Remove the current user from the list
    similar_users = similar_users.drop(user_id)

    # Get the recommendations from the most similar users
    recommendations = {}

    # Check top similar users
    for similar_user in similar_users.index:
        # Get movies rated by similar_user but not by current user
        similar_user_ratings = user_item_matrix.loc[similar_user]

        # Find unrated movies by current user
        unrated_by_current_user = user_item_matrix.loc[user_id] == 0

        # Get the unrated movies by the current user from the similar user's ratings
        movies_to_recommend = similar_user_ratings[unrated_by_current_user]

        # Add the movies and their ratings to the recommendations dictionary
        for movie, rating in movies_to_recommend.items():
            if movie not in recommendations:
                recommendations[movie] = rating
            else:
                recommendations[movie] += rating  # If multiple users recommend the same movie, add their ratings

    # Sort recommendations by summed rating (highest to lowest)
    sorted_recommendations = sorted(
        recommendations.items(), key=lambda x: x[1], reverse=True
    )

    # Return the top_n recommendations
    return sorted_recommendations[:top_n]

# Get recommendations for user 1
recommendations = get_user_recommendations('Ravi', top_n=2)
print("Recommendations for Ravi are :", recommendations)

# Get recommendations for user 2
recommendations = get_user_recommendations('Neeta', top_n=2)
print("Recommendations for Neeta are :", recommendations)

# Get recommendations for user 3
recommendations = get_user_recommendations('Aakash', top_n=2)
print("Recommendations for Aakash are :", recommendations)