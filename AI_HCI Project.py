
"""Importing necessary libraries and preprocessing the dataset by filling missing values and combining features for similarity calculation. 
Transforming text data using TF-IDF and cosine similarity to recommend top books based on user selection and similarity scores.
"""

# Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading the dataset from the specified path
books_data = pd.read_csv("D:/datasets/books.csv", on_bad_lines='skip')

# Handling missing data by filling with Unknown
books_data.fillna('Unknown', inplace=True)

# Combining authors and titles to create a new feature for similarity calculation
books_data['combined_features'] = books_data['authors'] + " " + books_data['title']

# Creating a TF-IDF Vectorizer to convert text data into numerical vectors
vectorizer = TfidfVectorizer(stop_words='english')

# Fitting the vectorizer and transforming the combined features into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(books_data['combined_features'])

# Calculating the cosine similarity matrix between books
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Defining a function to recommend books based on a given book title
def recommend_books(book_title, cosine_sim=cosine_sim):
    # Stripping spaces and converting the input title to lowercase for flexible matching
    book_title = book_title.strip().lower()
    books_data['title_clean'] = books_data['title'].str.strip().str.lower()

    # Finding book titles that contain the given title 
    matched_idx = books_data[books_data['title_clean'].str.contains(book_title, na=False)].index

    # Returning a message if no matching book is found
    if len(matched_idx) == 0:
        return "Book title not found. Please make sure the title is correct."

    # Getting the index of the first matched book
    idx = matched_idx[0]
    
    # Getting the similarity scores of all books with the selected book
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sorting the books based on similarity in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Getting the indices of the top 5 most similar books
    sim_scores = sim_scores[1:6]  
    book_indices = [i[0] for i in sim_scores]
    
    # Returning the titles of the top 5 recommended books
    return books_data['title'].iloc[book_indices]

# Displaying the first 10 book titles from the dataset as sample options
print("Sample Book Titles from the Dataset:\n")
print(books_data['title'].head(10))  

# Asking the user to select a book title from the list
selected_book = input("\nSelect a book title from the list above: ")

# Displaying the top 5 recommended books based on the selected book
recommended_books = recommend_books(selected_book)
print(f"\nBooks recommended for '{selected_book}':")
print(recommended_books)