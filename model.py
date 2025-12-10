# Import libraries
import pandas as pd


# Taking the relevant columns in the datadet
books_df = pd.read_csv("books.csv", usecols=[0,1,2,3])

# Making user preference matrix and dataframe

Aviral_preference = [
    {'title': 'The Lord of the Rings (The Lord of the Rings  #1-3)','rating':5},
    {'title':'The House on Mango Street', 'rating':3.5},
    {'title':'Anna Karenina', 'rating':0},
    {'title':'Dalit: The Black Untaouchables of India', 'rating':3},
    {'title':'Intimate Communion: Awakening Your Sexual Essence', 'rating':2}
    ]

Aviral_preference = pd.DataFrame(Aviral_preference)

Aviral_preference_id = books_df[books_df['title'].isin(Aviral_preference['title'])]

Aviral_preference = pd.merge(Aviral_preference_id, Aviral_preference, on = 'title')

Aviral_preference = Aviral_preference.drop(columns=['average_rating','authors'])


# Selecting text feture to implemented similarity functions on:
books_df['features'] = books_df['title'].astype(str) + " " + books_df['authors'].astype(str)

#tf-idf:
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_df['features'])

# Implementing cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tfidf_matrix)

rating_weights = dict(zip(Aviral_preference['title'], Aviral_preference['rating']))


# Adding weights based on rating given by user:
def weighted_recommend(book_title, weight, n=100):
    try:
        index = books_df[books_df['title'] == book_title].index[0]
    except:
        return []

    sim_scores = list(enumerate(similarity_matrix[index]))

    # Apply weight to similarity score
    weighted_scores = [(i, score * weight) for i, score in sim_scores]

    weighted_scores = sorted(weighted_scores, key=lambda x: x[1], reverse=True)

    rec_books = [books_df.iloc[i[0]]['title'] for i in weighted_scores[1:n+1]]
    return rec_books


final_scores = {}

for book in Aviral_preference['title']:
    weight = rating_weights[book]

    recs = weighted_recommend(book, weight, n=100)

    for r in recs:
        if r not in Aviral_preference['title']:  # don't recommend books you already like
            final_scores[r] = final_scores.get(r, 0) + weight   # accumulate score

# Make ranked recommendations:
ranked_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

print("\n Your Weighted Personalized Recommendations:\n")
for book, score in ranked_recommendations[:10]:
    print(f"→ {book}")

