# Import necessary modules for vector space model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define the FAQ class (this can be adjusted to match your database structure if needed)
class FAQ:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer

# Sample list of FAQ objects
faqs = [
    FAQ('What is Django?', 'Django is a Python-based web framework for rapid development of web applications.'),
    FAQ('How do I install Django?', 'You can install Django using pip: pip install django.'),
    FAQ('What is a model in Django?', 'In Django, a model is a Python class that defines the structure of your database tables.'),
    FAQ('What is Python?', 'Python is a high-level programming language used for general-purpose programming.'),
    FAQ('How do I run a Django project?', 'You can run a Django project using: python manage.py runserver.')
]

# 1. Boolean Search Model with Ranking
def boolean_search(query, faqs):
    query_terms = query.lower().split()
    results = []
    
    for faq in faqs:
        faq_text = faq.question.lower() + ' ' + faq.answer.lower()
        matched_terms = [term for term in query_terms if term in faq_text]
        match_count = len(matched_terms)  # Number of terms matched
        
        if match_count > 0:
            results.append((faq, match_count))  # Append FAQ with the match count
    
    # Sort results by match count in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return only FAQs, sorted by the number of matched terms
    return [faq for faq, _ in results]

# 2. Extended Boolean Search Model with Ranking
def extended_boolean_search(query, faqs):
    query_terms = query.lower().split()
    results = []

    for faq in faqs:
        faq_text = faq.question.lower() + ' ' + faq.answer.lower()
        match_count = sum(1 for term in query_terms if term in faq_text)
        
        # Assign a score based on the number of matches
        score = match_count / len(query_terms) if query_terms else 0
        if score > 0:
            results.append((faq, score))
    
    # Sort by score in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return sorted FAQs based on score
    return [faq for faq, _ in results]

# 3. Vector Space Model with Ranking using Cosine Similarity
def vector_space_search(query, faqs):
    documents = [faq.question + ' ' + faq.answer for faq in faqs]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    query_vec = vectorizer.transform([query])
    
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix)
    
    # Sort documents by cosine similarity score
    results = sorted(zip(faqs, cosine_similarities[0]), key=lambda x: x[1], reverse=True)
    
    # Return FAQs sorted by cosine similarity score
    return [faq for faq, _ in results]

# Function to perform the search with a specific model
def perform_search(query, faqs, model='boolean'):
    if model == 'boolean':
        return boolean_search(query, faqs)
    elif model == 'extended_boolean':
        return extended_boolean_search(query, faqs)
    elif model == 'vector_space':
        return vector_space_search(query, faqs)
    else:
        raise ValueError("Unsupported search model!")

# Test search queries
query = "install django"
model_choice = 'vector_space'  # Choose between 'boolean', 'extended_boolean', 'vector_space'

# Perform the search
search_results = perform_search(query, faqs, model=model_choice)

# Display search results
print(f"Search Results for query: '{query}' using {model_choice} model")
print("="*50)
for idx, faq in enumerate(search_results, start=1):
    print(f"{idx}. {faq.question} - {faq.answer}")
