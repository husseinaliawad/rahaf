from django.shortcuts import render, redirect
from django.utils.safestring import mark_safe  # Required for safely rendering HTML
from .models import FAQ
from .forms import FAQForm
from django.db.models import Q
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def home_view(request):
    return render(request, 'faq_app/home.html')

# Highlight terms in text
def highlight_terms(text, query_terms):
    """
    Highlights the query terms in the given text by wrapping them in a <span> tag.
    """
    for term in query_terms:
        if term:  # Ensure the term is not empty
            text = text.replace(term, f'<span class="highlight">{term}</span>')
    return mark_safe(text)  # Mark the text as safe for rendering in HTML

# View for adding questions and answers
def add_view(request):
    if request.method == 'POST':
        form = FAQForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('add')  # Redirect back to the add page after successful submission
    else:
        form = FAQForm()
    return render(request, 'faq_app/add.html', {'form': form})

# Boolean Search Model (Exact Match)
def boolean_search(query, faqs):
    query_terms = query.lower().split()
    results = []

    for faq in faqs:
        faq_text = faq.question.lower() + ' ' + faq.answer.lower()
        if all(term in faq_text for term in query_terms):
            results.append(faq)
    
    return results

# Extended Boolean Search Model (Fuzzy or Complex Matching)
def extended_boolean_search(query, faqs):
    query_terms = query.lower().split()
    results = []

    for faq in faqs:
        faq_text = faq.question.lower() + ' ' + faq.answer.lower()
        match_count = sum(1 for term in query_terms if term in faq_text)
        
        # We can assign a score based on the number of matches.
        score = match_count / len(query_terms) if query_terms else 0
        if score > 0:
            results.append((faq, score))
    
    # Sorting by score in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    
    return [faq for faq, _ in results]

# Vector Space Model (TF-IDF and Cosine Similarity)
def vector_space_search(query, faqs):
    # Prepare documents
    documents = [faq.question + ' ' + faq.answer for faq in faqs]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Transform query to vector
    query_vec = vectorizer.transform([query])
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix)
    
    # Filter results by a relevance threshold
    threshold = 0.1
    results = sorted(
        [(faq, score) for faq, score in zip(faqs, cosine_similarities[0]) if score > threshold],
        key=lambda x: x[1],
        reverse=True
    )
    
    return [faq for faq, _ in results]

# View for searching questions and retrieving answers
def search_view(request):
    query = request.GET.get('query', '')
    model_type = request.GET.get('model_type', 'boolean')  # Default to Boolean model

    if query:
        faqs = FAQ.objects.all()  # Get all FAQs from the database
        query_terms = query.lower().split()  # Split the query into terms

        # Apply different search models based on the selected model_type
        if model_type == 'boolean':
            faqs = boolean_search(query, faqs)
        elif model_type == 'extended_boolean':
            faqs = extended_boolean_search(query, faqs)
        elif model_type == 'vector':
            faqs = vector_space_search(query, faqs)

        # Highlight terms in questions and answers
        for faq in faqs:
            faq.question = highlight_terms(faq.question.lower(), query_terms)
            faq.answer = highlight_terms(faq.answer.lower(), query_terms)
    else:
        faqs = None

    return render(request, 'faq_app/search.html', {'faqs': faqs, 'query': query, 'model_type': model_type})
