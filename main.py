import re
import spacy
from spacy.tokenizer import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# For the proper nouns method we need to ensure quotation marks are not parsed as their own tokens
# THis counteracts the "Goku" example
customRe = re.compile("\".+\"|\'.+\'")

def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, prefix_search=customRe.search)

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = custom_tokenizer(nlp)

def main():
    inputText = input("Enter text: ")
    emailLess = removeEmail(inputText)
    phoneLess = removePhoneNo(emailLess)
    print(phoneLess)
    redactedText = removeProperNouns(phoneLess)
    keywords = getKeywords(inputText)
    print(' '.join(redactedText))
    print(keywords)

def removeEmail(text):
    return ' '.join(["[REDACTED]" if re.search(".+@.+\.com|\.org", word) != None else word for word in text.split()])

def removePhoneNo(text):
    # This regex is taken from this stack overflow question https://stackoverflow.com/questions/13354221/regex-to-match-1-or-less-occurrence-of-string
    # Currently only works in the format (XXX)XXXXXXX
    # May even work for things like ID numbers
    return ' '.join(["[REDACTED]" if re.search(".*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?", word) != None else word for word in text.split()])

def removeProperNouns(text):
    tokenized = nlp(text)
    redactedText = []

    for token in tokenized:
        print(token.text)
        # Proper nouns could divulge info so redact them
        # Proper nouns have UPOS code PROPN
        # The custom re makes it so that Proper Nouns in quotes aren't redacted. This is kind of a heuristic though as quoted PNs could still be PII
        # NOTE: not the best approach, as not all proper nouns are PII, and some could still be important to text
        if token.pos_ == "PROPN" and re.search("\".+\"|\'.+\'", token.text) is None:
            redactedText.append("[REDACTED]")
            continue

        redactedText.append(token.text)
    return redactedText

def getKeywords(text):
    # This function is from Grootendorst's article (cited in README)
    n_gram_range = (1, 1)
    stop_words = "english"

    try:
        # Extract candidate words/phrases
        count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text])
    except ValueError:
        print("NO VOCAB")
        return None

    candidates = count.get_feature_names()

    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_embedding = model.encode([text])
    candidate_embeddings = model.encode(candidates)
    top_n = 5
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    return keywords

if __name__ == "__main__":
    main()
