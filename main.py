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
    """
    Modifies spaCy's tokenizer class with custom regex Match object
    Parameters: nlp -> spacy.lang..en.English object
    Returns: spacy.lang..en.English.Tokenizer object
    """
    return Tokenizer(nlp.vocab, prefix_search=customRe.search)

nlp = spacy.load("en_core_web_sm")
# nlp.tokenizer = custom_tokenizer(nlp)

def main():
    """
    Main driver function
    """
    inputText = input("Enter text: ")
    emailLess = removeEmail(inputText)
    phoneLess = removePhoneNo(emailLess)
    redactedText = removeEntities(phoneLess)
    keywords = getKeywords(inputText)
    keywords = removeRedactedKeywords(keywords, redactedText)
    print(' '.join(redactedText))
    print(keywords)

def removeEmail(text):
    """
    Somewhat reliable heuristic for removing emails
    Parameters: text -> str
    Returns: str
    """
    return ' '.join(["[REDACTED]" if re.search(".+@.+\.com|\.org", word) != None else word for word in text.split()])

def removePhoneNo(text):
    """
    Heuristic for removing phone numbers
    Parameters: text -> str
    Returns: str
    """
    # This regex is taken from this stack overflow question https://stackoverflow.com/questions/13354221/regex-to-match-1-or-less-occurrence-of-string
    # Works for the format (XXX)XXXXXXX
    # May even work for things like ID numbers
    return ' '.join(["[REDACTED]" if re.search(".*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?", word) != None else word for word in text.split()])

def removeEntities(text):
    """
    Removes all instances of specified entity types from text
    Parameters: text -> str
    Returns: str
    """
    tokenized = nlp(text)
    redactedText = []
    entitiesToRedact = ["PERSON", "NORP", "ORG", "GPE", "LOC"]
    for token in tokenized:
        # Some proper nouns could divulge info so redact them
        # The custom re makes it so that Proper Nouns in quotes aren't redacted. This is kind of a heuristic though as quoted PNs could still be PII
        # NOTE: not the best approach, as not all proper nouns are PII, and some could still be important to text
        if token.ent_type_ in entitiesToRedact and re.search("\".+\"|\'.+\'", token.text) is None:
            redactedText.append("[REDACTED]")
            continue

        redactedText.append(token.text)
    return redactedText

def getKeywords(text):
    """
    Finds keywords in a blob of text
    Parameters: text -> str
    Returns: str[]
    """
    # This keyword extraction function is from Grootendorst's article (cited in README)
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
    top_n = 8
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    return keywords

def removeRedactedKeywords(keywords, textLst):
    """
    Removes all the keywords that were previously redacted due to being considered personal information
    Parameters: keywords -> str[], textLst -> str[]
    Returns: str[]
    """
    textLst = [word for word in textLst if word != "[REDACTED]" ]
    textLst = [word.lower() for word in textLst]
    return list(set(keywords).intersection(set(textLst)))

if __name__ == "__main__":
    main()
