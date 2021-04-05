import re
from itertools import groupby
import spacy
from spacy.tokenizer import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from os import path

# from pdfminer.layout import LAParams, LTTextBox
# from pdfminer.pdfpage import PDFPage
# from pdfminer.pdfinterp import PDFResourceManager
# from pdfminer.pdfinterp import PDFPageInterpreter
# from pdfminer.converter import PDFPageAggregator


OUTPUT_FILE_PATH = "result.txt"

nlp = spacy.load("en_core_web_sm")

def main():
    """
    Main driver function
    """
    options = ["1", "2"]
    choice = input("1. Text\n2. File Path\n")
    while choice not in options:
        print("ERROR: Invalid option. Please try again.")
        choice = input("1. Text\n2. File Path\n")

    # Hacky way of doing it but I'm rushing :D
    if not int(choice) - 1:
        inputText = input("Enter text: ")
    else:
        filePath = input("Enter file path to .txt file: ")
        if not filePath.endswith(".txt"):
            filePath += ".txt"
        while not path.exists(filePath):
            print("ERROR: File not found or invalid file type. Please try again.\n")
            filePath = input("Enter file path to .txt file: ")
            if not filePath.endswith(".txt"):
                filePath += ".txt"

        with open(filePath, "r") as f:
            inputText = f.read()

    # if filePath[-4:] != ".pdf":
    #     filePath += ".pdf"
    # print()
    #
    # # Validate file path
    # while not path.exists(filePath):
    #     print("ERROR: Could not find file. Please try again.")
    #     filePath = input("Enter file name: ")
    #     if filePath[-4:] != ".pdf":
    #         filePath += ".pdf"
    #     print()

    # # PDF Handling
    # fp = open(filePath, 'rb')
    # rsrcmgr = PDFResourceManager()
    # laparams = LAParams()
    # device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    # interpreter = PDFPageInterpreter(rsrcmgr, device)
    # pages = PDFPage.get_pages(fp)
    #
    # totalText = []
    #
    # for page in pages:
    #     print('Processing next page...')
    #     interpreter.process_page(page)
    #     layout = device.get_result()
    #     for lobj in layout:
    #         if isinstance(lobj, LTTextBox):
    #             print(lobj.bbox)
    #             x0, y0, x1, y1 = lobj.bbox[0], lobj.bbox[3], lobj.bbox[1], lobj.bbox[4]
    #             text = lobj.get_text()
    #             totalText.append(text)
    #             removeEmail(text)
    #             removePhoneNo(text)
    #             removeLinks(text)
    #             removeEntities(text)

    redactedText = removeEmail(inputText)
    redactedText = removePhoneNo(redactedText)
    redactedText = removeLinks(redactedText)
    redactedText = removeEntities(redactedText)

    # Remove consecutive occurences of [REDACTED]
    redactedText = [x[0] for x in groupby(redactedText)]

    keywords = getKeywords(inputText)
    if keywords is None:
        print("There were no keywords found")
        quit()
    keywords = removeRedactedKeywords(keywords, redactedText)

    print("REDACTED TEXT:")
    print(' '.join(redactedText))
    print()
    print("KEYWORDS: ")
    print(', '.join(keywords))

    with open(OUTPUT_FILE_PATH, "w+") as f:
        f.write("REDACTED TEXT:\n")
        f.write(' '.join(redactedText)+"\n")
        f.write("KEYWORDS:\n")
        f.write(' '.join(keywords))

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

def removeLinks(text):
    """
    Heuristic for removing web links
    Parameters: text -> str
    Returns: str
    """
    return ' '.join(["[REDACTED]" if re.search("www\..+\.", word) != None else word for word in text.split()])

def removeEntities(text):
    """
    Removes all instances of specified entity types from text
    Parameters: text -> str
    Returns: str[]
    """
    tokenized = nlp(text)
    redactedText = []
    entitiesToRedact = ["PERSON", "NORP", "ORG", "GPE", "LOC"]

    # Spacy is not 100%, so this set will act as a lookup table to check if the current token has been previously redacted
    redactedEntityLookup = set()

    for token in tokenized:
        if token.text in "-().,;:/`''-_!?&":
            continue
        # Some proper nouns could divulge info so redact them
        # The custom re makes it so that Proper Nouns in quotes aren't redacted. This is kind of a heuristic though as quoted PNs could still be PII
        # NOTE: not the best approach, as not all proper nouns are PII, and some could still be important to text
        if token.text.strip() in redactedEntityLookup or (token.ent_type_ in entitiesToRedact and not token.is_stop and re.search("\".+\"|\'.+\'", token.text) is None):
            redactedEntityLookup.add(token.text)
            continue

    # Sometimes the spacy innacuracies can be BEFORE the token first gets redacted
    # Because of this we need to parse the text one more time
    for token in tokenized:
        if token.text in redactedEntityLookup:
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
