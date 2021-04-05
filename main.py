import spacy
from spacy.lang.en.examples import sentences 
nlp = spacy.load("en_core_web_sm")

def main():
    inputText = input("Enter text: ")
    processedText = nlp(inputText)
    print(processedText)
if __name__ == "__main__":
    main()
