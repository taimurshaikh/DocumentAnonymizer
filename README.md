# Document Anonymizer
This program takes in a blob of text or the path to a .txt file and tries to remove all PII whilst still retaining information that is important to the text via keyword extraction.
 E-mails, Phone #s and links are redacted using heuristics. Other PII such as name, nationality, organization are redacted using Spacy's Named Entity Recognition (NER) system. A keyword extraction algorithm is run on the input text to extract words that contain the 'gist' of the text. The keywords that have been redacted by Spacy are then removed.

User Inputs: A string of text of any length<br/>
Outputs: The text

## Dependencies
Python 3.6+<br/>

### Windows
1. Run ```pip install -r requirements.txt```<br/>
2. Run ```python -m spacy download en_core_web_sm```

### Mac
1. Run ```pip3 install -r requirements.txt```<br/>
2. Run ```python3 -m spacy download en_core_web_sm```

## Usage

### Windows
Run ```python docAnon.py``` in the project directory

### Mac
Run ```python3 docAnon.py``` in the project directory

## Future Improvements
Get the PDF functionality working <br/>
Clean up some of these ugly empty print() statements and use delimiters instead<br/>
Better NER algorithm that doesn't have to make two passes of the tokenized text<br/>
My own keyword extraction algorithm/ tweaking the existing one<br/>
Support for non-PDF file formats

## Sources
Spacy Linguistic Features: https://spacy.io/usage/linguistic-features <br/>
Keyword extraction algorithm taken from Maarten Grootendorst "Keyword Extraction with BERT": https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea <br/>
PDF Coordinates: https://stackoverflow.com/questions/22898145/how-to-extract-text-and-text-coordinates-from-a-pdf-file
