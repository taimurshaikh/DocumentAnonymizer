# Document Anonymizer
This program takes in a blob of text and tries to remove all PII whilst still retaining information that is important to the text.

## Dependencies

### Windows
1. Run ```pip install -r requirements.txt```<br/>

2. Run ```python -m spacy download en_core_web_sm```

### Mac
1. Run ```pip3 install -r requirements.txt```<br/>

2. Run ```python3 -m spacy download en_core_web_sm```

## Usage

### Windows
```python main.py```

### Mac
```python3 main.py```

## Sources
Spacy Linguistic Features: https://spacy.io/usage/linguistic-features <br/>
Maarten Grootendorst "Keyword Extraction with BERT": https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea <br/>
