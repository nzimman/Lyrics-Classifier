# Lyrics Classifier (Natural Language Processing project)

## Synopsis
Project based on a text classification model to predict an artist from a piece of text.

## Code workflow
Given a list of artists, the model extracts data from HTML pages, vectorize and transform text lyrics and applies a model to classify the text to predict the artist.

### Extract data from HTML pages
For each artist, the model:
- Scrapes a given website (hardcoded in the code), creates a list with all the links to the pages that contains the songs and saves this list in a csv file
- Given the previous list, the program goes to each of these pages and finds the link to the lyrics text. Saves the information in another csv file
- The last step consists of reading the file with the song links, finds the text of the song and saves it in a list. It saves also a list with the titles of the songs (csv format)

For extracting the data, the follwoing libraries are used:
- To download web pages the *requests* module
- Because the output of a request is HTML code, the *BeautifulSoup* library helps to parse the code. Once it is converted to a BeautifulSoup object, it is possible to find the tag that contains the data needed.
- *Pandas* library is also used for saving the data in csv format

### Text cleaning
The goal is to remove from the text punctuation, stop words (most common words in a language) and pronouns. For this, use the spyCy library (the model *en_core_web_md*) to break the text into tokens, see the part of the speech of each token and find the root of each word (lemmatization) to clean the text.  

### Vectorize
Convert the clean text into a (sparse) matrix of token counts.

Library needed: *sklearn*


### Transform
Transform the count matrix to a normalized tf (term-frequency) or tf-idf (term-frequency times inverse document-frequency) representation. Instead of the raw frequencies of occurrence of a token, the goal is to adjust for words that are generally more frequent.

TFIDF=TF(t,d)∗(log1+n1+df(d,t)+1)

TF(t, d) is the number of times term t occurs in document d
df(d, t) is the number of documents containing term t
n is the number of documents


### 

## Implementation
The file lyrics_classifier.py contains all the code and the list of artists can be input using the command line. The code expects a folder called *Songs* in the same folder where the .py file is. All the data extracted from the HTML pages is saved in csv format in this folder.  

There is a class *Artist* and additional functions defined.

Necessary libraries for running the code:
- Pandas
- Numpy
- BeuatifulSoup
- spaCy
- sklearn





