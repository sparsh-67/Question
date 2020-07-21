import nltk
import sys
import string
import os
import math
FILE_MATCHES = 2
SENTENCE_MATCHES = 20


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files={}
    for filname in os.listdir(directory):
        with open(os.path.join(directory,filname)) as f:
            content=''.join([letter.lower() for letter in f.read()])
            files[filname]=content.lower()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    raw_words=[word.lower() for word in nltk.word_tokenize(document) if word.isalpha()]
    raw_words=[word.lower() for word in raw_words]
    final_words=[]
    for word in raw_words:
        if word not in string.punctuation and word not in nltk.corpus.stopwords.words("english"):
            final_words.append(word.lower())
    return final_words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf={}
    words={}
    # idf= no.of doc/no. of doc in which it lies
    for doc in documents:
        for wrd in set(documents[doc]):
            if wrd.lower() not in words:
                words[wrd.lower()]=0
            words[wrd.lower()]+=1 
    for word in words:
        idf[word]=len(documents)/words[word]
    return idf


def top_files(query, files, idf, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # file name vs total idf value
    tfidf={}
    for file in files:
        total_idf=0
        for word in query:
            if word in files[file]:
                total_idf+=idf[word]
        tfidf[file]=total_idf
    ll=[(tfidf[i],i) for i in tfidf]
    ll.sort(reverse=True)
    ans=[]
    for i in range(n):
        ans.append(ll[i][1])
    return ans



def top_sentences(query, sentences, idf, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    ll=[]
    for s in sentences:
        st=sentences[s]
        st=[word.lower() for word in st]
        found_word=0
        total_idf=0

        for word in query:
            if word in st:
                total_idf+=idf[word]
                found_word+=1 
        ll.append((total_idf,found_word/len(st),s))
    ll.sort(reverse=True)
    #print(ll)
    ans=[]
    for i in range(n):
        ans.append(ll[i][2])
    #print("answer is : ",*ans)
    return ans



if __name__ == "__main__":
    main()
