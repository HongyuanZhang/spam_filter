import email
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_up_plain_text(string):
    words=list()
    tokens_raw = nltk.word_tokenize(string)
    i = 0
    while i < len(tokens_raw):
        cur = tokens_raw[i]  # current token
        if cur.isalpha() and len(cur) > 1:
            word = lemmatizer.lemmatize(cur.lower())  # lemmatize the lower-cased word
            if word not in stop_words:
                words.append(word)
        i += 1
        continue
    return words

def clean_up_html(html):
    words=list()
    tokens_raw = nltk.word_tokenize(html)
    inside_tag = False
    i = 0
    # go over the tokens
    while i < len(tokens_raw):
        cur = tokens_raw[i]  # current token
        # if encountering '<', which starts a tag of html
        if cur == '<' and not inside_tag:
            inside_tag = True
            i += 1
            continue
        if cur == '>' and inside_tag:  # if reaches the end of an html tag
            inside_tag = False
            i += 1
            continue
        if not inside_tag and cur.isalpha() and len(cur) > 1:  # if reaches a valid word
            word = lemmatizer.lemmatize(cur.lower())  # lemmatize the lower-cased word
            if word not in stop_words:
                words.append(word)  # append to list of words if not a stop word
        i += 1
        continue
    return words

def process_node(part):
    '''
    check whether this part is multipart or not
    '''
    if part.is_multipart():
        words=list()
        for payload in part.get_payload():
            words.extend(process_node(payload))
    else:
        #print(part.get_payload())
        content_type=part.get_content_type()
        if content_type=='text/plain':
            words=clean_up_plain_text(part.get_payload())
        elif content_type=='text/html':
            words=clean_up_html(part.get_payload())
        else:
            words=[]
    return words

def read_email(email_path):
    '''
    takes an email path
    traverse the multipart email until the end like a tree
    returns an array of words inside the email 
    '''
    with open(email_path, 'r') as f:
        raw=f.read()
        root=email.message_from_string(raw)
        words=process_node(root)
        return words

def read_labels(label_path):
    with open(label_path, 'r') as f:
        labels = f.readlines()
        labels = [label.partition(' ')[0] for label in labels]
        for i in range(len(labels)):
            if labels[i] == 'spam':
                labels[i] = 1
            else:
                labels[i] = 0
        return labels

if __name__ == '__main__':
    words=read_email('trec07p/data/inmail.1')
    print(words)
