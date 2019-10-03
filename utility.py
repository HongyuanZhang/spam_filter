import email
import html2text

email_path = 'spam_classifier\\trec07p\\trec07p\data\inmail.100'
label_path = 'spam_classifier\index'

def read_emails(email_path):
    with open(email_path, 'r') as f:
        a = f.read()
        b = email.message_from_string(a)
        if b.is_multipart():
            for payload in b.get_payload():
                if payload.is_multipart():
                # if payload.is_multipart(): ...
                    for p in payload.get_payload():
                        print(p.get_payload())
                else:
                    print(payload.get_payload())
        else:
            print(b.get_payload())

def read_labels(label_path):
    with open(label_path, 'r') as f:
        for i in range(99):
            f.readline()
        print(f.readline())

read_emails(email_path)
#read_labels(label_path)