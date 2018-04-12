from xlrd import open_workbook
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import spacy
nlp = spacy.load('pt')

#count words
def f1(post):
    return len([word for word in word_tokenize(post) if word not in string.punctuation+"\..."])

#count ner
def f2(post):
    post = nlp(post)
    return len(post.ents)

#lexical diversity(LD) all words
def f3(post):
    post = nlp(post)
    return len(set([word.lower_ for word in post if not word.is_punct]))/len(post)

#post depth
def f4(post):
    return 0

#LD content word (without stopwords)
def f5(post):
    post = nlp(post)
    return len(set([word.lower_ for word in post if not word.is_stop and word.lower_ not in stopwords.words('portuguese')]))/len(post)    

#average giveness
def f6(post):
    return 0

#question marks
def f7(post):
    return word_tokenize(post).count("?")

#similarity with previous message
def f8(post, previous):
    post = nlp(post)
    previous = nlp(previous)
    return post.similarity(previous)

#LD - VOC
def f9(post):
    post = nlp(post)
    return 0

#money-related words
def f10(post):
    money = [
        "bill",
        "capital",
        "cash",
        "check",
        "fund",
        "pay",
        "payment",
        "property",
        "salary",
        "wage",
        "wealth",
        "banknote",
        "bankroll",
        "bread",
        "bucks",
        "chips",
        "coin",
        "coinage",
        "dough",
        "finances",
        "funds",
        "gold",
        "gravy",
        "greenback",
        "loot",
        "pesos",
        "resources",
        "riches",
        "roll",
        "silver",
        "specie",
        "treasure",
        "wad",
        "wherewithal",
    ]
    post = nlp(post)
    count = 0
    for word in post:
        if(word.lower_ in money):
            count+=1
    return count

# avg sent per paragraph
def f11(post):
    paragraph = post.split('\n')
    count = 0
    for text in paragraph:
        text = nlp(text)
        count+=len([sent for sent in text.sents])
    return count/len(paragraph)

#similarity with posterior message
def f12(post, posterior):
    post = nlp(post)
    posterior = nlp(posterior)
    return post.similarity(posterior)

#numer of replies
def f13(postId,parentIds):
    return parentIds.count(postId)

#number of sentences
def f14(post):
    post = nlp(post)
    return len([sent for sent in post.sents])

#avg lsa similarity sentences
def f15(post):
    return 0

#avg sentence length
def f16(post):
    post = nlp(post)
    sents = [sent for sent in post.sents]
    count = 0
    for sent in sents:
        count+=len([word for word in sent if not word.is_punct])
    return count/len(sents)

#standard desviation of word syllables count
def f17(post):
    import pyphen
    import numpy as np
    dic = pyphen.Pyphen(lang='pt-br')
    post = nlp(post)
    return np.std([len(dic.inserted(word.lower_).split('-')) for word in post])

#abre planilhas
corpus = open_workbook('DBForumBrazilteste.xlsx',on_demand=True)

# acessa a planilha Planilha1
planilha1 = corpus.sheet_by_name("Sheet1")

#armazenando postagens em uma lista (posts) coluna G, linhas de 1 a 8
posts = planilha1.col_slice(colx=6,start_rowx=0,end_rowx=8)
ids = planilha1.col_slice(colx=0,start_rowx=0,end_rowx=8)
parentIds = planilha1.col_slice(colx=6,start_rowx=0,end_rowx=8)

for i in range(0,len(posts)):
    post = posts[i].value

    #exemple
    print(f1(post))