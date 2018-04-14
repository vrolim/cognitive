from xlrd import open_workbook
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import spacy
import pyphen
import numpy as np

nlp = spacy.load('pt')

class Node:
    def __init__(self,post,parent):
        self.post = post
        self.parent = parent
        self.depth = 0

class Tree:
    def __init__(self,postIds, parentIds):
        self.ids = postIds
        self.parents = parentIds
        self.nodes = []

        for i in range(len(self.ids)):
            nd = self.nodeExists(i)
            node = Node(self.ids[i],self.parents[i])
            self.nodes.append(node)           

    def nodeExists(self,id):
        for node in self.nodes:
            if(id==node.post):
                return node
        return False

    def depth(self,id,node=False):
        nd = self.nodeExists(id)
        if(nd.parent!=0):
            if(node):
                node.depth+=1
                self.depth(nd.parent,node)
            else:
                nd.depth+=1
                self.depth(nd.parent,nd)
        return nd.depth

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
def f4(postId):
    return tree.depth(postId)

#LD content word (without stopwords)
def f5(post):
    post = nlp(post)
    return len(set([word.lower_ for word in post if not word.is_stop and word.lower_ not in stopwords.words('portuguese')]))/len(post)    

#average giveness
def f6(post,parent):
    post = nlp(post)
    parent = nlp(parent)
    return post.similarity(parent)

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
    post = nlp(post)
    sents = [sent for sent in post.sents]
    if(len(sents)<=1):
        return 0
    similarities = []   
    for sent in sents:
        for sent2 in sents:
            if not(sent==sent2):
                similarities.append(sent.similarity(sent2))
        sents.remove(sent)
    return np.mean(similarities)

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
    dic = pyphen.Pyphen(lang='pt-br')
    post = nlp(post)
    return np.std([len(dic.inserted(word.lower_).split('-')) for word in post])

#number os first person singular words
def f18(post):
    pp = ['eu','me','mim','comigo']
    words = word_tokenize(post)
    count = 0
    print(words)
    return sum([words.count(word) for word in pp])

#Flesch-Kincaid Grade Level
def f19(post):
    dic = pyphen.Pyphen(lang='pt-br')
    meanWordsSentence = f16(post)
    numWords = f1(post)
    post = nlp(post)
    meanSyllabe = np.sum([len(dic.inserted(word.lower_).split('-')) for word in post])
    return 248.835-(1.015*meanWordsSentence)-(84.6*(meanSyllabe/numWords))

#average of word's hypernonims
def f20(post):
    post = nlp(post)
    verbs = [word for word in post if word.pos_=="VERB"]
    sinonims = [verb for verb in verbs]
    return len(sinonims)/len(verbs)

#dataset reader
corpus = open_workbook('DBForumBrazilteste.xlsx',on_demand=True)

sheet = corpus.sheet_by_name("Sheet1")

#armazenando postagens em uma lista (posts) coluna G, linhas de 1 a 8
posts = sheet.col_slice(colx=6,start_rowx=1,end_rowx=8)
ids = sheet.col_slice(colx=0,start_rowx=1,end_rowx=8)
parentIds = sheet.col_slice(colx=1,start_rowx=1,end_rowx=8)
ids = [int(id.value) for id in ids]
parentIds = [int(id.value) for id in parentIds]

tree = Tree(ids,parentIds)

for i in range(0,len(posts)):
    post = posts[i].value
    
    #example
    # print(f15(post))
    # print(f16(post))
    # print(f19(post))