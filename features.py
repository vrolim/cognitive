from xlrd import open_workbook
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import spacy
import pyphen
import numpy as np
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

nlp = spacy.load('pt')

wn = open('layout-one.txt','r',encoding='utf-8',errors='ignore').read().split('\n')
wordnet = []
for line in wn:
    line = line.replace('[','').replace(']','').replace('{','').replace('}','').replace(',','').split(' ')[1::]
    if(line!=[]):
        category = line[0]
        words = line[1::]
        for w in words:
            if('<' in w):
                words.remove(w)
        wordnet.append(words)

def num_synonyms(word):
    count = 0
    if(type(word)!=str):
        return 0
    for wordSet in wordnet:
        if(wordSet.__contains__(word)):
            count+=len(wordSet)-1
    return count 

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
    if(post==parent):
        return 0
    post = nlp(post)
    parent = nlp(parent)
    return post.similarity(parent)

#question marks
def f7(post):
    return word_tokenize(post).count("?")

#similarity with previous message
def f8(post, previous):
    if(post==previous):
        return 0
    post = nlp(post)
    previous = nlp(previous)
    return post.similarity(previous)

#LD - VOC
def f9(post):
    # post = nlp(post)
    return f5(post)

#money-related words
def f10(post):
    money = [
    'dinheiro', 'tostão', 'ouro', 'vintém', 'prata', 'nota', 'moeda', 'níquel', 'metal', 'cobre', 'cédula', 'soma', 'numo', 'pecúnia', 'montante', 'importância', 'numerário', 'quantia', 'verba', 'trocado', 'bufunfa', 'bago', 'tutu', 'arame', 'bagarote', 'bolada', 'capim', 'grana',
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
    if(post==posterior):
        return 0
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
    if(len(verbs)==0):
        return 0
    synonyms = [num_synonyms(verb.lower_) for verb in verbs]
    return sum(synonyms)/len(verbs)

#aditional features
def aditionals(post):
    postOriginal = post.lower()
    post = nlp(post)

    above6 = len([word for word in post if len(word)>6])
    pronouns = len([word for word in post if word.pos_=='PROP' or word.pos_=='PROPN'])
    ppronouns = len([word for word in post if word.pos_== 'PROPN'])
    fpp = sum([word_tokenize(postOriginal).count(word) for word in ['nós','nos','conosco']])
    sps = sum([word_tokenize(postOriginal).count(word) for word in ['tu','te','ti','contigo']])
    tps = sum([word_tokenize(postOriginal).count(word) for word in ['ele','ela','se','si','consigo','lhe']])
    tpp = sum([word_tokenize(postOriginal).count(word) for word in ['eles','elas','se','si','consigo','lhes']])
    ipronouns = sum([word_tokenize(postOriginal).count(word) for word in ['algum','nenhum','todo','muito','pouco','vário','tanto','outro','quanto','alguma','nenhuma','toda','muita','pouca','vária','tanta','outra','quanta','alguns','nenhuns','todos','muitos','poucos','vários','tantos','outros','quantos','algumas','nenhumas','todas','muitas','poucas','várias','tantas','outras','quantas','alguém','ninguém','outrem','tudo','nada','algo','cada','qualquer','quaisquer']])
    articles = len([word for word in post if word.pos_== 'DET' and 'art' in word.tag_.lower()])    
    verbs = len([word for word in post if word.pos_== 'VERB'])
    pastTense = len([word for word in post if word.pos_== 'VERB' and 'ps' in word.tag_.lower()])
    presentTense = len([word for word in post if word.pos_== 'VERB' and 'pr' in word.tag_.lower()])
    futureTense = len([word for word in post if word.pos_== 'VERB' and 'fut' in word.tag_.lower()])
    adverbs = len([word for word in post if word.pos_== 'ADV'])
    prepositions = len([word for word in post if word.pos_== 'ADP' and 'prp' in word.tag_.lower()]) 
    conjs = len([word for word in post if word.pos_== 'CONJ' or word.pos_== 'CCONJ' or word.pos_=='SCONJ'])
    negs = sum([word_tokenize(postOriginal).count(word) for word in ['não', 'tampouco', 'nem', 'nunca', 'jamais']])
    quants = len([word for word in post if 'quant' in word.tag_.lower()])
    nums = len([word for word in post if word.pos_== 'NUM'])

    return [above6,pronouns,ppronouns,fpp,sps,tps,tpp,ipronouns,articles,verbs,pastTense,presentTense,futureTense,adverbs,prepositions,conjs,negs,quants,nums]


#dataset reader
corpus = open_workbook('DBForumBrazil-Final.xlsx',on_demand=True)

sheet = corpus.sheet_by_name("Sheet1")

posts = sheet.col_slice(colx=6,start_rowx=1,end_rowx=1501)
ids = sheet.col_slice(colx=0,start_rowx=1,end_rowx=1501)
parentIds = sheet.col_slice(colx=1,start_rowx=1,end_rowx=1501)
labels = sheet.col_slice(colx=7,start_rowx=1,end_rowx=1501)

#converting in list format
posts = [str(post.value) for post in posts]
ids = [int(id.value) for id in ids]
parentIds = [int(id.value) for id in parentIds]
labels = [int(label.value) for label in labels]
postDict = {ids[i]:posts[i] for i in range(len(posts))}

tree = Tree(ids,parentIds)

output = open('cognitive.arff','w')
output.write('''% 1. Title: Cognitive Presence
% 
% 2. Sources:
%      (a) Creators: Valter Neto, Vitor Rolim, Rafael Mello, Vitomir Kovanovic, Dragan Gasevic, Rafael Dueire and Rodrigo Lins 
%      (b) Date: April, 2018
% 
@RELATION Cognitive Presence

@ATTRIBUTE f1 NUMERIC
@ATTRIBUTE f2 NUMERIC
@ATTRIBUTE f3 NUMERIC
@ATTRIBUTE f4 NUMERIC
@ATTRIBUTE f5 NUMERIC
@ATTRIBUTE f6 NUMERIC
@ATTRIBUTE f7 NUMERIC
@ATTRIBUTE f8 NUMERIC
@ATTRIBUTE f9 NUMERIC
@ATTRIBUTE f10 NUMERIC
@ATTRIBUTE f11 NUMERIC
@ATTRIBUTE f12 NUMERIC
@ATTRIBUTE f13 NUMERIC
@ATTRIBUTE f14 NUMERIC
@ATTRIBUTE f15 NUMERIC
@ATTRIBUTE f16 NUMERIC
@ATTRIBUTE f17 NUMERIC
@ATTRIBUTE f18 NUMERIC
@ATTRIBUTE f19 NUMERIC
@ATTRIBUTE f20 NUMERIC
@ATTRIBUTE fa1 NUMERIC
@ATTRIBUTE fa2 NUMERIC
@ATTRIBUTE fa3 NUMERIC
@ATTRIBUTE fa4 NUMERIC
@ATTRIBUTE fa5 NUMERIC
@ATTRIBUTE fa6 NUMERIC
@ATTRIBUTE fa7 NUMERIC
@ATTRIBUTE fa8 NUMERIC
@ATTRIBUTE fa9 NUMERIC
@ATTRIBUTE fa10 NUMERIC
@ATTRIBUTE fa11 NUMERIC
@ATTRIBUTE fa12 NUMERIC
@ATTRIBUTE fa13 NUMERIC
@ATTRIBUTE fa14 NUMERIC
@ATTRIBUTE fa15 NUMERIC
@ATTRIBUTE fa16 NUMERIC
@ATTRIBUTE fa17 NUMERIC
@ATTRIBUTE fa18 NUMERIC
@ATTRIBUTE fa19 NUMERIC   

@ATTRIBUTE class {0,1,2,3,4}

@DATA
''')

for i in range(len(posts)):
    print(i)
    post = posts[i]
    previous = i if i-1<0 else i-1
    posterior = i if i+1>len(posts)-1 else i+1
    try:
        parent = postDict[parentIds[i]]
    except:
        parent = post
    p1 = round(float(f1(post)),3)
    p2 = round(float(f2(post)),3)
    p3 = round(float(f3(post)),3)
    p4 = round(float(f4(ids[i])),3)
    p5 = round(float(f5(post)),3)
    p6 = round(float(f6(post,parent)),3)
    p7 = round(float(f7(post)),3)
    p8 = round(float(f8(post,posts[previous])),3)
    p9 = round(float(f9(post)),3)
    p10 = round(float(f10(post)),3)
    p11 = round(float(f11(post)),3)
    p12 = round(float(f12(post,posts[posterior])),3)
    p13 = round(float(f13(i,parentIds)),3)
    p14 = round(float(f14(post)),3)
    p15 = round(float(f15(post)),3)
    p16 = round(float(f16(post)),3)
    p17 = round(float(f17(post)),3)
    p18 = round(float(f18(post)),3)
    p19 = round(float(f19(post)),3)
    p20 = round(float(f20(post)),3)
    padd = aditionals(post)
    features = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20]+padd+[labels[i]]
    features = str(features).replace('[','').replace(']','')
    #writing arff file
    output.write(features)
    output.write("\n")

output.close()



# import pickle
# pickle.dump(x,open("train.pickle", "wb"))
