from xlrd import open_workbook
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import spacy
import pyphen
import numpy as np
from sklearn import svm, cross_validation
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

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
    post = nlp(post)
    return 0

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
    sinonims = [verb for verb in verbs]
    return len(sinonims)/len(verbs)

#aditional features
def aditionals(post):
    postOriginal = post
    post = nlp(post)

    above6 = len([word for word in post if len(word)>6])
    pronouns = len([word for word in post if word.pos_=='PROP' or word.pos_=='PROPN'])
    ppronouns = len([word for word in post if word.pos_== 'PROPN'])
    fpp = sum([word_tokenize(postOriginal).count(word) for word in ['nós','nos','conosco']])
    sps = sum([word_tokenize(postOriginal).count(word) for word in ['tu','te','ti','contigo']])
    tps = sum([word_tokenize(postOriginal).count(word) for word in ['ele','ela','se','si','consigo','lhe']])
    tpp = sum([word_tokenize(postOriginal).count(word) for word in ['eles','elas','se','si','consigo','lhes']])
    pi = sum([word_tokenize(postOriginal).count(word) for word in ['algum','nenhum','todo','muito','pouco','vário','tanto','outro','quanto','alguma','nenhuma','toda','muita','pouca','vária','tanta','outra','quanta','alguns','nenhuns','todos','muitos','poucos','vários','tantos','outros','quantos','algumas','nenhumas','todas','muitas','poucas','várias','tantas','outras','quantas','alguém','ninguém','outrem','tudo','nada','algo','cada','qualquer','quaisquer']])
    articles = len([word for word in post if word.pos_== 'DET' and 'art' in word.tag_.lower()])    
    verbs = len([word for word in post if word.pos_== 'VERB'])
    adverbs = len([word for word in post if word.pos_== 'ADV'])
    prepositions = len([word for word in post if word.pos_== 'ADP' and 'prp' in word.tag_.lower()]) 

def train(classifier, X, y, class_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
   
    ##TREINANDO ALGORITMO"
    classifier.fit(X_train, y_train)
    
    ##Predicoes para medição da Acurácia"
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print("Accuracy: %s" % classifier.score(X_test,y_test))
    print("F-measure: %s" % str(f1_score(y_test, y_pred, average=None )))
    print("Recall: %s"% str(recall_score(y_test, y_pred, average=None)))
    print("Precision: %s" % str(precision_score(y_test, y_pred, average=None)))
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Matriz de Confusao')
    plt.show()
    return classifier

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
y = [int(label.value) for label in labels]
postDict = {ids[i]:posts[i] for i in range(len(posts))}

tree = Tree(ids,parentIds)

x = []
for i in range(len(posts)):
    print(i)
    post = posts[i]
    previous = i if i-1<0 else i-1
    posterior = i if i+1>len(posts)-1 else i+1
    try:
        parent = postDict[parentIds[i]]
    except:
        parent = post
    p1 = f1(post)
    p2 = f2(post)
    p3 = f3(post)
    p4 = f4(ids[i])
    p5 = f5(post)
    p6 = f6(post,parent)
    p7 = f7(post)
    p8 = f8(post,posts[previous])
    p9 = f9(post)
    p10 = f10(post)
    p11 = f11(post)
    p12 = f12(post,posts[posterior])
    p13 = f13(i,parentIds)
    p14 = f14(post)
    p15 = f15(post)
    p16 = f16(post)
    p17 = f17(post)
    p18 = f18(post)
    p19 = f19(post)
    # p20 = f20(post)
    # padd = aditionals(post)
    x.append([float(p1),float(p2),float(p3),float(p4),float(p5),float(p6),float(p7),float(p8),float(p9),float(p10),float(p11),float(p12),float(p13),float(p14),float(p15),float(p16),float(p17),float(p18),float(p19)])    
import pickle
pickle.dump(x,open("train.pickle", "wb"))

train(svm.SVC(kernel='linear',C=1.0),x,y,[0,1,2,3,4])