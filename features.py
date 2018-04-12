from xlrd import open_workbook
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import spacy
nlp = spacy.load('pt')

def f1(post):
    return len([word for word in word_tokenize(post) if word not in string.punctuation+"\..."])

def f2(post):
    post = nlp(post)
    return len(post.ents)

def f3(post):
    post = nlp(post)
    print(set([word.tag_ for word in post]))
    return len(set([word.tag_ for word in post]))

def f4(post):
    return 0

def f5(post):
    post = nlp(post)
    return len(set([word.tag_ for word in post if not word.is_stop and word.lower_ not in stopwords.words('portuguese')]))    

def f6(post):
    return 0

def f7(post):
    return word_tokenize(post).count("?")

def f8(post):
    return 0
#abre planilhas
corpus = open_workbook('DBForumBrazilteste.xlsx',on_demand=True)

# acessa a planilha Planilha1
planilha1 = corpus.sheet_by_name("Sheet1")

#armazenando postagens em uma lista (posts) coluna G, linhas de 1 a 8
posts = planilha1.col_slice(colx=6,start_rowx=0,end_rowx=8)


for i in range(0,len(posts)):
    post = posts[i].value
    #f1
    print(f1(post))
    #f2
    print(f2(post))
    #f3
    print(f3(post))
    #f4
    print(f4(post))
    #f5
    print(f5(post))
    # #f6
    # print(f6(post))
    # #f7
    # print(f7(post))
    # #f8
    # print(f8(post))
    # #f9
    # print(f9(post))
    # #f10 
    # print(f10(post))