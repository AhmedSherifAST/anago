import gensim
import re
import numpy as np
from nltk import ngrams
from itertools import chain
import codecs




# check if word is an English or not
def pureEnglish(word):
    for eachIndex in range(len(word)):
        StringUnicode=ord(word[eachIndex])
        if not (StringUnicode>=65 and StringUnicode<=122):
            return False
    return True




# =========================
# ==== Helper Methods =====

# Clean/Normalize Arabic Text


def writeTupleArray(x_test_org,y_pred,fileToWrite="experment.txt"):

    fileWriteName = open(fileToWrite, "w", encoding='utf-8')
    for index in range(len(x_test_org)):
        for eachIndex in range(len(x_test_org[index])):
            fileWriteName.write(x_test_org[index][eachIndex]+" "+y_pred[index][eachIndex]+"\n")










def clean_str(text):
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t',
              '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ',
               ' ! ']

    # remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", text)

    # remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)

    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    # trim
    text = text.strip()

    return text


def get_vec(n_model, dim, token):
    vec = np.zeros(dim)
    is_vec = False
    if token not in n_model.wv:
        _count = 0
        is_vec = True
        for w in token.split("_"):
            if w in n_model.wv:
                _count += 1
                vec += n_model.wv[w]
        if _count > 0:
            vec = vec / _count
    else:
        vec = n_model.wv[token]
    return vec


def calc_vec(pos_tokens, neg_tokens, n_model, dim):
    vec = np.zeros(dim)
    for p in pos_tokens:
        vec += get_vec(n_model, dim, p)
    for n in neg_tokens:
        vec -= get_vec(n_model, dim, n)

    return vec


## -- Retrieve all ngrams for a text in between a specific range
def get_all_ngrams(text, nrange=3):
    text = re.sub(r'[\,\.\;\(\)\[\]\_\+\#\@\!\?\؟\^]', ' ', text)
    tokens = [token for token in text.split(" ") if token.strip() != ""]
    ngs = []
    for n in range(2, nrange + 1):
        ngs += [ng for ng in ngrams(tokens, n)]
    return ["_".join(ng) for ng in ngs if len(ng) > 0]


## -- Retrieve all ngrams for a text in a specific n
def get_ngrams(text, n=2):
    text = re.sub(r'[\,\.\;\(\)\[\]\_\+\#\@\!\?\؟\^]', ' ', text)
    tokens = [token for token in text.split(" ") if token.strip() != ""]
    ngs = [ng for ng in ngrams(tokens, n)]
    return ["_".join(ng) for ng in ngs if len(ng) > 0]


## -- filter the existed tokens in a specific model
def get_existed_tokens(tokens, n_model):
    return [tok for tok in tokens if tok in n_model.wv]


def checkerLen(x_test_org,y_pred):
    if (len(x_test_org) == len(y_pred)):
        for index in range(len(x_test_org)):
            if (len(x_test_org[index]) != len(y_pred[index])):
                print("Invalid")
    else:
        print("Invalid test size of words and labels")




def vectorSim(t_model,t_en_model,word,ratioSimilarity,topn):

    token = clean_str(word)
    simArray=[]
    if (token in t_model.wv.vocab and not pureEnglish(token)):

        most_similar = t_model.wv.most_similar(token, topn=topn)
        for term, score in most_similar:
            term = clean_str(term).replace(" ", "_")
            termArray=term.split("_")
            if(score>=ratioSimilarity):
                #print(term, score)
                for TA in termArray:
                    simArray.append(TA)
        if(token in simArray):
            simArray.remove(token)
        return simArray

    elif(token in t_en_model.wv.vocab and pureEnglish(token) ):
            most_similar = t_model.wv.most_similar(token, topn=topn)
            for term, score in most_similar:
                term = clean_str(term).replace(" ", "_")
                termArray = term.split("_")
                if (score >= ratioSimilarity):
                    # print(term, score)
                    for TA in termArray:
                        simArray.append(TA)
            if (token in simArray):
                simArray.remove(token)
            return simArray
    else:
        return simArray

def getAllPredTags(x_test_org,y_pred):

    tupleArray=[]
    PERS=[]
    LOC=[]
    ORG=[]
    MISC=[]

    for eachIndex in range(len(y_pred)):
        for index in range(len(y_pred[eachIndex])):
            if(y_pred[eachIndex][index]!="O" ):
                tag=y_pred[eachIndex][index]
                word=x_test_org[eachIndex][index]
                tag=tag[2:]
                if(tag.lower()=="pers"):
                    PERS.append(word)

                if(tag.lower() == "org"):
                    ORG.append(word)

                if(tag.lower() == "loc"):
                    LOC.append(word)

                if(tag.lower() == "misc"):
                    MISC.append(word)

    persArray=(PERS,"PERS")
    orgArray=(ORG,"ORG")
    locArray=(LOC,"LOC")
    miscArray=(MISC,"MISC")
    tupleArray.append(persArray)
    tupleArray.append(orgArray)
    tupleArray.append(locArray)
    tupleArray.append(miscArray)

    return tupleArray






def AdjustPredTag(t_model,t_en_model,x_test_org,y_pred,ratioSimilarity,topn):

    tupleArray=getAllPredTags(x_test_org,y_pred)
    array=[]
    allTagArray=[]
    for element in tupleArray:
        wordArray=element[0]
        tag=element[1]
        for eachWord in wordArray:
            simArray=vectorSim(t_model,t_en_model,eachWord,ratioSimilarity,topn)
            if(len(simArray)>0):
                array.append(simArray)

        flatten_array = list(chain.from_iterable(array))
        allTagArray.append((flatten_array,tag))
        array=[]


    for i in range(len(x_test_org)):
        for j in range(len(x_test_org[i])):

            if("pad" in y_pred[i][j]):
                y_pred="O"

            for tokenArray,label in allTagArray:
                if(x_test_org[i][j] in tokenArray):
                    if("pad" in label):
                        print("pad is here")
                    y_pred[i][j]=label+" "+"mod"







# ============================
# ====== N-Grams Models ======

# t_model = gensim.models.Word2Vec.load('/home/ahmed/PycharmProjects/BachelorProject/CheckerDataDir/full_grams_sg_100_twitter/full_grams_sg_100_twitter.mdl')
#
# # python 3.X
# token = clean_str(u'ابو تريكه').replace(" ", "_")
# # python 2.7
# # token = clean_str(u'ابو تريكه'.decode('utf8', errors='ignore')).replace(" ", "_")
#
# if token in t_model.wv:
#     most_similar = t_model.wv.most_similar(token, topn=10)
#     for term, score in most_similar:
#         term = clean_str(term).replace(" ", "_")
#         if term != token:
#             print(term, score)

# تريكه 0.752911388874054
# حسام_غالي 0.7516342401504517
# وائل_جمعه 0.7244222164154053
# وليد_سليمان 0.7177559733390808
# ...

# =========================================
# == Get the most similar tokens to a compound query
# most similar to
# عمرو دياب + الخليج - مصر

# pos_tokens = [clean_str(t.strip()).replace(" ", "_") for t in ['عمرو دياب', 'الخليج'] if t.strip() != ""]
# neg_tokens = [clean_str(t.strip()).replace(" ", "_") for t in ['مصر'] if t.strip() != ""]
#
# vec = calc_vec(pos_tokens=pos_tokens, neg_tokens=neg_tokens, n_model=t_model, dim=t_model.vector_size)
#
# most_sims = t_model.wv.similar_by_vector(vec, topn=10)
# for term, score in most_sims:
#     if term not in pos_tokens + neg_tokens:
#         print(term, score)

# راشد_الماجد 0.7094649076461792
# ماجد_المهندس 0.6979793906211853
# عبدالله_رويشد 0.6942606568336487
# ...

# ====================
# ====================


# ==============================
# ====== Uni-Grams Models ======

#t_model = gensim.models.Word2Vec.load('/home/ahmed/PycharmProjects/BachelorProject/CheckerDataDir/full_grams_sg_100_twitter/full_grams_sg_100_twitter.mdl')
from gensim.models import KeyedVectors
#embeddings = KeyedVectors.load_word2vec_format(link to the .vec file)

#import gensim.models.wrappers.fasttext
#embeddings = gensim.models.wrappers.fasttext.FastTextKeyedVectors.load_word2vec_format(emb_path+emb_file, binary=False, encoding='utf8').wv
#t_model = KeyedVectors.load_word2vec_format("cc.ar.300.vec")
#t_model = gensim.models.Word2Vec.load('/home/ahmed/PycharmProjects/BachelorProject/CheckerDataDir/full_grams_sg_100_twitter/full_grams_sg_100_twitter.mdl')

#print("here")
# python 3.X
#token = clean_str('تونس')
# python 2.7
# token = clean_str('تونس'.decode('utf8', errors='ignore'))

#most_similar = t_model.wv.most_similar(token, topn=40)
#for term, score in most_similar:
    #print(term, score)

# ليبيا 0.8864325284957886
# الجزائر 0.8783721327781677
# السودان 0.8573237061500549
# مصر 0.8277812600135803
# ...


# get a word vector
#word_vector = t_model.wv[token]