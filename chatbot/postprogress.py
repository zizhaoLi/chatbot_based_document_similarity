import nltk
import string, pprint, os, sys, csv, logging, multiprocessing, itertools

import numpy

from operator import itemgetter
from collections import Counter, OrderedDict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from gensim import corpora, models, similarities
from gensim.summarization import bm25

TBBT_DIR = "../data/texts/"
IMDB_DIR = "../corpus/IMDB/"


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_sentences(file):
    with open(file, 'r', encoding='latin-1') as d:
        text = d.read()
        tmp = nltk.sent_tokenize(text)
        sentences = [clean_text(sent) for sent in tmp]
    return sentences

def get_original_sentences(file):
    with open(file, 'r', encoding='latin-1') as d:
        text = d.read()
        original_sentences = text.split(".")
    return original_sentences

def get_tokens(file):
    with open(file, 'r', encoding='latin-1') as d:
        text = d.read()
        tokens = nltk.word_tokenize(clean_text(text))
    return tokens


def get_most_common_tokens(tokens, num):
    count = Counter(tokens)
    return count.most_common(num)


def get_pos_tag(tokens):
    tags = nltk.pos_tag(tokens)
    return tags


def get_lemma(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas


def get_stem(tokens):
    porter_stemmer = PorterStemmer()
    stems = [porter_stemmer.stem(token) for token in tokens]
    return stems


def remove_stop_word(tokens, gensim=False):
    normalized_tokens = [token for token in tokens if token not in stopwords.words('english')]
    if gensim:
        return normalized_tokens
    else:
        normalized_sentence = " ".join(normalized_tokens)
        return normalized_sentence


def remove_stop_tag(token_tags, gensim=False):
    stop_tag = ['IN', 'DT', 'CC', 'TO', 'PRP', 'MD', 'WDT', 'WP', 'RP', 'EX', 'PDT', 'WP$', 'UH']
    normalized_tags = [token for token, tag in token_tags if tag not in stop_tag]
    if gensim:
        return normalized_tags
    else:
        normalized_sentence = " ".join(normalized_tags)
        return normalized_sentence


def clean_text(text):
    lowers = text.lower()  # lower case for everyone
    # remove the punctuation using the character deletion step of translate
    punct_killer = str.maketrans('', '', string.punctuation)
    no_punctuation = lowers.translate(punct_killer)
    return no_punctuation


def get_text(file):
    with open(file, 'r', encoding='latin-1') as d:
        text = d.read()
    return clean_text(text)


def tokenize(text):
    return nltk.word_tokenize(text)


def create_tfidf(dir):
    text_list, text_names = [], []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".txt"):
                # print("treating "+file)
                file_path = subdir + os.path.sep + file
                text_list.append(get_text(file_path))
                text_names.append(file_path)

    return text_list, text_names


# def tf(tokens, token):
#     count = Counter(tokens)
#     tf = count[token] / sum(count.values())
#     return tf


def get_cosine_similarity(text1, text2, v):
    t1 = v.transform([text1])
    t2 = v.transform([text2])
    return cosine_similarity(t1, t2)
    # return entropy(t1.toarray()[0,:],t2.toarray()[0,:])


# def get_euclidean_distances(text1, text2, v):
#     t1 = v.transform([text1])
#     t2 = v.transform([text2])
#     return euclidean_distances(t1, t2)
#
#
# def get_jaccard_similarity(query, document):
#     intersection = set(query).intersection(set(document))
#     union = set(query).union(set(document))
#     return len(intersection) / len(union)

def get_similarity_gensim(model, num_feature, query, d2v=False):
    if d2v:
        return doc2vec.docvecs.most_similar([doc2vec.infer_vector(query)], topn=len(doc2vec.docvecs))
    else:
        index = similarities.MatrixSimilarity(model, num_features=num_feature)
        # similarities.Similarity(model)
        # index = similarities.SparseMatrixSimilarity(model, num_features=12)
        # print(list(index)[0])
        # similarity with each text theme
        sims = index[query]
        sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        return sims


def get_query():
    query = input("Please input your query: ")
    query = remove_stop_word(tokenize(clean_text(query)))
    return query


def get_query_gensim(query):
    # query = input("Please input your query: ")
    # remove stop word from request
    query = remove_stop_word(tokenize(clean_text(query)), gensim=True)
    # print(query)
    # remove unimportant tag from query
    # query = remove_stop_tag(get_pos_tag(tokenize(clean_text(query))), gensim=True)
    return query



# def expand_morpho(morphology, tokens):
#     lemmas = get_lemma(tokens)
#     lemma_morpho = {lemma: morpho for morpho in morphology for lemma in lemmas if lemma in morpho}
#     return lemma_morpho


# def expand_syno(synonym, tokens):
#     lemmas = get_lemma(tokens)
#     lemma_syno = {lemma: syno for syno in synonym for lemma in lemmas if lemma in syno}
#     return lemma_syno


# -----A tfidf model using sklearn----- #
text_list, text_names = create_tfidf(TBBT_DIR)
sentences = get_original_sentences("../data/all.txt")

text_list_imdb, text_names_imdb = create_tfidf(IMDB_DIR)

texts = {file: get_text(file) for file in text_names}

# eval_sk = {}
#
# v = TfidfVectorizer(encoding='latin-1', tokenizer=tokenize, stop_words='english')
# tfidf = v.fit_transform(text_list)
#
# while 1:
#     query = get_query()
#
#     # look for reply per scene, than per sentence with fixed scene
#     similarity_text = {text_file: get_cosine_similarity(query, txt, v)[0][0] for text_file, txt in texts.items()}
#     sorted_similarity_text = sorted(similarity_text.items(), key=lambda x: x[1], reverse=True)
#
#     sentences = get_sentences(sorted_similarity_text[0][0])
#
#     similarity_sentences = {sent: get_cosine_similarity(query, sent, v)[0][0] for sent in sentences}
#     sorted_similarity_sentences = sorted(similarity_sentences.items(), key=lambda x: x[1], reverse=True)
#
#     print("possible answers of system A are: ")
#     print(sorted_similarity_sentences[0:5])
#     print("the reply of system A is: " + sorted_similarity_sentences[0][0])
#
#     # look for reply in the whole text
#     similarity_sentences = {sent: get_cosine_similarity(query, sent, v)[0][0] for sent in sentences}
#     sorted_similarity_sentences = sorted(similarity_sentences.items(), key=lambda x: x[1], reverse=True)
#
#     print("possible answers of system B are: ")
#     print(sorted_similarity_sentences[0:5])
#     print("the reply of system B is: " + sorted_similarity_sentences[0][0])
#
#     while 1:
#         eval_tmp = input("\nWhich system you think is better(A, B, 0, 1 where O means neither, 1 means both)? ")
#         if eval_tmp in ['A', 'B', '0', '1']:
#             eval_sk[query] = eval_tmp
#             break

# print(eval)

## remove words appear only once
# all_stems = sum(texts_stemmed, [])
# stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
# texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]


'''
# -----A tfidf model using gensim----- #
corpus = [remove_stop_word(tokenize(line), gensim=True) for line in sentences]
dictionary = corpora.Dictionary(corpus)
# word frequence tf
doc_vectors = [dictionary.doc2bow(text) for text in corpus]

# print(corpus)
# print(dictionary)
# print(dictionary.token2id)
# print(doc_vectors[0])

tfidf = models.TfidfModel(doc_vectors)
tfidf_vectors = tfidf[doc_vectors]
# print(tfidf_vectors)
# print(tfidf_vectors[0])

# -----A bm25 model using gensim----- #
bm25Model = bm25.BM25(corpus)
average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())

# -----A lsi model using gensim----- #
corpus_scene = [remove_stop_word(tokenize(line), gensim=True) for line in text_list]
dictionary_scene = corpora.Dictionary(corpus_scene)
doc_vectors_scene = [dictionary_scene.doc2bow(text) for text in corpus_scene]

tfidf_scene = models.TfidfModel(doc_vectors_scene)
tfidf_vectors_scene = tfidf_scene[doc_vectors_scene]

lsi = models.LsiModel(tfidf_vectors_scene, id2word=dictionary_scene, num_topics=12)
# topic weighted num_document*num_topics
lsi_vectors = lsi[tfidf_vectors_scene]
# print(len(lsi_vectors), lsi_vectors[0:2])
# for vec in lsi_vectors[0:2]:
#     print(vec)

# -----A lda model using gensim----- #
lda = models.LdaModel(doc_vectors_scene, id2word=dictionary_scene, num_topics=12, iterations=300)
lda_vectors = lda[doc_vectors_scene]
# for vec in lda_vectors[0:2]:
#     print(vec)
# # topic info
# print(lda.print_topics(12))
'''
print('haha')
# -----A doc2vec model using gensim----- #
# sentences = PathLineSentences(os.getcwd() + '\corpus\')
# corpus_doc2vec = [models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
corpus_doc2vec = [models.doc2vec.TaggedDocument(remove_stop_word(tokenize(line), gensim=True), [i]) for i, line in enumerate(text_list_imdb)]
print('haha')

doc2vec = models.Doc2Vec(size=24, dm=0, min_count=2, workers=multiprocessing.cpu_count(), iter=1000)
# Build a Vocabulary
doc2vec.build_vocab(corpus_doc2vec)
print('haha')

doc2vec.save('doc2vec')
# doc2vec = models.Doc2Vec.load('doc2vec')

# print(doc2vec)
# print(len(doc2vec.wv.vocab))
# print(doc2vec.wv.vocab['so'].count)
# size: parameter size * 1
# a = doc2vec.infer_vector(['only', 'you', 'can', 'prevent', 'forrest', 'fires'])
# print(len(doc2vec.docvecs))

# self-similarity
ranks = []
# second_ranks = []
for doc_id in range(len(corpus_doc2vec)):
    inferred_vector = doc2vec.infer_vector(corpus_doc2vec[doc_id].words)
    sims = get_similarity_gensim(doc2vec, None, corpus_doc2vec[doc_id].words, d2v=True)
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    # second_ranks.append(sims[1])

print(Counter(ranks))


# # # Random Projections
# # model = models.RpModel(tfidf_vectors, num_topics=500)
# # # Hierarchical Dirichlet Process
# # model = models.HdpModel(doc_vectors, id2word=dictionary)
#
# # Pick a random document from the test corpus and infer a vector from the doc2vec
# # doc_id = random.randint(0, len(test_corpus))
# # inferred_vector = doc2vec.infer_vector(test_corpus[doc_id])
# # sims = doc2vec.docvecs.most_similar([inferred_vector], topn=len(doc2vec.docvecs))
