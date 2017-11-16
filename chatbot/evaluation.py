import postprogress as pp
import itertools
# from collections import Counter

def compute_ROS(target):
    numBetter = 0
    numBothGood = 0
    for answer in eva:
        if answer.find(target) == -1:
            continue
        else:
            if answer.find('a') == 0:
                numBothGood = numBothGood + 1
                continue
            else:
                numBetter = numBetter + 1
                continue

    return (numBetter + numBothGood) * 1.0 / N


if __name__ == '__main__':

    # read and save 50 queries in list
    query_list = []
    query_file = "../data/query/query.txt"
    with open(query_file, 'r') as f:
        while 1:
            line = f.readline()
            if not line:
                break
            else:
                query_list.append(line[:-1])

    eva = []
    N = len(query_list)

    for q in query_list:
        query = pp.get_query_gensim(q)
        # system A
        # look for the most related sentence using tfidf
        query_bow = pp.dictionary.doc2bow(query)

        # sorted similarity with each sentence 1*num_sent
        sims_tfidf = pp.get_similarity_gensim(pp.tfidf_vectors, len(pp.dictionary), query_bow)
        print(sims_tfidf[0:5])
        #print("the reply of system A (tfidf) is: " + pp.sentences[sims_tfidf[0][0]])
        reply_A = pp.sentences[sims_tfidf[0][0]]

        # system B
        # look for the most related sentence using bm25
        scores = pp.bm25Model.get_scores(query, pp.average_idf)
        max_score_idx = scores.index(max(scores))
        #print("the reply of system B (bm25) is: " + pp.sentences[max_score_idx])
        reply_B = pp.sentences[max_score_idx]

        # system C
        # look for the most related text using lsi
        query_lsi = pp.lsi[query_bow]
        # sorted similarity with each scene 1*115
        sims_lsi = pp.get_similarity_gensim(pp.lsi_vectors, 12, query_lsi)

        # look for the most related sentence using tfidf
        sentences_lsi = pp.get_sentences(pp.text_names[sims_lsi[0][0]])

        corpus_lsi = [pp.remove_stop_word(pp.tokenize(line), gensim=True) for line in sentences_lsi]
        dictionary_lsi = pp.corpora.Dictionary(corpus_lsi)
        doc_vectors_lsi = [dictionary_lsi.doc2bow(text) for text in corpus_lsi]

        tfidf_vectors_lsi = pp.tfidf[doc_vectors_lsi]

        query_bow_lsi = dictionary_lsi.doc2bow(query)

        sims_lsi_scene = pp.get_similarity_gensim(tfidf_vectors_lsi, len(dictionary_lsi), query_bow_lsi)
        print(sims_lsi_scene[0:5])
        #print("the reply of system C (lsi) is: " + sentences_lsi[sims_lsi_scene[0][0]])
        reply_C = sentences_lsi[sims_lsi_scene[0][0]]

        # system D
        # look for the most related text using lda
        query_lda = pp.lda[query_bow]
        sims_lda = pp.get_similarity_gensim(pp.lda_vectors, 12, query_lda)

        # look for the most related sentence using tfidf
        sentences_lda = pp.get_sentences(pp.text_names[sims_lda[0][0]])

        corpus_lda = [pp.remove_stop_word(pp.tokenize(line), gensim=True) for line in sentences_lda]
        dictionary_lda = pp.corpora.Dictionary(corpus_lda)
        doc_vectors_lda = [dictionary_lda.doc2bow(text) for text in corpus_lda]

        tfidf_vectors_lda = pp.tfidf[doc_vectors_lda]

        query_bow_lda = dictionary_lda.doc2bow(query)

        sims_lda_scene = pp.get_similarity_gensim(tfidf_vectors_lda, len(dictionary_lda), query_bow_lda)
        print(sims_lda_scene[0:5])
        #print("the reply of system D (lda) is: " + sentences_lda[sims_lda_scene[0][0]])
        reply_D = sentences_lda[sims_lda_scene[0][0]]

        # system E
        # look for the most related sentence using doc2vec
        sims_doc2vec = pp.get_similarity_gensim(pp.doc2vec, None, query, d2v=True)
        print(sims_doc2vec[0:5])
        # print("the reply of system E (doc2vec) is: " + pp.sentences[sims_doc2vec[0][0]])
        # print('Document ({}): «{}»\n'.format(sims_doc2vec[0][0], ' '.join(corpus_doc2vec[sims_doc2vec[0][0]].words)))
        reply_E = pp.sentences[sims_doc2vec[0][0]]

        print("The query is: " + q)
        print("the reply of system A (tfidf) is: " + reply_A)
        print("the reply of system B (bm25) is: " + reply_B)
        print("the reply of system C (lsi) is: " + reply_C)
        print("the reply of system D (lda) is: " + reply_D)
        print("the reply of system E (doc2vec) is: " + reply_E)

        eval_possible = []
        [eval_possible.extend([''.join(i) for i in itertools.permutations('abcde', j)]) for j in range(1, 6)]


        while 1:
            answer = input("Which reply is better in your opinion? Answer like ABC,ac,ed: ")
            answer = answer.strip().lower()

            if answer in eval_possible:
                eva.append(answer)
                break
            else:
                print("Wrong answer, answer again.")

    # model A is the baseline, we use ROS = (NumOfTargetBetter + NumOfBothGood) / N

    ROS_B = compute_ROS('b')
    print("The ROS of system B is: " + str(ROS_B))

    ROS_C = compute_ROS('c')
    print("The ROS of system C is: " + str(ROS_C))

    ROS_D = compute_ROS('d')
    print("The ROS of system D is: " + str(ROS_D))

    ROS_E = compute_ROS('e')
    print("The ROS of system E is: " + str(ROS_E))