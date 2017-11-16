#! /Users/xiaozi/anaconda2/envs/python36/bin/python3.6
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd

import os, string

'''
dir = "../corpus/TBBT/"
trans_dir = "NE_1st2nd3rd/"
scene_dir = "scenes/"
s3_dir = "S3/"
res_dir = "../data/"


if __name__ == '__main__':
    # -----Read the time information of the scene of all episode, it's a list of list of tuple----- #
    # [
    # [(episode1_scene_1_start, episode1_scene1_end), (episode1_scene_2_start, episode1_scene2_end)],
    # [(episode2_scene_1_start, episode2_scene1_end), (episode2_scene_2_start, episode2_scene2_end)]
    # ]
    time_list = []
    for i in range(12):
        scene_file = dir + scene_dir + "tbbt.season01.episode" + str(i + 1).zfill(2) + ".scenes.txt"
        with open(scene_file, 'r') as f:
            tmp = []
            while 1:
                line = f.readline()
                if not line:
                    break
                else:
                    line = line[:-1].split(' ')
                    tmp.append((line[1], line[2]))
            time_list.append(tmp)
    # print(time_list)

    # -----Read the translation of all episode, like a list of DataFrame----- #
    # -----Write the translation in res_dir in order of scenes----- #
    # [DF1, DF2, DF3]
    for i in range(12):
        trans_file = dir + trans_dir + "TheBigBangTheory.Season01.Episode" + str(i + 1).zfill(2) + ".speakername.ctm"
        # print(trans_file)
        bigBang_tmp = pd.read_csv(trans_file, sep=' ', usecols=[2, 3, 4, 5, 7], names=['temps_debut_mot', 'duree_mot', 'mot', 'mesure_confiance_alignement', 'nom_locuteur'], quoting=3, error_bad_lines=False, dtype={"temps_debut_mot": np.float64, 'duree_mot': np.float64})
        # print(bigBang_tmp.head())
        for j in range(len(time_list[i])):
            scene_start = bigBang_tmp[bigBang_tmp['temps_debut_mot'] == float(time_list[i][j][0])].index.tolist()[0]
            scene_stop = bigBang_tmp[(bigBang_tmp['temps_debut_mot'] + bigBang_tmp['duree_mot']) == float(time_list[i][j][1])].index.tolist()[-1]
            text = ' '.join(np.array(bigBang_tmp['mot'][scene_start: scene_stop+1]))

            data_file = res_dir + "texts/" + "TheBigBangTheory.Season01.Episode" + str(i + 1).zfill(2) + ".Scene" + str(j + 1).zfill(2) + ".txt"
            with open(data_file, 'w', encoding='latin-1') as f:
                f.writelines(text + '\n')

            # print(scene_start, scene_stop)
            # print(text)

    # -----Extract the corpus(season 1 and season 3) and save it in res_dir/all.txt----- #
    text = []
    for i in range(17):
        trans_file = dir + trans_dir + "TheBigBangTheory.Season01.Episode" + str(i + 1).zfill(2) + ".speakername.ctm"
        bigBang_tmp = pd.read_csv(trans_file, sep=' ', usecols=[4], names=['word'], quoting=3, error_bad_lines=False)
        text.extend(np.array(bigBang_tmp['word']))

    text = " ".join(text)

    for i in range(5):
        s3_file = dir + s3_dir + "tbbts03e" + str(i+1).zfill(2) + ".txt"
        with open(s3_file, 'r', encoding='latin-1') as f:
            s3_text = f.readlines()

    s3_text = " ".join([line.strip() for line in s3_text])

    text = text + s3_text

    with open(res_dir + "all.txt", 'w', encoding='latin-1') as f:
        f.write(text + '\n')

'''

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


dir = "../corpus/IMDB/"
text_list, text_names = create_tfidf(dir)
print(text_list[0:5])
