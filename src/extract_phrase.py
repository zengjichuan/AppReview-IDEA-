import logging
from gensim.models.phrases import Phrases
import os
import itertools
from extractSentenceWords import *

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)


def build_input(filename):
    doc_sent_word = []
    num_words = 0
    num_docs = 0
    with open(filename) as fin:
        for line in fin.readlines():
            words_sents, wc = extractSentenceWords(line, lemma=True)
            doc_sent_word.append(words_sents)
            num_docs += 1
            num_words += wc
    print("Read %d docs, %d words!" % (num_docs, num_words))
    return doc_sent_word


def extract_phrases(filename, min_count):
    rst = build_input(filename)
    gen = list(itertools.chain.from_iterable(rst))
    bigram = Phrases(gen, threshold=5, min_count=min_count)
    trigram = Phrases(bigram[gen], threshold=2, min_count=2)
    # write
    with open('data/phrases_%d_%s' % (min_count, os.path.basename(filename)), 'wb') as fout:
        ph_dic = {}
        for phrase, score in bigram.export_phrases(gen):
            ph_dic[phrase] = score
        for phrase, score in trigram.export_phrases(bigram[gen]):
            ph_dic[phrase] = score
        for phrase, score in ph_dic.items():
            if re.search(r'\d+', phrase):  # remove digits
                continue
            phrase = b"_".join(phrase.split(b' '))
            fout.write(phrase + b'\n')


if __name__ == '__main__':
    extract_phrases('data/adReview', 10)
