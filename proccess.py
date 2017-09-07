import nltk
import codecs
import syllables_en
import numpy
import argparse
from nltk.tag import StanfordNERTagger
import os


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str,
                    help='pos | ner | syllable | wiki | google')
parser.add_argument('-combine', default=[], nargs='+', type=str,
                    help='.')

opt = parser.parse_args()
print(opt)

train_src = 'PWKP_108016.tag.80.aner.ori.train.src'
valid_src = 'PWKP_108016.tag.80.aner.ori.valid.src'
test_src = 'PWKP_108016.tag.80.aner.ori.test.src'
# train_dst_src = 'PWKP_108016.tag.80.aner.ori.train.dst'
# valid_dst_src = 'PWKP_108016.tag.80.aner.ori.valid.dst'
src_files = [train_src, valid_src, test_src]

if opt.mode == 'ner':
    stanford_classifier = os.environ.get('STANFORD_MODELS').split(':')[0]
    stanford_ner_path = os.environ.get('CLASSPATH').split(':')[0]
    st = StanfordNERTagger(stanford_classifier, stanford_ner_path, encoding='utf-8')


def load_wiki_groups():
    wiki_words = []
    wiki_freqs = []
    with codecs.open('enwiki-20150602-words-frequency.txt', 'r', 'utf-8') as wiki_file:
        s = wiki_file.readlines()
        s = [line.strip().split() for line in s]
        wiki_words = [line[0] for line in s]
        wiki_freqs = [int(line[1]) for line in s]
        assert len(s) == len(wiki_words) == len(wiki_freqs)
    order_words, _ = list(zip(*sorted(zip(wiki_words, wiki_freqs),
                                                key=lambda x:x[1],
                                                reverse=True)))
    return order_words[:1000], order_words[1000:5000], order_words[5000:]

def load_google_groups():
    with codecs.open('google-10000-english-no-swears.txt', 'r', 'utf-8') as wiki_file:
        s = wiki_file.readlines()
        wiki_words = [line.strip() for line in s]
        assert len(s) == len(wiki_words)
    return wiki_words[:1000], wiki_words[1000:]


def syllables_count(word):
    return syllables_en.count(word)

def find_group(word, g1, g2):
    if word in g1:
        return 2
    elif word in g2:
        return 1
    return 0



def main():
    if not opt.mode and not opt.combine:
        return
    if opt.combine:
        for src in src_files:
            res = []
            with codecs.open(src, "r", "utf-8") as corpus_file:
                words = [line.strip().split() for line in corpus_file.readlines()]
            for tag in opt.combine:
                with codecs.open(tag + '.' + src[src.index('ori'):], "r", "utf-8") as tag_file:
                    ws = [line.strip().split() for line in tag_file.readlines()]
                    assert len(words) == len(ws)
                    for i in range(len(words)):
                        l1, l2 = words[i], ws[i]
                        length = len(l1)
                        assert len(l1) == len(l2)
                        words[i] = [l1[j] + "￨" + l2[j] for j in range(length)]
            with codecs.open('combine.' + '.'.join(opt.combine) + '.' + src[src.index('ori'):],
                             'w', 'utf-8') as out_file:
                out_file.write('\n'.join([' '.join(x) for x in words]))
        return

    if opt.mode == 'wiki':
        g1, g2, _ = load_wiki_groups()
    if opt.mode == 'google':
        g1, g2 = load_google_groups()
    
    for src in src_files:
        print('process: ' + src)
        with codecs.open(src, "r", "utf-8") as corpus_file:
            lines = [line.split() for line in corpus_file.readlines()]
        res = []
        for line in lines:
            if opt.mode == 'pos':
                line = nltk.pos_tag(line)
                line = ' '.join([tags[1] for tags in line])
            elif opt.mode == 'ner':
                line = st.tag(line)
                line = ' '.join([tags[1] for tags in line])
            elif opt.mode == 'syllable':
                line = ' '.join([str(syllables_count(word)) for word in line])
            elif opt.mode == 'wiki':
                line = ' '.join([str(find_group(word, g1, g2)) for word in line])
            elif opt.mode == 'google':
                line = ' '.join([str(find_group(word, g1, g2)) for word in line])
            res.append(line)

        with codecs.open(opt.mode + '.' + src[src.index('ori'):],
                         'w', 'utf-8') as out_file:
            out_file.write('\n'.join(res))

    

if __name__ == '__main__':
    main()
