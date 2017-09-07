import nltk
import codecs


def addPos(src, dst):
    with codecs.open(src, "r", "utf-8") as corpus_file:
        lines = [line.split() for line in corpus_file.readlines()]
    res = []
    for line in lines:
        pos = nltk.pos_tag(line)
        res.append(' '.join([tags[0] + "￨" + tags[1] for tags in pos]))

    with codecs.open(dst, 'w', 'utf-8') as out_file:
        out_file.write('\n'.join(res))



train_src_src = 'PWKP_108016.tag.80.aner.ori.train.src'
train_src_dst = 'PWKP_108016.tag.80.aner.ori.pos.train.src'
valid_src_src = 'PWKP_108016.tag.80.aner.ori.valid.src'
valid_src_dst = 'PWKP_108016.tag.80.aner.ori.pos.valid.src'
test_src_src = 'PWKP_108016.tag.80.aner.ori.test.src'
test_src_dst= 'PWKP_108016.tag.80.aner.ori.pos.test.src'
train_dst_src = 'PWKP_108016.tag.80.aner.ori.train.dst'
train_dst_dst = 'PWKP_108016.tag.80.aner.ori.pos.train.dst'
valid_dst_src = 'PWKP_108016.tag.80.aner.ori.valid.dst'
valid_dst_dst = 'PWKP_108016.tag.80.aner.ori.pos.valid.dst'

addPos(train_src_src, train_src_dst)
addPos(valid_src_src, valid_src_dst)
addPos(test_src_src, test_src_dst)
addPos(train_dst_src, train_dst_dst)
addPos(valid_dst_src, valid_dst_dst)
