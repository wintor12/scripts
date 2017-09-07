import argparse
import codecs

parser = argparse.ArgumentParser()
parser.add_argument('-src', required=True, type=str, help='the path of the data')
parser.add_argument('-tgt', required=True, type=str, help='the path of the data')
parser.add_argument('-pred', required=True, type=str, help='the path of the data')

opt = parser.parse_args()
print(opt)

def readSentences(src):
    with codecs.open(src, 'r', 'utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def main():
    src_lines, tgt_lines, pred_lines = readSentences(opt.src), readSentences(opt.tgt), readSentences(opt.pred)
    index = 1
    for s, t, p in zip(src_lines, tgt_lines, pred_lines):
        print(index)
        print('src:  ' + s)
        print('tgt:  ' + t)
        print('pred: ' + p)
        index += 1

if __name__ == '__main__':
    main()
