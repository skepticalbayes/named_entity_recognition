import collections
import os
import re

import config

re_normaposnot = re.compile("([a-z]{2,6})n'?t", re.IGNORECASE)


def aposnot(match):
    prefix = match.group(1)
    if prefix.lower() in ['should','could','would','have','does','did', 'do', 'is', 'was','were','has', 'had']:
        return prefix+ ' not'
    elif prefix.lower() == 'ca':
        return prefix+'n not'
    elif prefix.lower() == 'wo':
        return 'will not'
    return match.group()


Tag = collections.namedtuple('Tag',['len','ttype'])


class Negtagger(object):
    def __init__(self):
        self.tup2tag = self.create()
        self.word_pattern = re.compile("(?u)\\b[a-z'A-Z0-9_-]+\\b")
        self.negpat = re.compile('N+(W{1,4})J?|(W{1,3})T')
        self.rulepat = re.compile('(N+)(W{1,4})J?|(W{1,3})(T+)')
        self.ruletup2tag = self.create_rules()

    def create(self):
        with open(os.path.join(config.external_dir, 'rulebase', 'negex_triggers.txt')) as f:
            lines = f.read().split('\n')
        tags = {}
        tup2tag = {}
        for l in lines:
            text,ttype = l.strip().split('\t\t')
            ttype = ttype[1:-1]
            words = text.split(' ')
            if (len(words), ttype) in tags:
                tup2tag[tuple(words)] = tags[(len(words), ttype)]
            else:
                tags[(len(words), ttype)] = Tag(len=len(words), ttype=ttype)
                tup2tag[tuple(words)] = tags[(len(words), ttype)]
        return tup2tag

    def create_rules(self):
        with open(os.path.join(config.external_dir, 'rulebase', 'triggers_classes.txt')) as f:
            lines = f.read().split('\n')
        tags = {}
        tup2tag = {}
        for l in lines:
            text,ttype = l.strip().split('\t')
            pos,ktype = ttype.split('_')
            ttype = ktype+pos[-1]
            words = text.split(' ')
            if (len(words), ttype) in tags:
                tup2tag[tuple(words)] = tags[(len(words), ttype)]
            else:
                tags[(len(words), ttype)] = Tag(len=len(words), ttype=ttype)
                tup2tag[tuple(words)] = tags[(len(words), ttype)]
        return tup2tag

    def ruletype_indexes(self,text,tokens=None,window=5):
        if tokens is None:
            text = re_normaposnot.sub(aposnot, text)
            tokens = self.word_pattern.findall(text.lower())
        results = [None]*len(tokens)
        for i in range(1, len(tokens)+1):
            for j in range(window,0,-1):
                if i-j < 0:
                    continue
                t = tokens[i-j:i]
                if t:
                    tt = tuple(t)
                    tag = self.ruletup2tag.get(tt)
                    if tag:
                        anybigger= False
                        for k in range(i-j,i):
                            if results[k] is not None and results[k].len > tag.len: # if [1,2,3] tag.len< [2,3] tag.len
                                anybigger = True
                        if not anybigger:
                            results[i-j:i] = [tag]*tag.len
                            break
        s = ''.join(['W' if e is None else e.ttype[-1] for e in results])
        span_rules = []
        for m in self.rulepat.finditer(s):
            if m.group(1):
                span_rules.append((m.span(1), m.span(2),results[m.span(1)[0]].ttype[:-1]))
            elif m.group(3):
                span_rules.append((m.span(4), m.span(3),results[m.span(4)[0]].ttype[:-1]))
        return span_rules