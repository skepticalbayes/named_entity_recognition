import collections
import itertools
import nltk
import re
import traceback
from collections import defaultdict

import enchant
import numpy as np
from nltk import SnowballStemmer

import findnegation


class WsEntityTagger(object):
    def __init__(self, entity_ws, ws_id, ambigous=None, negative=False, ui_tag=False, ui_span_diff=None, stop_words=None, remove_neg=False):
        self.stemmer = SnowballStemmer("english")
        self.ws_id = ws_id
        self.entity_ws = entity_ws
        self.word_pattern = re.compile("(?u)\\b[a-z'A-Z0-9_-]+\\b")
        self.ambig = None
        self.ui_tag = ui_tag
        self.ui_span_diff = ui_span_diff
        if ambigous:
            self.ambig = ambigous
        try:
            self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        except:
            nltk.download('punkt')
            self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.neg_bool = False
        self.remove_neg = remove_neg
        if negative:
            self.neg_bool = True
            self.negtagger = findnegation.Negtagger()
            #set of ids with negation
            neg_words = {'no', 'not', 'without', 'free', 'never', 'absence'}
            self.neg_ws_set = set([ws for ws, mcid in self.ws_id.iteritems() if any(w in neg_words for w in ws)])
        self.sent_split = re.compile("[,\n]+")
        self.phrase_pat = re.compile('[^,]+')
        self.stop_words = stop_words

    def stem(self,w):
        w = w.lower()
        if w in ['over-eat', 'overeat', 'overate', 'over-ate', 'overeating', 'over-eating', 'overeated', 'over-eated']:
            return 'overeat'
        sw = self.stemmer.stem(w)
        if self.ambig:
            if sw in self.ambig:
                return self.ambig[sw].get(w, w)
        return sw

    # from tokens to stemmed_kwds to wordsets, indices, negtagger(negation type)

    def tag(self, text):
        text = text.lower()
        # load negation tagger
        if self.neg_bool:
            text = findnegation.re_normaposnot.sub(findnegation.aposnot, text)
        # Sentence splitting
        sents = self.sent_detector.tokenize(text)
        sents = [e for sent in sents for e in sent.split('.') if e]
        sents = [e for sent in sents for e in sent.split(',') if e]
        final_sents = []
        if self.ui_tag:
            ui_tags = defaultdict()
            coincidence = defaultdict(list)
            occurences = defaultdict(list)
        alltags = defaultdict()
        for senti, sent in enumerate(sents):
            td = collections.defaultdict(list)
            tokens = self.word_pattern.findall(sent)
            for i, w in enumerate(tokens):
                if w.isupper() and w in self.entity_ws:
                    for ws in self.entity_ws[w]:
                        td[ws].append((w,i))
                else:
                    stemw = self.stem(w)
                    if stemw in self.entity_ws:
                        for ws in self.entity_ws[stemw]:
                            td[ws].append((stemw, i))
            tags = []
            for ws in td:
                wl, il = zip(*td[ws])
                if tuple(sorted(set(wl))) == ws:
                    for ili, wli in zip(il,wl):
                        tokens[ili] = wli
                    if not self.ui_tag:
                        if len(ws) < len(il):
                            imedian = np.median(il)
                            stdws = sorted([(w, i - imedian, i) for w, i in td[ws]])
                            il = sorted([e[2] for i, e in enumerate(stdws) if not(i >= 1 and  e[0] == stdws[i-1][0])])
                        tags.append((tuple([tokens[j] for j in il]), self.ws_id[ws], il, ''))
                    else:
                        tags.append((wl,  self.ws_id[ws], il, ''))
            if self.stop_words and not self.ui_tag:
                tokens = [i for i in tokens if i not in self.stop_words and i]
            final_sents.append(tokens)
            # remove subset tags
            seen = defaultdict(list)
            tags = sorted(tags, key=lambda e: len(e[0]), reverse=True)
            for tag in tags[:]:
                if len(set(tag[0])) == len(set(tag[0]) & set(seen.keys())):
                    tags.remove(tag)
                else:
                    for ix, w in enumerate(tag[0]):
                        if tag[2][ix] not in seen[w]:
                            seen[w].append(tag[2][ix])
            # Get non overlapping tags for UI interlinking
            if self.ui_tag:
                tags = sorted(tags, key=lambda x: x[2], reverse=False)
                res_tags = defaultdict()
                for ix, tag in enumerate(tags):
                    if len(tags) > 1:
                        for j in tags[ix+1:]:
                            if set(tag[1]) & set(j[1]):
                                first = [seen[w] for w in set(tag[0]).difference(set(j[0]))]
                                second = [seen[w] for w in set(j[0]).difference(set(tag[0]))]
                                first = sorted([i for i in itertools.product(*first)],
                                               key=lambda x: max(x)-min(x), reverse=False)[0]
                                second = sorted([i for i in itertools.product(*second)],
                                                key=lambda x: max(x) - min(x), reverse=False)[0]
                                if max(first) - min(first) - len(first) < max(second) - min(second) - len(second):
                                    first_bool = True
                                elif max(first) - min(first) - len(first) == max(second) - min(second) - len(second):
                                    if min(first) <= min(second):
                                        first_bool = True
                                    else:
                                        first_bool = False
                                else:
                                    first_bool = False
                                common = set(tag[0]) & set(j[0])
                                for w in common:
                                    if len(seen[w]) > 1:
                                        first_w = sorted([i for i in itertools.product(seen[w], first)],
                                                         key=lambda x:max(x)-min(x), reverse=False)
                                        second_w = sorted([i for i in itertools.product(seen[w], second)],
                                                          key=lambda x: max(x) - min(x), reverse=False)
                                        if first_bool:
                                            for index in first_w:
                                                for index1 in second_w:
                                                    if index[0] != index1[0]:
                                                        first = list(first)
                                                        first.append(index[0])
                                                        first = tuple(sorted(first))
                                                        second = list(second)
                                                        second.append(index1[0])
                                                        second = tuple(sorted(second))
                                                        break
                                                break
                                        else:
                                            for index1 in second_w:
                                                for index in first_w:
                                                    if index[0] != index1[0]:
                                                        first = list(first)
                                                        first.append(index[0])
                                                        first = tuple(sorted(first))
                                                        second = list(second)
                                                        second.append(index1[0])
                                                        second = tuple(sorted(second))
                                                        break
                                                break
                                    else:
                                        first = list(first)
                                        first.append(seen[w][0])
                                        first = tuple(sorted(first))
                                        second = list(second)
                                        second.append(seen[w][0])
                                        second = tuple(sorted(second))
                                res_tags[tuple(sorted(set(tag[0])))] = (tag[1], first, '')
                                res_tags[tuple(sorted(set(j[0])))] = (j[1], second, '')
                        if not res_tags.get(tuple(sorted(set(tag[0]))), None):
                            first = [seen[w] for w in set(tag[0])]
                            first = sorted([i for i in itertools.product(*first)],
                                           key=lambda x: max(x) - min(x), reverse=False)[0]
                            res_tags[tuple(sorted(set(tag[0])))] = (tag[1], tuple(sorted(first)), '')
                    else:
                        if not res_tags.get(tuple(sorted(set(tag[0]))), None):
                            first = [seen[w] for w in set(tag[0])]
                            first = sorted([i for i in itertools.product(*first)],
                                           key=lambda x: max(x) - min(x), reverse=False)[0]
                            res_tags[tuple(sorted(set(tag[0])))] = (tag[1], tuple(sorted(first)), '')

            if self.neg_bool:
                nspans = self.negtagger.ruletype_indexes(sent, tokens=[e.lower() for e in tokens])
                if nspans:
                    temp = tags
                    for i,e in enumerate(temp):
                        ktype = inrulespans(e[2], nspans)
                        if ktype:
                            e = list(e)
                            e[0] = tuple(sorted(set(e[0])))
                            e[-1] = ktype
                            if 'NEG' in ktype:
                                if self.remove_neg:
                                    tags.remove(temp[i])
                                    if self.ui_tag:
                                        res_tags.pop(e[0])
                                else:
                                    tags[i] = e
                                    if self.ui_tag:
                                        tag = list(res_tags[e[0]])
                                        tag[-1] = ktype
                                        res_tags[e[0]] = tuple(tag)

            # if self.neg_bool:
            #     for e in tags:
            #         if 'NEG' in e[-1]:
            #             tags.remove(e)
            #             if self.ui_tag:
# order and Iterate over res_tags and
            if self.ui_tag:
                assert len(res_tags) == len(tags)
                ui_tag = defaultdict()
                if tags:
                    res_tags = sorted(res_tags.iteritems(),
                                      key=lambda (k, v): (max(v[1])-min(v[1])-len(v[1]), min(v[1])), reverse=False)
                    for ix,(ws, tag) in enumerate(res_tags):
                        tag = list(tag)
                        tag.append(max(tag[1])-min(tag[1])-len(ws))
                        if tag[-1] + 1 <= self.ui_span_diff:
                            ui_tag[ws] = tuple(tag)
                            occurences[ws].append(senti)
                            if len(tags) > 1:
                                for ws1, tag1 in res_tags[ix+1:]:
                                    if max(tag1[1]) - min(tag1[1]) - len(ws1) + 1 <= self.ui_span_diff:
                                        if min(tag1[1]) <= max(tag[1]) <= max(tag1[1]):
                                            coincidence[senti].append((ws, ws1))
                ui_tags[senti] = ui_tag
            alltags[senti] = tags

        if self.ui_tag:
            try:
                ui_tags = sorted(ui_tags.iteritems(),
                                  key=lambda (k, v): k, reverse=False)
                final_tags = defaultdict(list)
                insertions = []
                for senti, Tags in ui_tags:
                    if Tags:
                        if coincidence.get(senti, None):
                            for ws_pair in coincidence.get(senti, None):
                                first = sorted(list(set(occurences[ws_pair[0]]).difference(set(occurences[ws_pair[1]]))))
                                second = sorted(list(set(occurences[ws_pair[1]]).difference(set(occurences[ws_pair[0]]))))
                                if first or second:
                                    if second:
                                        if Tags[ws_pair[0]][0] not in insertions:
                                            final_tags[senti].append({ws_pair[0]: Tags[ws_pair[0]]})
                                            insertions.append(Tags[ws_pair[0]][0])
                                        else:
                                            if Tags[ws_pair[1]][0] not in insertions:
                                                final_tags[senti].append({ws_pair[1]: Tags[ws_pair[1]]})
                                                insertions.append(Tags[ws_pair[1]][0])
                                    elif not second:
                                        if Tags[ws_pair[1]][0] not in insertions:
                                            final_tags[senti].append({ws_pair[1]: Tags[ws_pair[1]]})
                                            insertions.append(Tags[ws_pair[1]][0])
                                        else:
                                            if Tags[ws_pair[0]][0] not in insertions:
                                                final_tags[senti].append({ws_pair[0]: Tags[ws_pair[0]]})
                                                insertions.append(Tags[ws_pair[0]][0])

                                else:
                                    if Tags[ws_pair[0]][-1] == Tags[ws_pair[1]][-1]:
                                        if min(Tags[ws_pair[0]][1]) > min(Tags[ws_pair[1]][1]):
                                            if Tags[ws_pair[0]][0] not in insertions:
                                                final_tags[senti].append({ws_pair[0]: Tags[ws_pair[0]]})
                                                insertions.append(Tags[ws_pair[0]][0])
                                                if not set([e for e in occurences[ws_pair[1]] if e!= senti]):
                                                    if ws_pair[1] not in insertions:
                                                        common_tag = list(Tags[ws_pair[1]])
                                                        residual_ix = tuple(sorted(set((common_tag[1])).difference(set(Tags[ws_pair[0]][1]))))
                                                        common_tag[1] = residual_ix
                                                        final_tags[senti].append({ws_pair[1]: common_tag})
                                                        insertions.append(Tags[ws_pair[1]][0])
                                            else:
                                                if Tags[ws_pair[1]][0] not in insertions:
                                                    final_tags[senti].append({ws_pair[1]: Tags[ws_pair[1]]})
                                                    insertions.append(Tags[ws_pair[1]][0])
                                        elif min(Tags[ws_pair[0]][1]) < min(Tags[ws_pair[1]][1]):
                                            if Tags[ws_pair[1]][0] not in insertions:
                                                final_tags[senti].append({ws_pair[1]: Tags[ws_pair[1]]})
                                                insertions.append(Tags[ws_pair[1]][0])
                                                if not set([e for e in occurences[ws_pair[0]] if e!=senti]):
                                                    if Tags[ws_pair[0]][0] not in insertions:
                                                        common_tag = list(Tags[ws_pair[0]])
                                                        residual_ix = tuple(sorted(set((common_tag[1])).difference(set(Tags[ws_pair[1]][1]))))
                                                        common_tag[1] = residual_ix
                                                        final_tags[senti].append({ws_pair[0]: common_tag})
                                                        insertions.append(Tags[ws_pair[0]][0])
                                            else:
                                                if Tags[ws_pair[0]][0] not in insertions:
                                                    final_tags[senti].append({ws_pair[0]: Tags[ws_pair[0]]})
                                                    insertions.append(Tags[ws_pair[0]][0])
                                        else:
                                            if Tags[ws_pair[0]][0] not in insertions:
                                                final_tags[senti].append({ws_pair[0]: Tags[ws_pair[0]]})
                                                insertions.append(Tags[ws_pair[0]][0])
                                                if not set([e for e in occurences[ws_pair[1]] if e!= senti]):
                                                    if Tags[ws_pair[1]][0] not in insertions:
                                                        common_tag = list(Tags[ws_pair[1]])
                                                        residual_ix = tuple(sorted(set((common_tag[1])).difference(set(Tags[ws_pair[0]][1]))))
                                                        common_tag[1] = residual_ix
                                                        final_tags[senti].append({ws_pair[1]: common_tag})
                                                        insertions.append(Tags[ws_pair[1]][0])
                                            else:
                                                if Tags[ws_pair[1]][0] not in insertions:
                                                    final_tags[senti].append({ws_pair[1]: Tags[ws_pair[1]]})
                                                    insertions.append(Tags[ws_pair[1]][0])
                                                    if not set([e for e in occurences[ws_pair[0]] if e!=senti]):
                                                        if Tags[ws_pair[0]][0] not in insertions:
                                                            common_tag = list(Tags[ws_pair[0]])
                                                            residual_ix = tuple(sorted(
                                                                set((common_tag[1])).difference(set(Tags[ws_pair[1]][1]))))
                                                            common_tag[1] = residual_ix
                                                            final_tags[senti].append({ws_pair[0]: common_tag})
                                                            insertions.append(Tags[ws_pair[0]][0])
                                    elif Tags[ws_pair[0]][-1] < Tags[ws_pair[1]][-1]:
                                        if Tags[ws_pair[0]][0] not in insertions:
                                            final_tags[senti].append({ws_pair[0]: Tags[ws_pair[0]]})
                                            insertions.append(Tags[ws_pair[0]][0])
                                            if not set([e for e in occurences[ws_pair[1]] if e!= senti]):
                                                if Tags[ws_pair[1]][0] not in insertions:
                                                    common_tag = list(Tags[ws_pair[1]])
                                                    residual_ix = tuple(
                                                        sorted(set((common_tag[1])).difference(set(Tags[ws_pair[0]][1]))))
                                                    common_tag[1] = residual_ix
                                                    final_tags[senti].append({ws_pair[1]: common_tag})
                                                    insertions.append(Tags[ws_pair[1]][0])
                                        else:
                                            if Tags[ws_pair[1]][0] not in insertions:
                                                final_tags[senti].append({ws_pair[1]: Tags[ws_pair[1]]})
                                                insertions.append(Tags[ws_pair[1]][0])
                                                if not set([e for e in occurences[ws_pair[0]] if e!=senti]):
                                                    if Tags[ws_pair[1]][0] not in insertions:
                                                        common_tag = list(Tags[ws_pair[0]])
                                                        residual_ix = tuple(sorted(
                                                            set((common_tag[1])).difference(set(Tags[ws_pair[1]][1]))))
                                                        common_tag[1] = residual_ix
                                                        final_tags[senti].append({ws_pair[0]: common_tag})
                                                        insertions.append(Tags[ws_pair[1]][0])
                                    else:
                                        if Tags[ws_pair[1]][0] not in insertions:
                                            final_tags[senti].append({ws_pair[1]: Tags[ws_pair[1]]})
                                            insertions.append(Tags[ws_pair[1]][0])
                                            if not set([e for e in occurences[ws_pair[0]] if e!=senti]):
                                                if Tags[ws_pair[0]][0] not in insertions:
                                                    common_tag = list(Tags[ws_pair[0]])
                                                    residual_ix = tuple(sorted(
                                                        set((common_tag[1])).difference(set(Tags[ws_pair[1]][1]))))
                                                    common_tag[1] = residual_ix
                                                    final_tags[senti].append({ws_pair[0]: common_tag})
                                                    insertions.append(Tags[ws_pair[0]][0])
                                        else:
                                            if Tags[ws_pair[0]][0] not in insertions:
                                                final_tags[senti].append({ws_pair[0]: Tags[ws_pair[0]]})
                                                insertions.append(Tags[ws_pair[0]][0])
                                                if not set([e for e in occurences[ws_pair[1]] if e!= senti]):
                                                    if Tags[ws_pair[1]][0] not in insertions:
                                                        common_tag = list(Tags[ws_pair[1]])
                                                        residual_ix = tuple(
                                                            sorted(set((common_tag[1])).difference(set(Tags[ws_pair[0]][1]))))
                                                        common_tag[1] = residual_ix
                                                        final_tags[senti].append({ws_pair[1]: common_tag})
                                                        insertions.append(Tags[ws_pair[1]][0])
                        else:
                            for ws, ws_info in Tags.iteritems():
                                if ws_info[0] not in insertions:
                                    final_tags[senti].append({ws: ws_info})
                                    insertions.append(ws_info[0])
            except:
                print Tags
                print ws_pair
                print ui_tags
                print occurences
                print coincidence
                print sents
                raise
            all_tags = defaultdict()
            try:
                for senti,tags_ in alltags.iteritems():
                    ws_tags = []
                    for ws_info in tags_:
                        il = ws_info[2]
                        if len(set(ws_info[0])) < len(set(il)):
                            imedian = np.median(il)
                            stdws = sorted([(w, i - imedian, i) for w, i in zip(ws_info[0], il)])
                            il = sorted([e[2] for i, e in enumerate(stdws) if not (i >= 1 and e[0] == stdws[i - 1][0])])
                        ws_tags.append(([final_sents[senti][j] for j in il], ws_info[1], tuple(il), ''))
                    all_tags[senti] = ws_tags
                return final_tags, all_tags, [' '.join(token_list) for token_list in final_sents], sents
            except:
                print "alltags", alltags
                print "final_sents", final_sents
                print 'senti', senti
                print 'ws_info', ws_info
                raise
        else:
            return alltags, [' '.join(token_list) for token_list in final_sents], sents


def inrulespans(windexes,spans):
    for rspan,wspan,rtype in spans:
        wbool = any((i >= wspan[0] and i < wspan[1]) for i in windexes)
        rbool = any((i >= rspan[0] and i < rspan[1]) for i in windexes)
        if wbool and not rbool:
            return rtype
    return ''


class ExtractEntity(object):
    def __init__(self, entity_id, entity, tup_id):
        self.word_pattern = re.compile('(?u)\\b[a-zA-Z0-9-]+\\b')
        self.entity_id = entity_id
        self.entity = entity
        self.entity_tup_id = tup_id
        self.US = enchant.DictWithPWL("en_US")
        self.UK = enchant.Dict("en_UK")

    def wordcheckfunc(self, w):
        try:
            return (self.US.check(w) or self.UK.check(w))
        except:
            print traceback.format_exc()
        return False

    def find_entity(self, text, tokens=None, window=3):
        if tokens is None:
            tokens = self.word_pattern.findall(text.lower())
        results = []
        for i in range(1, len(tokens)+1):
            for j in range(2, window+1):
                if i-j <= 0:
                    continue
                t = tokens[i-j:i]
                if t:
                    tt = tuple(t)
                    id = self.entity_tup_id.get(tt)
                    if id:
                        w = ' '.join(tt)
                        if (w, id[-1]) not in results:
                            results.append((w, id[-1]))
        return results

    def extract_entity(self, text):
        entity_name = []
        for i, m in enumerate(self.word_pattern.finditer(text)):
            w = m.group()
            w = w.lower() if len(w) > 3 else w
            if self.wordcheckfunc(w):
                continue
            if self.entity:
                if w in self.entity:
                    entity_name.append((m.span(), w))
                    continue
            else:
                id = self.entity_id.get(w)
                if id and (w, id[-1]) not in entity_name:
                    entity_name.append((w, id[-1]))
        if entity_name:
            return entity_name
        return []

    def find(self, text, tokens=None):
        d = self.find_entity(text, tokens=tokens)
        s = self.extract_entity(text)
        dws = [e for e, _ in d]
        s = [(e, medid) for e, medid in s if not any([e in dw for dw in dws])]
        return s + d