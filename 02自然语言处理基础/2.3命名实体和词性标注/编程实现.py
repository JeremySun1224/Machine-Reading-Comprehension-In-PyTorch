# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/5/14 -*-

# 在Python语言中，中文命名实体识别和词性标注可以通过jieba软件包实现
import jieba.posseg as pseg
words = pseg.cut('我爱北京天安门')
for word, pos in words:
    print('word: {word}, pos: {pos}'.format(word=word, pos=pos))


"""
Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\Public\Documents\Wondershare\CreatorTemp\jieba.cache
Loading model cost 0.722 seconds.
Prefix dict has been built successfully.
word: 我, pos: r  # 代词
word: 爱, pos: v  # 动词
word: 北京, pos: ns  # 名词，s表示地名
word: 天安门, pos: ns
"""

# 英文命名实体识别和词性标注恶意通过spaCy软件包实现
import spacy

nlp = spacy.load(name='en_core_web_sm')
doc = nlp(u'Apple may buy a U.K. startup for $1 billion')
print('-----Part of Speech')
for token in doc:
    print(token.text, token.pos_)
print('-----Named Entity Recognition')
for ent in doc.ents:
    print(ent.text, ent.label_)

"""
-----Part of Speech
Apple PROPN  # 专有名词
may VERB  # 动词
buy VERB
a DET  # 冠词
U.K. PROPN
startup NOUN
for ADP  # 介词
$ SYM  # 符号
1 NUM  # 数字
billion NUM
-----Named Entity Recognition
Apple ORG  # 组织机构名
U.K. GPE  # 政府相关地名
$1 billion MONEY  # 钱数
"""