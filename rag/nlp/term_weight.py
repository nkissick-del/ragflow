#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import math
import json
import re
import os
import numpy as np
from rag.nlp import rag_tokenizer
from common.file_utils import get_project_base_directory

PATT_SPECIAL_CHARS = re.compile(r"[~—\t @#%!<>,\.\?\":;'\{\}\[\]_=\(\)\|，。？》•●○↓《；‘’：“”【¥ 】…￥！、·（）×`&\\/「」\\]")
PATT_NUM_END = re.compile(r"[0-9]$")
PATT_ONE_TERM = re.compile(r"[0-9a-z]{1,2}$")
PATT_ALPHANUM = re.compile(r"[0-9a-zA-Z]")
PATT_WHITESPACE = re.compile(r"[ \t]+")
PATT_ALPHA_END = re.compile(r".*[a-zA-Z]$")
PATT_NUM_MULTI = re.compile(r"[0-9,.]{2,}$")
PATT_SHORT_LETTER = re.compile(r"[a-z]{1,2}$")
PATT_NUM_SPACE = re.compile(r"[0-9. -]{2,}$")
PATT_LETTER = re.compile(r"[a-z. -]+$")
PATT_NUM_HYPHEN = re.compile(r"[0-9-]+")

class Dealer:
    def __init__(self):
        self.stop_words = set(["请问",
                               "您",
                               "你",
                               "我",
                               "他",
                               "是",
                               "的",
                               "就",
                               "有",
                               "于",
                               "及",
                               "即",
                               "在",
                               "为",
                               "最",
                               "有",
                               "从",
                               "以",
                               "了",
                               "将",
                               "与",
                               "吗",
                               "吧",
                               "中",
                               "#",
                               "什么",
                               "怎么",
                               "哪个",
                               "哪些",
                               "啥",
                               "相关"])

        def load_dict(fnm):
            res = {}
            f = open(fnm, "r")
            while True:
                line = f.readline()
                if not line:
                    break
                arr = line.replace("\n", "").split("\t")
                if len(arr) < 2:
                    res[arr[0]] = 0
                else:
                    res[arr[0]] = int(arr[1])

            c = 0
            for _, v in res.items():
                c += v
            if c == 0:
                return set(res.keys())
            return res

        fnm = os.path.join(get_project_base_directory(), "rag/res")
        self.ne, self.df = {}, {}
        try:
            self.ne = json.load(open(os.path.join(fnm, "ner.json"), "r"))
        except Exception:
            logging.warning("Load ner.json FAIL!")
        try:
            self.df = load_dict(os.path.join(fnm, "term.freq"))
        except Exception:
            logging.warning("Load term.freq FAIL!")

    def pretoken(self, txt, num=False, stpwd=True):
        res = []
        for t in rag_tokenizer.tokenize(txt).split():
            tk = t
            if (stpwd and tk in self.stop_words) or (
                    PATT_NUM_END.match(tk) and not num):
                continue

            if PATT_SPECIAL_CHARS.match(t):
                tk = "#"

            # tk = re.sub(r"([\+\\-])", r"\\\1", tk)
            if tk != "#" and tk:
                res.append(tk)
        return res

    def token_merge(self, tks):
        def one_term(t):
            return len(t) == 1 or PATT_ONE_TERM.match(t)

        res, i = [], 0
        while i < len(tks):
            j = i
            if i == 0 and one_term(tks[i]) and len(
                    tks) > 1 and (len(tks[i + 1]) > 1 and not PATT_ALPHANUM.match(tks[i + 1])):  # 多 工位
                res.append(" ".join(tks[0:2]))
                i = 2
                continue

            while j < len(
                    tks) and tks[j] and tks[j] not in self.stop_words and one_term(tks[j]):
                j += 1
            if j - i > 1:
                if j - i < 5:
                    res.append(" ".join(tks[i:j]))
                    i = j
                else:
                    res.append(" ".join(tks[i:i + 2]))
                    i = i + 2
            else:
                if len(tks[i]) > 0:
                    res.append(tks[i])
                i += 1
        return [t for t in res if t]

    def ner(self, t):
        if not self.ne:
            return ""
        res = self.ne.get(t, "")
        if res:
            return res

    def split(self, txt):
        tks = []
        for t in PATT_WHITESPACE.sub(" ", txt).split():
            if tks and PATT_ALPHA_END.match(tks[-1]) and \
                    PATT_ALPHA_END.match(t) and tks and \
                    self.ne.get(t, "") != "func" and self.ne.get(tks[-1], "") != "func":
                tks[-1] = tks[-1] + " " + t
            else:
                tks.append(t)
        return tks

    def weights(self, tks, preprocess=True):
        def ner(t):
            if PATT_NUM_MULTI.match(t):
                return 2
            if PATT_SHORT_LETTER.match(t):
                return 0.01
            if not self.ne or t not in self.ne:
                return 1
            m = {"toxic": 2, "func": 1, "corp": 3, "loca": 3, "sch": 3, "stock": 3,
                 "firstnm": 1}
            return m[self.ne[t]]

        def postag(t):
            t = rag_tokenizer.tag(t)
            if t in set(["r", "c", "d"]):
                return 0.3
            if t in set(["ns", "nt"]):
                return 3
            if t in set(["n"]):
                return 2
            if PATT_NUM_HYPHEN.match(t):
                return 2
            return 1

        def freq(t):
            if PATT_NUM_SPACE.match(t):
                return 3
            s = rag_tokenizer.freq(t)
            if not s and PATT_LETTER.match(t):
                return 300
            if not s:
                s = 0

            if not s and len(t) >= 4:
                s = [tt for tt in rag_tokenizer.fine_grained_tokenize(t).split() if len(tt) > 1]
                if len(s) > 1:
                    s = np.min([freq(tt) for tt in s]) / 6.
                else:
                    s = 0

            return max(s, 10)

        def df(t):
            if PATT_NUM_SPACE.match(t):
                return 5
            if t in self.df:
                return self.df[t] + 3
            elif PATT_LETTER.match(t):
                return 300
            elif len(t) >= 4:
                s = [tt for tt in rag_tokenizer.fine_grained_tokenize(t).split() if len(tt) > 1]
                if len(s) > 1:
                    return max(3, np.min([df(tt) for tt in s]) / 6.)

            return 3

        def idf(s, N):
            return math.log10(10 + ((N - s + 0.5) / (s + 0.5)))

        tw = []
        if not preprocess:
            idf1 = np.array([idf(freq(t), 10000000) for t in tks])
            idf2 = np.array([idf(df(t), 1000000000) for t in tks])
            wts = (0.3 * idf1 + 0.7 * idf2) * \
                  np.array([ner(t) * postag(t) for t in tks])
            wts = [s for s in wts]
            tw = list(zip(tks, wts))
        else:
            for tk in tks:
                tt = self.token_merge(self.pretoken(tk, True))
                idf1 = np.array([idf(freq(t), 10000000) for t in tt])
                idf2 = np.array([idf(df(t), 1000000000) for t in tt])
                wts = (0.3 * idf1 + 0.7 * idf2) * \
                      np.array([ner(t) * postag(t) for t in tt])
                wts = [s for s in wts]
                tw.extend(zip(tt, wts))

        S = np.sum([s for _, s in tw])
        return [(t, s / S) for t, s in tw]
