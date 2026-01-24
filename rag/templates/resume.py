#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
import base64
import datetime
import json
import os
import re
import time
import pandas as pd
import requests
from api.db.services.knowledgebase_service import KnowledgebaseService
from rag.nlp import rag_tokenizer
from rag.parsers.deepdoc.resume import refactor
from rag.parsers.deepdoc.resume import step_one, step_two
from common.string_utils import remove_redundant_spaces

# Resume parser configuration from environment variables
RESUME_PARSER_ENDPOINT = os.environ.get("RESUME_PARSER_ENDPOINT", "http://127.0.0.1:61670/tog")
RESUME_PARSER_UID = int(os.environ.get("RESUME_PARSER_UID", "1"))
RESUME_PARSER_USER = os.environ.get("RESUME_PARSER_USER", "default_user")

# Minimum number of fields required for a valid parsed resume
MIN_REQUIRED_RESUME_FIELDS = 7

forbidden_select_fields4resume = ["name_pinyin_kwd", "edu_first_fea_kwd", "degree_kwd", "sch_rank_kwd", "edu_fea_kwd"]


def remote_call(filename, binary):
    q = {
        "header": {"uid": RESUME_PARSER_UID, "user": RESUME_PARSER_USER, "log_id": filename},
        "request": {
            "p": {"request_id": "1", "encrypt_type": "base64", "filename": filename, "langtype": "", "fileori": base64.b64encode(binary).decode("utf-8")},
            "c": "resume_parse_module",
            "m": "resume_parse",
        },
    }

    for i in range(3):
        try:
            resume = requests.post(RESUME_PARSER_ENDPOINT, data=json.dumps(q), timeout=5)
            resume = resume.json()["response"]["results"]
            resume = refactor(resume)
            for k in ["education", "work", "project", "training", "skill", "certificate", "language"]:
                if not resume.get(k) and k in resume:
                    del resume[k]

            resume = step_one.refactor(pd.DataFrame([{"resume_content": json.dumps(resume), "tob_resume_id": "x", "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]))
            resume = step_two.parse(resume)
            return resume
        except requests.exceptions.RequestException as e:
            logging.warning(f"Resume parsing HTTP request failed: {e}")
        except Exception:
            logging.exception("Resume parsing failed")
        time.sleep(1 * (i + 1))
    return {}


def chunk(filename, binary=None, callback=None, **kwargs):
    """
    The supported file formats are pdf, doc, docx and txt.
    To maximize the effectiveness, parse the resume correctly, please concat us: https://github.com/infiniflow/ragflow
    """
    if not re.search(r"\.(pdf|doc|docx|txt)$", filename, flags=re.IGNORECASE):
        raise NotImplementedError("file type not supported yet(pdf supported)")

    if not binary:
        with open(filename, "rb") as f:
            binary = f.read()

    if callback:
        callback(0.2, "Resume parsing is going on...")
    resume = remote_call(filename, binary)
    if len(resume.keys()) < MIN_REQUIRED_RESUME_FIELDS:
        if callback:
            callback(-1, "Resume is not successfully parsed.")
        raise Exception("Resume parser remote call fail!")
    if callback:
        callback(0.6, "Done parsing. Chunking...")
    logging.debug("chunking resume: " + json.dumps(resume, ensure_ascii=False, indent=2))

    field_map = {
        "name_kwd": "姓名/名字",
        "name_pinyin_kwd": "姓名拼音/名字拼音",
        "gender_kwd": "性别（男，女）",
        "age_int": "年龄/岁/年纪",
        "phone_kwd": "电话/手机/微信",
        "email_tks": "email/e-mail/邮箱",
        "position_name_tks": "职位/职能/岗位/职责",
        "expect_city_names_tks": "期望城市",
        "work_exp_flt": "工作年限/工作年份/N年经验/毕业了多少年",
        "corporation_name_tks": "最近就职(上班)的公司/上一家公司",
        "first_school_name_tks": "第一学历毕业学校",
        "first_degree_kwd": "第一学历（高中，职高，硕士，本科，博士，初中，中技，中专，专科，专升本，MPA，MBA，EMBA）",
        "highest_degree_kwd": "最高学历（高中，职高，硕士，本科，博士，初中，中技，中专，专科，专升本，MPA，MBA，EMBA）",
        "first_major_tks": "第一学历专业",
        "edu_first_fea_kwd": "第一学历标签（211，留学，双一流，985，海外知名，重点大学，中专，专升本，专科，本科，大专）",
        "degree_kwd": "过往学历（高中，职高，硕士，本科，博士，初中，中技，中专，专科，专升本，MPA，MBA，EMBA）",
        "major_tks": "学过的专业/过往专业",
        "school_name_tks": "学校/毕业院校",
        "sch_rank_kwd": "学校标签（顶尖学校，精英学校，优质学校，一般学校）",
        "edu_fea_kwd": "教育标签（211，留学，双一流，985，海外知名，重点大学，中专，专升本，专科，本科，大专）",
        "corp_nm_tks": "就职过的公司/之前的公司/上过班的公司",
        "edu_end_int": "毕业年份",
        "industry_name_tks": "所在行业",
        "birth_dt": "生日/出生年份",
        "expect_position_name_tks": "期望职位/期望职能/期望岗位",
    }

    titles = []
    for n in ["name_kwd", "gender_kwd", "position_name_tks", "age_int"]:
        v = resume.get(n, "")
        if isinstance(v, list):
            v = v[0]
        if n.find("tks") > 0:
            v = remove_redundant_spaces(v)
        titles.append(str(v))
    doc = {"docnm_kwd": filename, "title_tks": rag_tokenizer.tokenize("-".join(titles) + "-简历")}
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    pairs = []
    for n, m in field_map.items():
        if not resume.get(n):
            continue
        v = resume[n]
        if isinstance(v, list):
            v = " ".join(v)
        if n.find("tks") > 0:
            v = remove_redundant_spaces(v)
        pairs.append((m, str(v)))

    doc["content_with_weight"] = "\n".join(["{}: {}".format(re.sub(r"（[^（）]+）", "", k), v) for k, v in pairs])
    doc["content_ltks"] = rag_tokenizer.tokenize(doc["content_with_weight"])
    doc["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(doc["content_ltks"])
    for n, _ in field_map.items():
        if n not in resume:
            continue
        if isinstance(resume[n], list) and (len(resume[n]) == 1 or n not in forbidden_select_fields4resume):
            resume[n] = resume[n][0]
        if n.find("_tks") > 0:
            if isinstance(resume[n], list):
                resume[n] = [rag_tokenizer.fine_grained_tokenize(t) for t in resume[n]]
            else:
                resume[n] = rag_tokenizer.fine_grained_tokenize(resume[n])
        doc[n] = resume[n]

    logging.debug("chunked resume to " + str(doc))
    kb_id = kwargs.get("kb_id")
    if kb_id:
        KnowledgebaseService.update_parser_config(kb_id, {"field_map": field_map})
    return [doc]


if __name__ == "__main__":
    import sys

    def dummy(a, b):
        pass

    chunk(sys.argv[1], callback=dummy)
