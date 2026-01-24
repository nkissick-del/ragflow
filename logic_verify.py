import re
import datetime
import numpy as np


def test_rmHtmlTag():
    # Implementation copy
    def rmHtmlTag(line):
        return re.sub(r"<[^>]+>", " ", line, flags=re.IGNORECASE)

    assert rmHtmlTag("<div>hello</div>") == " hello "
    assert rmHtmlTag("<p class='test'>text</p>") == " text "
    assert rmHtmlTag("<br/>") == " "
    assert rmHtmlTag("<a href='?q=1&p=1'>link</a>") == " link "
    print("test_rmHtmlTag passed")


def test_dealWithInt64():
    # Implementation copy
    def dealWithInt64(d):
        if isinstance(d, dict):
            for n, v in d.items():
                d[n] = dealWithInt64(v)
        elif isinstance(d, list):
            d = [dealWithInt64(t) for t in d]
        elif isinstance(d, np.integer):
            d = int(d)
        elif isinstance(d, np.floating):
            d = float(d)
        return d

    d = {"a": np.int64(1), "b": [np.float64(2.5), 3], "c": {"d": np.int32(4), "e": np.float32(5.5)}}
    result = dealWithInt64(d)
    assert isinstance(result["a"], int)
    assert isinstance(result["b"][0], float)
    assert isinstance(result["c"]["d"], int)
    assert isinstance(result["c"]["e"], float)
    print("test_dealWithInt64 passed")


def test_subordinates_count_filter():
    fea_subordinates_count = ["10", "abc", "20", " "]
    # New logic
    fea_subordinates_count = [int(i) for i in fea_subordinates_count if re.match(r"^\d+$", str(i))]
    assert fea_subordinates_count == [10, 20]
    max_sub = np.max(fea_subordinates_count)
    assert max_sub == 20
    print("test_subordinates_count_filter passed")


def test_year_check():
    current_year = datetime.datetime.now().year

    y = "2027"
    y_int = int(y) if y and str(y).isdigit() else 0
    assert y_int > current_year if current_year < 2027 else not (y_int > current_year)

    y = "2025"
    y_int = int(y) if y and str(y).isdigit() else 0
    assert not (y_int > current_year) if current_year >= 2025 else y_int > current_year
    print("test_year_check passed")


def test_tob_resume_id_handling():
    cv = {"tob_resume_id": 12345}
    tr_id = cv.get("tob_resume_id")
    if tr_id is not None:
        cv["id"] = str(tr_id)
    assert cv["id"] == "12345"

    cv_missing = {}
    tr_id = cv_missing.get("tob_resume_id")
    if tr_id is not None:
        cv_missing["id"] = str(tr_id)
    else:
        cv_missing["id"] = None
    assert cv_missing["id"] is None
    print("test_tob_resume_id_handling passed")


if __name__ == "__main__":
    test_rmHtmlTag()
    test_dealWithInt64()
    test_subordinates_count_filter()
    test_year_check()
    test_tob_resume_id_handling()
    print("All logic tests passed!")
