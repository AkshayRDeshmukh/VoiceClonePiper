"""Text cleaning, number spelling, balanced script selection and WER alignment."""
import heapq
import math
import unicodedata
from difflib import SequenceMatcher

import regex as re

CHARS_PER_SEC = 14.0

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth",
       "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def n2w(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + " hundred" + (" " + n2w(n % 100) if n % 100 else "")
    for v, w in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= v:
            return n2w(n // v) + " " + w + (" " + n2w(n % v) if n % v else "")
    return str(n)


def year_words(n: int) -> str:
    if 2000 <= n < 2010:
        return n2w(n)
    if n % 100 == 0:
        return n2w(n // 100) + " hundred"
    lo = n % 100
    return n2w(n // 100) + " " + ("oh " + ONES[lo] if lo < 10 else n2w(lo))


def ordinal(n: int) -> str:
    w = n2w(n)
    last = re.search(r"([a-z]+)$", w).group(1)
    head = w[: -len(last)]
    if last in ORD:
        return head + ORD[last]
    return head + (last[:-1] + "ieth" if last.endswith("y") else last + "th")


def dec_words(s: str) -> str:
    a, _, b = s.partition(".")
    return n2w(int(a)) + (" point " + " ".join(ONES[int(d)] for d in b) if b else "")


def expand_numbers(s: str) -> str:
    s = re.sub(r"(\d),(?=\d{3}\b)", r"\1", s)

    def money(m):
        a, b = int(m.group(1)), m.group(2)
        out = n2w(a) + (" dollar" if a == 1 else " dollars")
        if b and int(b):
            out += " and " + n2w(int(b)) + (" cent" if int(b) == 1 else " cents")
        return out

    def decade(m):
        w = year_words(int(m.group(1)))
        return w[:-1] + "ies" if w.endswith("y") else w + "s"

    def num(m):
        t = m.group(0)
        if len(t) > 9:
            return t
        n = int(t)
        if len(t) == 4 and 1100 <= n < 2100:
            return year_words(n)
        if len(t) > 1 and t[0] == "0":
            return " ".join(ONES[int(d)] for d in t)
        return n2w(n)

    s = re.sub(r"\$(\d+)(?:\.(\d{2}))?\b", money, s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s?%", lambda m: dec_words(m.group(1)) + " percent", s)
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), s, flags=re.I)
    s = re.sub(r"\b(\d{4})s\b", decade, s)
    s = re.sub(r"\b\d+\.\d+\b", lambda m: dec_words(m.group(0)), s)
    s = re.sub(r"\b\d+\b", num, s)
    return s


ABBR = [(re.compile(p, f), w) for p, w, f in [
    (r"\bMr\.", "Mister", 0), (r"\bMrs\.", "Missus", 0), (r"\bDr\.", "Doctor", 0),
    (r"\bvs\.", "versus", re.I), (r"\betc\.", "et cetera", re.I), (r"\be\.g\.", "for example", re.I),
    (r"\bi\.e\.", "that is", re.I), (r"\bSt\. (?=[A-Z])", "Saint ", 0), (r"\s&\s", " and ", 0)]]


def tidy(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r'[\u201C\u201D\u201E\u00AB\u00BB"]', "", s)
    s = re.sub(r"[\u2018\u2019\u02BC`\u00B4]", "'", s)
    s = re.sub(r"\s*[\u2014\u2013]\s*|\s+-\s+|--+", ", ", s)
    s = s.replace("\u2026", "...")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?।])", r"\1", s)
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"^[\s,;:'.\-]+", "", s)
    s = re.sub(r"[,;:]\s*$", ".", s)
    return s.strip()


def clean_sentence(s: str, english: bool, min_chars=20, max_chars=180, skip_caps=True):
    """Returns (clean_text, reason_rejected_or_None)."""
    if english:
        for rx, w in ABBR:
            s = rx.sub(w, s)
    s = tidy(s)
    if english:
        s = tidy(expand_numbers(s))
    if re.search(r"[\p{L}\p{M}]$", s):
        s += "."
    if re.search(r"https?:|www\.|\S@\S", s):
        return s, "link or email"
    if re.search(r"[|#*_=<>{}\[\]\\/~^()$%+@&]", s):
        return s, "symbols"
    if re.search(r"\d", s):
        return s, "digits"
    if len(re.findall(r"[\p{L}\p{M}]", s)) < len(re.sub(r"\s", "", s)) * 0.7:
        return s, "too few letters"
    if len(s) < min_chars:
        return s, "too short"
    if len(s) > max_chars:
        return s, "too long"
    if skip_caps and re.search(r"(?<![\p{L}])\p{Lu}{2,}(?![\p{L}])", s):
        return s, "ALL-CAPS word"
    if re.search(r"(.)\1{3,}", s):
        return s, "repeated characters"
    return s, None


def sentence_type(s: str) -> str:
    return {"?": "question", "!": "exclaim"}.get(s[-1:], "statement")


def norm_key(s: str) -> str:
    return re.sub(r"[^\p{L}\p{M}\p{N}]+", " ", s.lower()).strip()


def est_sec(t: str) -> float:
    return len(t) / CHARS_PER_SEC + 0.4


def _feats(t):
    ws = re.findall(r"[\p{L}\p{M}']+", t.lower())
    tri = set()
    for w in ws:
        p = f" {w} "
        for i in range(len(p) - 2):
            tri.add(p[i:i + 3])
    return set(ws), tri


def pick_balanced(cands, target_sec):
    """Lazy-greedy selection maximising new words + letter trigrams, boosting questions/exclamations."""
    seen_w, seen_t, data, heap = set(), set(), [], []

    def gain(k):
        c = data[k]
        g = sum(1 for t in c["_t"] if t not in seen_t) + 2 * sum(1 for w in c["_w"] if w not in seen_w)
        g += 5 if c["type"] == "question" else 4 if c["type"] == "exclaim" else 0
        return g / math.sqrt(max(1, len(c["text"])))

    for k, c in enumerate(cands):
        w, t = _feats(c["text"])
        data.append({**c, "_w": w, "_t": t})
        heapq.heappush(heap, (-gain(k), k))
    out, total = [], 0.0
    while heap and total < target_sec:
        _, k = heapq.heappop(heap)
        g = gain(k)
        if heap and g < -heap[0][0] - 1e-9:
            heapq.heappush(heap, (-g, k))
            continue
        c = data[k]
        out.append(c)
        total += est_sec(c["text"])
        seen_w |= c["_w"]
        seen_t |= c["_t"]
    return out


def words_of(s: str, english=True):
    if english:
        s = expand_numbers(s)
    s = unicodedata.normalize("NFKC", s.lower()).replace("\u2019", "'")
    if english:  # café -> cafe (only for Latin script; Indic vowel signs must stay)
        s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^\p{L}\p{M}\p{N}' ]+", " ", s)
    ws = [w.strip("'") for w in s.split() if w.strip("'")]
    return [SAME_AS.get(w, w) for w in ws]


SAME_AS = {"ok": "okay", "mr": "mister", "mrs": "missus", "dr": "doctor", "alright": "all right"}


def _same(a, b):
    """Words count as matching if identical or a near-identical spelling (colour/color, Priya/Pria)."""
    return a == b or (min(len(a), len(b)) >= 4 and SequenceMatcher(None, a, b).ratio() >= 0.8)


def align(ref: str, hyp: str, english=True):
    """Word error rate and the reference words that were misheard or dropped.
    Tolerates transcription spelling variants and split/joined words (every day / everyday)."""
    r, h = words_of(ref, english), words_of(hyp, english)
    if not r:
        return (1.0 if h else 0.0), []
    if "".join(r) == "".join(h):
        return 0.0, []
    n, m = len(r), len(h)
    INF = 10**9
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = INF
            if i > 0:
                best = min(best, D[i - 1][j] + 1)
            if j > 0:
                best = min(best, D[i][j - 1] + 1)
            if i > 0 and j > 0:
                best = min(best, D[i - 1][j - 1] + (not _same(r[i - 1], h[j - 1])))
            if i > 1 and j > 0 and r[i - 2] + r[i - 1] == h[j - 1]:  # "every day" heard as "everyday"
                best = min(best, D[i - 2][j - 1])
            if i > 0 and j > 1 and r[i - 1] == h[j - 2] + h[j - 1]:  # "everyday" heard as "every day"
                best = min(best, D[i - 1][j - 2])
            D[i][j] = best
    i, j, bad = n, m, []
    while i > 0 or j > 0:
        if i > 1 and j > 0 and r[i - 2] + r[i - 1] == h[j - 1] and D[i][j] == D[i - 2][j - 1]:
            i, j = i - 2, j - 1
        elif i > 0 and j > 1 and r[i - 1] == h[j - 2] + h[j - 1] and D[i][j] == D[i - 1][j - 2]:
            i, j = i - 1, j - 2
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (not _same(r[i - 1], h[j - 1])):
            if not _same(r[i - 1], h[j - 1]):
                bad.append(r[i - 1])
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            bad.append(r[i - 1])
            i -= 1
        else:
            j -= 1
    return D[n][m] / n, bad[::-1]
