"""Multilingual post scoring in [-1, 1].

Generic sentiment models miss retail slang such as 떡상 ("moon"), 물렸다 ("bagholding"), 插水 ("plunged") or
入貨 ("loading up"). Each post therefore gets a finance-tuned transformer for its language blended with a slang
lexicon. If transformers can't be loaded the lexicon runs alone, and the report states which backend was used.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BULLISH: dict[str, float] = {
    # Korean
    "떡상": 1.0, "따상": 0.9, "불기둥": 0.9, "풀매수": 0.9, "상한가": 0.8, "가즈아": 0.8, "호재": 0.8, "어닝서프라이즈": 0.9,
    "어닝 서프라이즈": 0.9, "목표주가 상향": 0.8, "목표가 상향": 0.8, "컨센서스 상회": 0.8, "급등": 0.7, "신고가": 0.7, "대박": 0.6,
    "저평가": 0.5, "줍줍": 0.5, "반등": 0.5, "오른다": 0.5, "상승": 0.4, "돌파": 0.4, "익절": 0.4, "오를": 0.4, "매수": 0.3, "존버": 0.3,
    # Cantonese / Chinese
    "爆升": 0.9, "狂升": 0.8, "急升": 0.8, "大升": 0.8, "勝預期": 0.8, "超預期": 0.8, "超预期": 0.8, "利好": 0.8, "目標價上調": 0.8,
    "目标价上调": 0.8, "大涨": 0.8, "唱好": 0.7, "升穿": 0.6, "看好": 0.6, "睇好": 0.6, "看多": 0.6, "加倉": 0.6, "增持": 0.6, "發達": 0.6,
    "入貨": 0.5, "抄底": 0.5, "撈底": 0.5, "食糊": 0.5, "上涨": 0.5, "新高": 0.5, "買入": 0.4, "买入": 0.4, "反彈": 0.4, "反弹": 0.4,
    "突破": 0.4, "上望": 0.4, "上調": 0.4, "回購": 0.4, "回购": 0.4, "賺": 0.4, "勁": 0.3, "升": 0.3, "涨": 0.3,
    # English
    "beats estimates": 0.8, "beat estimates": 0.8, "upgrade": 0.7, "upgraded": 0.7, "surge": 0.7, "surges": 0.7, "soar": 0.7, "soars": 0.7,
    "rally": 0.6, "rallies": 0.6, "record high": 0.6, "raises guidance": 0.8, "buyback": 0.4, "bullish": 0.8, "outperform": 0.5, "gains": 0.3,
}
BEARISH: dict[str, float] = {
    # Korean
    "떡락": 1.0, "폭락": 0.9, "하한가": 0.9, "어닝쇼크": 0.9, "어닝 쇼크": 0.9, "악재": 0.8, "망했": 0.8, "망함": 0.8, "나락": 0.8, "곡소리": 0.8,
    "목표주가 하향": 0.8, "목표가 하향": 0.8, "컨센서스 하회": 0.8, "급락": 0.7, "설거지": 0.7, "한강": 0.7, "손절": 0.6, "물렸": 0.6, "쓰레기": 0.5,
    "사기": 0.5, "공매도": 0.4, "하락": 0.4, "탈출": 0.4, "매도": 0.3, "내린다": 0.5,
    # Cantonese / Chinese
    "暴跌": 0.9, "狂跌": 0.8, "急跌": 0.8, "大跌": 0.8, "插水": 0.8, "爆煲": 0.8, "利淡": 0.8, "遜預期": 0.8, "逊预期": 0.8, "目標價下調": 0.8,
    "目标价下调": 0.8, "唱淡": 0.7, "仆街": 0.7, "損手": 0.7, "割肉": 0.7, "跌穿": 0.7, "接火棒": 0.7, "看淡": 0.6, "睇淡": 0.6, "看空": 0.6,
    "蝕": 0.6, "减持": 0.6, "減持": 0.6, "沽空": 0.5, "止蝕": 0.5, "走貨": 0.5, "下跌": 0.5, "下調": 0.4, "沽出": 0.4, "韭菜": 0.4, "跌": 0.3,
    # English
    "misses estimates": 0.8, "missed estimates": 0.8, "downgrade": 0.7, "downgraded": 0.7, "plunge": 0.8, "plunges": 0.8, "slump": 0.7, "slumps": 0.7,
    "tumble": 0.7, "tumbles": 0.7, "cuts guidance": 0.8, "probe": 0.5, "bearish": 0.8, "lawsuit": 0.5, "underperform": 0.5, "losses": 0.4,
}
# Self-reported positioning: what the crowd says it is doing, as opposed to how it feels.
LONG_STANCE = re.compile(r"풀매수|추매|물타기|평단|보유\s*중|매수했|샀다|존버|入咗|買咗|揸住|坐貨|加倉|溝淡|持有|bought|holding|long\b", re.I)
EXIT_STANCE = re.compile(r"손절했|익절했|매도했|팔았|전량\s*매도|沽咗|走貨|止蝕咗|清倉|全沽|sold|short\b|exited", re.I)
NEGATORS = re.compile(r"(안\s*|못\s*|(唔|不|冇|未|沒|没|咪|别)(會|会|可能|能|想|好|得)?|not\s+|no\s+|never\s+|isn't\s+|won't\s+)$", re.I)

_TERMS = sorted([(t, w) for t, w in BULLISH.items()] + [(t, -w) for t, w in BEARISH.items()], key=lambda x: -len(x[0]))
_TERM_RE = re.compile("|".join(re.escape(t) for t, _ in _TERMS), re.I)
_TERM_WEIGHT = {t.lower(): w for t, w in _TERMS}


def detect_language(text: str) -> str:
    hangul = len(re.findall(r"[가-힣]", text))
    cjk = len(re.findall(r"[一-鿿]", text))
    if hangul and hangul >= cjk:
        return "ko"
    return "zh" if cjk else "en"


def lexicon_score(text: str) -> tuple[float, int]:
    """(score, number of hits). Regex alternation is longest-first, so 大跌 isn't also counted as 跌."""
    total, hits = 0.0, 0
    for match in _TERM_RE.finditer(text):
        weight = _TERM_WEIGHT[match.group(0).lower()]
        if NEGATORS.search(text[max(0, match.start() - 8):match.start()]):
            weight = -0.6 * weight
        total += weight
        hits += 1
    return math.tanh(total), hits


def stance(text: str) -> int:
    long_hit, exit_hit = bool(LONG_STANCE.search(text)), bool(EXIT_STANCE.search(text))
    return int(long_hit) - int(exit_hit)


class TransformerBackend:
    """One model per language. Label values convert class probabilities into an expected score."""

    MODELS = {
        "ko": "snunlp/KR-FinBert-SC",
        "en": "ProsusAI/finbert",
        "zh": "tabularisai/multilingual-sentiment-analysis",
    }
    VALUES = {"very negative": -1.0, "negative": -0.6, "neutral": 0.0, "positive": 0.6, "very positive": 1.0}
    THREE_CLASS = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

    def __init__(self) -> None:
        import torch  # imported here so lexicon-only installs don't need it
        from transformers import pipeline
        self._pipeline, self._device = pipeline, 0 if torch.cuda.is_available() else -1
        self._loaded: dict[str, object] = {}

    def _get(self, lang: str):
        if lang not in self._loaded:
            self._loaded[lang] = self._pipeline("text-classification", model=self.MODELS[lang], device=self._device,
                                                top_k=None, truncation=True, max_length=256)
        return self._loaded[lang]

    def score(self, texts: list[str], lang: str) -> list[float]:
        if not texts:
            return []
        values = self.VALUES if lang == "zh" else self.THREE_CLASS
        out = []
        for result in self._get(lang)(texts, batch_size=16):
            out.append(float(sum(values.get(r["label"].lower(), 0.0) * r["score"] for r in result)))
        return out


class Scorer:
    def __init__(self, backend: str, cache_path: Path | None = None) -> None:
        self.transformer: TransformerBackend | None = None
        self.used = "lexicon"
        if backend in ("auto", "transformer"):
            try:
                self.transformer = TransformerBackend()
                self.used = "transformer+lexicon"
            except ImportError as exc:
                if backend == "transformer":
                    raise
                log.warning("transformers unavailable (%s); using the lexicon only", exc)
        self.cache_path = cache_path
        self.cache: dict[str, float] = {}
        if cache_path and cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def _model_scores(self, texts: list[str], langs: list[str]) -> list[float | None]:
        out: list[float | None] = [None] * len(texts)
        if self.transformer is None:
            return out
        keys = [hashlib.sha1(f"{lang}|{t}".encode()).hexdigest()[:16] for t, lang in zip(texts, langs)]
        for lang in sorted(set(langs)):
            todo = sorted({k: t for k, t, l in zip(keys, texts, langs) if l == lang and k not in self.cache}.items())
            if todo:
                log.info("scoring %d new %s texts", len(todo), lang)
                scores = self.transformer.score([t for _, t in todo], lang)
                self.cache.update({k: s for (k, _), s in zip(todo, scores)})
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
        return [self.cache[k] for k in keys]

    def score_frame(self, posts: pd.DataFrame) -> pd.DataFrame:
        """Adds lang, score, model_score, lex_score and stance columns."""
        if posts.empty:
            return posts.assign(lang=[], score=[], model_score=[], lex_score=[], stance=[])
        df = posts.copy()
        text = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip().str.slice(0, 1200)
        df["lang"] = text.map(detect_language)
        lex = text.map(lexicon_score)
        df["lex_score"] = [s for s, _ in lex]
        hits = np.array([h for _, h in lex])
        df["stance"] = text.map(stance)
        model = self._model_scores(text.tolist(), df["lang"].tolist())
        df["model_score"] = pd.to_numeric(pd.Series(model, index=df.index, dtype=object), errors="coerce")
        has_model = df["model_score"].notna().to_numpy()
        combined = np.where(hits > 0, 0.5 * df["model_score"].fillna(0).to_numpy() + 0.5 * df["lex_score"].to_numpy(),
                            df["model_score"].fillna(0).to_numpy())
        df["score"] = np.where(has_model, combined, df["lex_score"].to_numpy())
        return df
