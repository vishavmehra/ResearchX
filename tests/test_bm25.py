import math

from services.retrieval.search.bm25 import BM25, tokenize

DOCS = [
    ("p1#a", "We fine-tune LLaMA-2-7B with learning rate 3e-5 on GSM8K."),
    ("p1#b", "Results on MMLU and GSM8K improve over the baseline."),
    ("p2#a", "A vision transformer (ViT-B/16) trained on ImageNet-1k."),
    ("p3#a", "The baseline uses a learning rate of 0.001 and batch size 64."),
]


def test_tokenizer_keeps_technical_terms_whole_and_split():
    toks = tokenize("LLaMA-2-7B lr 3e-5 ViT-B/16")
    assert {"llama-2-7b", "llama", "2", "7b", "3e-5", "vit-b/16", "vit", "b", "16"} <= set(toks)
    assert "3" not in toks and "5" not in toks  # numbers stay whole


def test_exact_technical_match_ranks_first_and_idf_is_bm25():
    m = BM25().build(DOCS)
    assert m.search("llama-2-7b 3e-5", k=2)[0][0] == "p1#a"
    assert m.search("imagenet-1k vit", k=1)[0][0] == "p2#a"
    assert m.search("unseen-term") == []
    j = m.vocab["gsm8k"]  # df 2 of 4 docs
    assert math.isclose(float(m.idf[j]), math.log(1 + (4 - 2 + 0.5) / (2 + 0.5)), rel_tol=1e-6)
    assert [d for d, _ in m.search("baseline", k=5, keep=lambda d: d.startswith("p3"))] == ["p3#a"]


def test_save_load_roundtrip(tmp_path):
    m = BM25().build(DOCS)
    m.save(tmp_path / "v0.2.0", {"release": "0.2.0"})  # dotted names must not lose their last segment
    assert (tmp_path / "v0.2.0.npz").exists() and (tmp_path / "v0.2.0.json").exists()
    m2 = BM25.load(tmp_path / "v0.2.0")
    for q in ("gsm8k learning rate", "baseline batch size", "vit-b/16"):
        assert m2.search(q, k=4) == m.search(q, k=4)
