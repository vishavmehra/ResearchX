import re

from services.runtime.orchestrator import detect_intent, render_answer, validate_claims
from services.runtime.trace import new_run_id

SOURCES = {"S1": "iBOT pre-trained with ImageNet-22K achieves 84.4% and 86.6% top-1 accuracy with ViT-B/16 and ViT-L/16, respectively.",
           "S2": "We use a batch size of 16 and an initial learning rate of 0.02."}


def test_run_id_is_rx_ulid_and_sortable():
    a, b = new_run_id(), new_run_id()
    assert re.fullmatch(r"rx_[0-9A-HJKMNP-TV-Z]{26}", a) and a[:13] <= b[:13]


def test_intents():
    assert detect_intent("What learning rate was used to train X?") == "protocol"
    assert detect_intent("Compare ViT and ResNet on ImageNet") == "comparison"
    assert detect_intent("Do these papers contradict each other?") == "contradiction"
    assert detect_intent("What accuracy does X achieve?") == "result"
    assert detect_intent("hello there") == "general"


def test_claims_need_verbatim_quote_known_source_and_numbers_in_quote():
    claims = [
        {"text": "The learning rate is 0.02 with batch size 16.", "sources": ["S2"], "quote": "an initial learning rate of 0.02"},  # 16 not in quote
        {"text": "Batch size 16 and learning rate 0.02 were used.", "sources": ["S2"], "quote": "a batch size of 16 and an initial learning rate of 0.02"},
        {"text": "ViT-B/16 reaches 86.6%.", "sources": ["S1"], "quote": "achieves 84.4% and 86.6% top-1 accuracy ... respectively"},  # ellipsis
        {"text": "Something.", "sources": ["S9"], "quote": "not a real source"},
    ]
    valid, rejected = validate_claims(claims, SOURCES, "What batch size?")
    assert [c["text"] for c in valid] == ["Batch size 16 and learning rate 0.02 were used."]
    assert [r["reason"] for r in rejected] == ["number_not_in_quote", "quote_not_found", "no_valid_source"]


def test_latex_formatting_differences_are_tolerated_but_words_and_numbers_are_not():
    src = {"S1": "\\Box ADE20K: We set the initial learning rate as 0.02, weight decay as 0.0001, crop size as 520\\times 520, batch size as 16."}
    ok = {"text": "The batch size is 16.", "sources": ["S1"], "quote": "crop size as 520×520, batch size as 16."}
    wrong_number = {"text": "The batch size is 32.", "sources": ["S1"], "quote": "crop size as 520×520, batch size as 32."}
    wrong_word = {"text": "The batch size is 16.", "sources": ["S1"], "quote": "crop size as 520×520, batch sizes as 16."}
    valid, rejected = validate_claims([ok, wrong_number, wrong_word], src, "batch size?")
    assert [c["quote"] for c in valid] == [ok["quote"]] and [r["reason"] for r in rejected] == ["quote_not_found", "quote_not_found"]


def test_numbers_from_the_question_are_allowed_and_answer_uses_valid_claims_only():
    valid, _ = validate_claims([{"text": "ViT-B/16 reaches 84.4% top-1 accuracy.", "sources": ["S1"],
                                 "quote": "achieves 84.4% and 86.6% top-1 accuracy with ViT-B/16"}], SOURCES, "accuracy of ViT-B/16?")
    assert len(valid) == 1
    assert render_answer(valid, partial=False) == "ViT-B/16 reaches 84.4% top-1 accuracy. [S1]"
    assert render_answer(valid * 2, partial=True).startswith("- ViT-B/16") and "Only part" in render_answer(valid, partial=True)
