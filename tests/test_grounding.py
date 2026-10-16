"""Regression tests for field-level evidence grounding (the 2026-10-04 false-abstention bug and its siblings).
Deterministic: no network, no model. S1 below is the real ResearchX corpus paragraph (paper:arxiv:1909.11065#S4.SS2.p3)."""

from services.evidence.grounding import (Fact, acronyms, assess, canon, config_match, coverage, dataset_context, parse_request, render,
                                         validate)

S1 = ("\\Box ADE20K: We set the initial learning rate as 0.02, weight decay as 0.0001, crop size as 520\\times 520, "
      "batch size as 16 and and training iterations as 150K if not specified.")
SEGNEXT = "For segmentation experiments ... The batch size is set to 8 for the Cityscapes dataset and 16 for all the other datasets."
QUESTION = "What learning rate and batch size were used to train HRNetV2 + OCR on ADE20K?"


def corpus_fact(field, raw, value, span, start, sid="S1", chunk="paper:arxiv:1909.11065#S4.SS2.p3", config="EXACT", exps=("exp:02f66c2765400aa1",)):
    text = S1 if sid == "S1" else SEGNEXT
    ok, reason, v = validate(field, value, span, text)
    return Fact(field=field, raw=raw, value=v, span=span, chunk_id=chunk, sid=sid, start=start, end=start + len(span), valid=ok,
                reason=reason, config=config, experiment_ids=exps, dataset_ok=dataset_context(text, span, "ADE20K", ["Cityscapes", "ADE20K"]))


def test_1_exact_parameter_answer_from_corpus_evidence():
    req = parse_request(QUESTION, ["ADE20K"])
    assert req.fields == ["training.learning_rate", "training.batch_size"] and req.model == "HRNetV2 + OCR" and req.dataset == "ADE20K"
    cls, _ = config_match(req, ["HRNetV2 + OCR + RMI (PaddleClas pretrained)"], ["ADE20K"], {})
    assert cls == "EXACT"
    facts = [corpus_fact("training.learning_rate", "0.02", 0.02, "initial learning rate as 0.02", 24),
             corpus_fact("training.batch_size", "16", 16.0, "batch size as 16", 107)]
    assert all(f.valid and f.dataset_ok for f in facts)
    status, results = assess(req, facts, relevant_sources=4)
    assert status == "SUFFICIENT" and [r.status for r in results] == ["SUPPORTED", "SUPPORTED"]
    text, claims = render(req, status, results)
    assert "- Learning rate: 0.02 [S1]" in text and "- Batch size: 16 [S1]" in text
    assert {c["quote"] for c in claims} == {"initial learning rate as 0.02", "batch size as 16"}


def test_2_partial_evidence_marks_missing_field_unknown():
    req = parse_request(QUESTION, ["ADE20K"])
    status, results = assess(req, [corpus_fact("training.learning_rate", "0.02", 0.02, "initial learning rate as 0.02", 24)], 3)
    assert status == "PARTIAL" and [r.status for r in results] == ["SUPPORTED", "UNKNOWN"]
    assert "- Batch size: not stated in the retrieved evidence" in render(req, status, results)[0]


def test_3_numerical_contradiction_is_reported_not_resolved():
    req = parse_request("What learning rate was used to train HRNetV2 + OCR on ADE20K?", ["ADE20K"])
    a = Fact("training.learning_rate", "0.02", "initial learning rate as 0.02", "paper:a#p1", "S1", 0.02, valid=True, config="EXACT", dataset_ok=True)
    b = Fact("training.learning_rate", "0.001", "learning rate of 0.001", "paper:b#p4", "S2", 0.001, valid=True, config="EXACT", dataset_ok=True)
    status, results = assess(req, [a, b], 2)
    assert status == "CONFLICTING_EVIDENCE" and results[0].status == "CONFLICTING"
    assert "sources disagree: 0.02 [S1]; 0.001 [S2]" in render(req, status, results)[0]


def test_4_distractor_papers_do_not_answer_for_the_wrong_configuration():
    req = parse_request(QUESTION, ["ADE20K"])
    assert config_match(req, ["SegNeXt-L"], ["ADE20K"], {})[0] == "RELATED"  # same dataset, different model
    distractor = corpus_fact("training.batch_size", "16", 16.0, "16 for all the other datasets", 0, sid="S2", chunk="paper:segnext#p3",
                             config="RELATED", exps=("exp:segnext",))
    exact = corpus_fact("training.batch_size", "16", 16.0, "batch size as 16", 107)
    status, results = assess(parse_request("What batch size was used to train HRNetV2 + OCR on ADE20K?", ["ADE20K"]), [distractor, exact], 2)
    assert status == "SUFFICIENT" and {f.sid for f in results[0].facts} == {"S1"}


def test_5_absent_parameter_is_insufficient_and_no_source_is_distinct():
    req = parse_request("Which optimizer was used to train HRNetV2 + OCR on ADE20K?", ["ADE20K"])
    assert req.fields == ["training.optimizer"]
    assert validate("training.optimizer", "SGD", "initial learning rate as 0.02", S1)[0] is False
    assert assess(req, [], relevant_sources=3)[0] == "INSUFFICIENT_EVIDENCE"
    assert assess(req, [], relevant_sources=0)[0] == "NO_RELEVANT_SOURCE_FOUND"


def test_6_aliases_only_when_the_source_establishes_them():
    ocr = acronyms("Segmentation Transformer: Object-Contextual Representations (OCR) for Semantic Segmentation")
    assert ocr == {"ocr": "object contextual representations"}
    q = "HRNetV2 + Object-Contextual Representations"
    assert coverage(q, ["HRNetV2 + OCR"], ocr) == 1.0
    assert coverage(q, ["HRNetV2 + OCR"], {}) < 1.0  # without the source-established alias nothing is assumed
    text_ocr = acronyms("Scene text recognition with optical character recognition (OCR) pipelines")
    assert coverage(q, ["HRNetV2 + OCR"], text_ocr) < 1.0  # a different OCR does not count


def test_7_unsupported_numerical_claim_is_rejected():
    assert validate("training.learning_rate", "0.01", "initial learning rate as 0.02", S1) == (False, "value_not_in_span", None)
    assert validate("training.learning_rate", "0.02", "learning rate as 0.02 is great", S1)[1] == "span_not_in_source"


def test_8_semantic_and_numeric_wording_variation_is_accepted():
    assert validate("training.learning_rate", "0.02", "initial learning rate as 0.02", S1)[:2] == (True, "ok")
    assert validate("training.learning_rate", "2e-2", "initial learning rate as 0.02", S1)[:2] == (True, "ok")
    src = "We use a learning rate of 2 \\times 10^{-2} for all runs."
    assert validate("training.learning_rate", "0.02", "learning rate of 2 \\times 10^{-2}", src)[:2] == (True, "ok")
    assert validate("training.steps", "150K", "training iterations as 150K", S1)[:2] == (True, "ok")
    assert validate("training.steps", "150000", "training iterations as 150K", S1)[:2] == (True, "ok")


def test_9_same_number_wrong_field_is_invalid():
    src = "We train with SGD, weight decay as 0.02 and momentum 0.9."
    assert validate("training.learning_rate", "0.02", "weight decay as 0.02", src) == (False, "wrong_field:training.weight_decay", None)
    assert validate("training.weight_decay", "0.0001", "weight decay as 0.0001", S1)[:2] == (True, "ok")
    assert validate("training.batch_size", "520", "crop size as 520\\times 520", S1)[0] is False
    warm = "We use linear warmup steps of 10K and train for 1M steps."
    assert validate("training.warmup_steps", "10K", "warmup steps of 10K", warm)[:2] == (True, "ok")  # compound cue, not "steps"
    assert validate("training.steps", "10K", "warmup steps of 10K", warm)[1] == "wrong_field:training.warmup_steps"
    assert validate("training.steps", "1M", "train for 1M steps", warm)[:2] == (True, "ok")
    assert parse_request("What number of warmup steps was used to train SEED?", []).fields == ["training.warmup_steps"]


def test_10_citation_points_to_the_exact_span_of_the_cited_source():
    req = parse_request(QUESTION, ["ADE20K"])
    facts = [corpus_fact("training.learning_rate", "0.02", 0.02, "initial learning rate as 0.02", 24),
             corpus_fact("training.batch_size", "16", 16.0, "batch size as 16", 107)]
    status, results = assess(req, facts, 1)
    for c in render(req, status, results)[1]:
        assert c["sources"] == ["S1"] and S1[c["start"]:c["end"]] == c["quote"]


def test_dataset_context_uses_the_nearest_preceding_dataset():
    text = "Cityscapes: we use a learning rate of 0.01. ADE20K: we set the initial learning rate as 0.02."
    assert dataset_context(text, "initial learning rate as 0.02", "ADE20K", ["Cityscapes", "ADE20K"]) is True
    assert dataset_context(text, "initial learning rate as 0.02", "Cityscapes", ["Cityscapes", "ADE20K"]) is False
    assert dataset_context(text, "learning rate of 0.01", "Cityscapes", ["Cityscapes", "ADE20K"]) is True


def test_metric_questions_parse_to_score_requests():
    req = parse_request("What top-1 accuracy does ViT-B/16 reach on ImageNet?", ["ImageNet"])
    assert req.fields == ["evaluation.score"] and req.metric == "top-1 accuracy" and req.model == "ViT-B/16" and req.dataset == "ImageNet"
    assert req.labels == ["top-1 accuracy"]
    # "Top 1" with a space is still top-1: plain "accuracy" would also admit the Top-5 outcome of the same model
    assert parse_request("What Top 1 Accuracy does Barlow Twins (ResNet-50) achieve on ImageNet?", ["ImageNet"]).metric == "Top 1 Accuracy"


def test_generic_and_unknown_dataset_names():
    # the entity table lists "Benchmark" as a benchmark name; a name the corpus does not know still constrains the question
    req = parse_request("What learning rate was used to train QuasarFlux-9 on the Zentrix-77 benchmark?", ["Benchmark"])
    assert (req.model, req.dataset) == ("QuasarFlux-9", "Zentrix-77")
    req = parse_request("What batch size was used for BERT pre-training on 64 TPU chips?", [])
    assert (req.model, req.dataset) == ("BERT", None)
    req = parse_request("What smoothed BLEU-4 score does RoBERTa achieve on CodeSearchNet JavaScript?", ["CodeSearchNet", "CodeSearchNet - JavaScript"])
    assert (req.model, req.dataset, req.metric) == ("RoBERTa", "CodeSearchNet - JavaScript", "BLEU-4")


def test_dataset_and_model_matching_use_whole_tokens():
    js = parse_request("What smoothed BLEU-4 score does RoBERTa achieve on CodeSearchNet JavaScript?", ["CodeSearchNet - JavaScript"])
    assert config_match(js, ["RoBERTa"], ["CodeSearchNet - JavaScript"], {})[0] == "EXACT"
    assert config_match(js, ["RoBERTa"], ["CodeSearchNet - Java"], {})[0] == "RELATED"  # Java is not JavaScript
    assert config_match(js, ["RoBERTa"], ["CodeSearchNet"], {})[0] == "STRONG"  # broader benchmark: never outvotes EXACT
    c10 = parse_request("What learning rate was used to train ResNet-18 on CIFAR-10?", ["CIFAR-10"])
    assert config_match(c10, ["ResNet-18"], ["CIFAR-100"], {})[0] == "RELATED"
    gpt = parse_request("What batch size was used to train GPT-3 on MMLU?", ["MMLU"])
    assert config_match(gpt, ["Llama 3"], ["MMLU"], {})[0] == "RELATED"  # a shared version number is not the same model
    assert canon(2.0) == "2.0"  # experiment fields can be stored as numbers


def test_question_reading_regressions_from_the_grounding_eval():
    # the entity table also lists "Accuracy" as a benchmark: a metric name never becomes the dataset
    req = parse_request("What Top 1 Accuracy does data2vec (ViT-H) achieve on ImageNet?", ["Accuracy", "ImageNet"])
    assert (req.model, req.dataset, req.fields) == ("data2vec (ViT-H)", "ImageNet", ["evaluation.score"])
    # a capitalised word ending the model name is part of the name, not a verb
    assert parse_request("What learning rate was used to train RL Training on Sokoban?", ["Sokoban"]).model == "RL Training"
    # the dataset is named after the last "on"; an entity inside the model name is not the dataset
    req = parse_request("What batch size was used to train R1 - 8B + UnsafeChain selected on JBB?", ["UnsafeChain"])
    assert (req.model, req.dataset) == ("R1 - 8B + UnsafeChain selected", "JBB")
    # a field word inside the model name is not what is asked
    req = parse_request("What Accuracy does Dropout 0.1 achieve on XNLI?", ["XNLI"])
    assert (req.fields, req.metric, req.model) == (["evaluation.score"], "Accuracy", "Dropout 0.1")


def test_several_values_from_one_table_are_ambiguous_not_a_conflict():
    req = parse_request("What BLEU-1 does Artpedia achieve on Artpedia?", ["Artpedia"])
    rows = [Fact("evaluation.score", v, v, "paper:x#S4.T2", "S1", float(v), valid=True, config="EXACT", dataset_ok=True,
                 experiment_ids=("exp:artpedia",)) for v in ("24.7", "29.9", "32.6")]  # one experiment record over a whole table
    status, results = assess(req, rows, 1)
    assert status == "AMBIGUOUS_EXPERIMENT_MATCH" and results[0].status == "AMBIGUOUS"


def test_asked_dataset_outranks_its_variants_and_values_display_without_latex():
    req = parse_request("What Top 1 Accuracy does Barlow Twins (ResNet-50) achieve on ImageNet?", ["ImageNet"])
    full = Fact("evaluation.score", "73.2", "73.2", "p#T1", "S1", 73.2, valid=True, config="EXACT", dataset_ok=True, experiment_ids=("exp:in",),
                ds_exact=True)
    one_pct = Fact("evaluation.score", "55.0", "55.0", "p#T2", "S2", 55.0, valid=True, config="EXACT", dataset_ok=True, experiment_ids=("exp:in1",),
                   ds_exact=False)  # "ImageNet - 1% labeled data"
    status, results = assess(req, [full, one_pct], 1)
    assert status == "SUFFICIENT" and [f.sid for f in results[0].facts] == ["S1"]
    assert assess(req, [one_pct], 1)[0] == "SUFFICIENT"  # a variant alone still answers (its configuration is shown on the source)
    from services.evidence.grounding import display, dataset_exact
    assert display("73.9\\pm3.1") == "73.9±3.1" and dataset_exact("ImageNet", ["ImageNet - 1% labeled data"]) is False


def test_validation_regressions_from_the_grounding_eval():
    src = "We fine-tune with a learning rate of 3\\cdot 10^{-5} and trained with SGD and momentum 0.9."
    assert validate("training.learning_rate", "3e-5", "learning rate of 3\\cdot 10^{-5}", src)[:2] == (True, "ok")
    assert validate("training.optimizer", "SGD", "trained with SGD", src)[:2] == (True, "ok")  # an optimizer name needs no cue word
    sched = "We use a cosine schedule. The model is linear in its inputs."
    assert validate("training.learning_rate_schedule", "linear", "linear in its inputs", sched)[1] == "no_field_cue"  # generic words still do
    table = "Table 8: Training hyperparameters\nConfig | Pre-training | SFT\nBatch size | 4096 | 128\nLearning rate | 2e-4 | 1e-5\nLR schedule | cosine | cosine"
    assert validate("training.batch_size", "4096", "4096", table)[:2] == (True, "ok")  # bare cell labelled by its row
    assert validate("training.learning_rate", "128", "128", table)[0] is False          # labelled batch size, not learning rate
    assert validate("training.learning_rate", "1e-5", "1e-5", table)[:2] == (True, "ok")
    assert validate("training.learning_rate_schedule", "cosine", "cosine", table)[:2] == (True, "ok")
    warm = "We warm up the learning rate over the first 5K steps and train for 100K steps."
    assert validate("training.warmup_steps", "5K", "warm up the learning rate over the first 5K steps", warm)[:2] == (True, "ok")


def test_another_model_variant_never_answers():
    def cls(model, name):
        return config_match(parse_request(f"What learning rate was used to train {model} on ImageNet?", ["ImageNet"]), [name], ["ImageNet"], {})[0]
    assert cls("ViT-L/16", "ViT-B/16") == "RELATED"
    assert cls("ViT-B/16", "ViT-B/32") == "RELATED"
    assert cls("ResNet-50", "ResNet-101") == "RELATED"
    assert cls("Llama 2 7B", "Llama 2 13B") == "RELATED"
    assert cls("BERT-large", "BERT-base") == "RELATED"
    assert cls("BERT-large", "BERT") == "STRONG"  # the general name: a close match, labelled as such
    assert cls("ViT", "ViT-L/16") == "EXACT"     # every asked token present (several variants then make it ambiguous)


def test_close_configuration_is_partial_and_labelled():
    req = parse_request("What learning rate was used to train BERT-large on SQuAD?", ["SQuAD"])
    fact = Fact("training.learning_rate", "3e-5", "learning rate of 3e-5", "paper:bert#p1", "S1", 3e-5, valid=True, config="STRONG",
                dataset_ok=True, config_detail="BERT", experiment_ids=("exp:bert",))
    status, results = assess(req, [fact], 1)
    assert status == "PARTIAL" and results[0].close
    text = render(req, status, results)[0]
    assert text.startswith("No experiment in the evidence matches BERT-large on SQuAD exactly")
    assert "- Learning rate: 3e-5 [S1] (closest matching configuration: BERT; no exact match)" in text


def test_paper_table_and_leaderboard_record_disagreeing_is_a_conflict():
    table = ("Table 4: Results on Code-to-Documentation generation, evaluated with smoothed BLEU-4 score.\n"
             "model | ruby | javascript | go | python | java | php | overall\nRoBERTa | 11.17 | 11.90 | 17.72 | 18.14 | 16.47 | 24.02 | 16.57")
    record = ('Papers With Code leaderboard "Code Documentation Generation" / "CodeSearchNet - JavaScript" (row 7): '
              'model "RoBERTa" reports {"Smoothed BLEU-4": "5.72"}')
    req = parse_request("What smoothed BLEU-4 score does RoBERTa achieve on CodeSearchNet JavaScript?", ["CodeSearchNet - JavaScript"])
    facts = []
    for sid, span, value, text, origin in (("S1", "11.90", 11.9, table, "corpus evidence"), ("S2", record, 5.72, record, "dataset record")):
        ok, reason, v = validate("evaluation.score", value, span, text)
        facts.append(Fact("evaluation.score", str(value), span, f"src{sid}", sid, v, valid=ok, reason=reason, config="EXACT", dataset_ok=True,
                          experiment_ids=("exp:78879e1a372e8b20",), origin=origin))
    assert all(f.valid for f in facts)
    status, results = assess(req, facts, 2)
    assert status == "CONFLICTING_EVIDENCE" and results[0].status == "CONFLICTING"
    assert "- BLEU-4: sources disagree: 11.9 [S1]; 5.72 [S2]" in render(req, status, results)[0]
