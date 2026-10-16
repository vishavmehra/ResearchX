"""Seeds, splits, dataset versions, evaluation protocols, metric definitions and checkpoints (grounding 0.4.0), their
normalised comparison, and the training-stage classifier of configuration identity. The passages imitate real paper
layouts (implementation paragraphs, appendix tables, captions); no network, no model call."""

import pytest

from services.analysis.configs import stage_of
from services.comparison.fields import compare_field
from services.datasets.schemas import FieldComparisonStatus as S
from services.datasets.schemas import FieldStatus, FieldValue
from services.evidence import grounding as G


def ok(field, value, span, source=None):
    good, reason, v = G.validate(field, value, span, source if source is not None else span)
    return v if good else reason


# ------------------------------------------------------------------------------------------------ seeds
@pytest.mark.parametrize("value,span,want", [
    ("42", "We fix the random seed to 42 for all runs.", 42),
    ("0, 1, 2", "Results are averaged over seeds 0, 1 and 2.", "seeds 0, 1, 2"),
    ("5", "We report the mean and standard deviation over 5 random seeds.", "5 seeds"),
    ("three", "Each configuration is trained with three different random seeds.", "3 seeds"),
    ("1234", "seed=1234", 1234),
])
def test_seed_values_and_policies(value, span, want):
    assert ok("training.seed", value, span) == want


def test_runs_without_the_word_seed_and_other_numbers_are_not_seeds():
    assert ok("training.seed", "3", "Numbers are averaged over 3 runs.") == "no_field_cue", "repeated runs alone do not state a seed policy"
    assert ok("training.batch_size", "42", "We use a batch size of 256 and random seed 42.") == "wrong_field:training.seed"
    assert ok("training.seed", "7", "We train for 7 epochs with seed 0.") == "value_not_in_span"


# ------------------------------------------------------------------------------------------------ splits
@pytest.mark.parametrize("field,value,span,want", [
    ("evaluation.dataset_split", "validation set", "We report top-1 accuracy on the ImageNet validation set.", "validation"),
    ("evaluation.dataset_split", "val", "Results on ADE20K val are single-scale.", "validation"),
    ("evaluation.dataset_split", "test-dev", "We report box AP on COCO test-dev.", "test-dev"),
    ("evaluation.dataset_split", "val2017", "All ablations use COCO val2017.", "val2017"),
    ("dataset.split", "train+val", "The detector is fine-tuned on the train+val set.", "train+val"),
])
def test_splits_are_normalised(field, value, span, want):
    assert ok(field, value, span) == want


def test_split_words_need_their_context_and_role():
    assert ok("evaluation.dataset_split", "test", "We test our model on ImageNet with 8 GPUs.") == "no_field_cue", "the verb is not a split"
    assert ok("evaluation.dataset_split", "train", "Models are trained on the train split.") == "wrong_split_role"
    assert ok("dataset.split", "validation", "We tune on the validation set.") == "wrong_split_role"
    table = "Table 7: Evaluation settings.\nDataset | Split | Metric\nCOCO | val2017 | box AP"
    assert ok("evaluation.dataset_split", "val2017", "val2017", table) == "val2017", "a table cell under a Split column"


# ------------------------------------------------------------------------------------------------ protocols
@pytest.mark.parametrize("value,span,want", [
    ("multi-scale testing with flipping", "At inference we use multi-scale testing with flipping.", "multi-scale + flip"),
    ("linear probing", "We follow the standard linear probing protocol on frozen features.", "linear probe"),
    ("5-shot", "Few-shot (5-shot) evaluation on MMLU.", "5-shot"),
    ("center crop", "At test time we resize to 256 and take a 224 center crop.", "center crop"),
    ("best checkpoint on the validation set", "We evaluate the best checkpoint on the validation set.", "best checkpoint on validation"),
    ("zero-shot", "All results are zero-shot transfer.", "zero-shot"),
])
def test_protocol_tags(value, span, want):
    assert ok("evaluation.evaluation_protocol", value, span) == want


def test_training_augmentation_is_not_an_evaluation_protocol():
    assert ok("evaluation.evaluation_protocol", "random crop", "We use random crop and horizontal flip augmentation during training.") == "not_a_protocol"
    assert ok("evaluation.evaluation_protocol", "multi-scale training", "We adopt multi-scale training as in prior work.") == "not_a_protocol"


def test_appendix_table_only_protocol_is_validated_from_the_table_row():
    appendix = "Table 12: Fine-tuning and evaluation settings.\nconfig | value\nevaluation protocol | multi-scale testing\noptimizer | AdamW"
    assert ok("evaluation.evaluation_protocol", "multi-scale testing", "evaluation protocol | multi-scale testing", appendix) == "multi-scale"


# ------------------------------------------------------------------------------------------------ metric definitions, versions, checkpoints
def test_metric_definition_needs_a_reporting_cue():
    assert ok("evaluation.metric_definition", "top-1 accuracy", "We report top-1 accuracy on ImageNet-1K.") == "top-1 accuracy"
    assert ok("evaluation.metric_definition", "case-sensitive SacreBLEU", "BLEU is computed as case-sensitive SacreBLEU.") == "case-sensitive SacreBLEU"
    assert ok("evaluation.metric_definition", "accuracy", "Accuracy improves with scale.") == "no_field_cue"


def test_dataset_versions_and_citation_years():
    assert ok("dataset.dataset_version", "COCO 2017", "We use COCO 2017 for detection.") == "COCO 2017"
    assert ok("dataset.dataset_version", "ImageNet-21K", "Models are pre-trained on ImageNet-21K.") == "ImageNet-21K"
    assert ok("dataset.dataset_version", "SQuAD v1.1", "We fine-tune on SQuAD v1.1.") == "SQuAD v1.1"
    assert ok("dataset.dataset_version", "2009", "ImageNet (Deng et al., 2009)") == "not_a_version"


def test_checkpoint_needs_a_checkpoint_cue():
    assert ok("model.checkpoint", "ViT-B/16 ImageNet-21k", "The backbone is initialized from the ViT-B/16 ImageNet-21k weights.") == "ViT-B/16 ImageNet-21k"
    assert ok("model.checkpoint", "ViT-B/16", "ViT-B/16 reaches 81.8 top-1.") == "no_field_cue"


def test_span_must_be_in_the_source():
    assert ok("evaluation.dataset_split", "validation", "on the validation set", "We report results on the test set.") == "span_not_in_source"


# ------------------------------------------------------------------------------------------------ normalised comparison
@pytest.mark.parametrize("field,a,b,want", [
    ("evaluation.dataset_split", "val", "ImageNet validation set", "SAME"),
    ("evaluation.dataset_split", "test", "validation", "DIFFERENT"),
    ("evaluation.dataset_split", "#Params(GB)", "test", "UNKNOWN"),
    ("evaluation.metric_definition", "accuracy", "top-1 accuracy", "UNKNOWN"),
    ("evaluation.metric_definition", "Top-1", "top-1 accuracy", "SAME"),
    ("evaluation.evaluation_protocol", "multi-scale", "multi-scale + flip", "UNKNOWN"),
    ("evaluation.evaluation_protocol", "single crop", "1-crop", "SAME"),
    ("evaluation.evaluation_protocol", "linear probe", "fine-tuned", "DIFFERENT"),
    ("evaluation.evaluation_protocol", "official script", "official script", "SAME"),
])
def test_condition_compare(field, a, b, want):
    assert G.condition_compare(field, a, b)[0] == want


def test_comparison_engine_normalises_conditions_and_never_matches_unverified_values():
    def fv(v):
        return FieldValue(status=FieldStatus.KNOWN, raw_value=v, value=v, evidence_ids=["e"])
    assert compare_field("evaluation.dataset_split", fv("val"), fv("validation set")).comparison_status is S.EQUIVALENT
    assert compare_field("evaluation.dataset_split", fv("#Params(GB)"), fv("#Params(GB)")).comparison_status is S.UNKNOWN, \
        "an unrecognisable split is unverified even when both records hold it"
    assert compare_field("evaluation.dataset_split", fv("test"), fv("val")).comparison_status is S.MATERIALLY_DIFFERENT


def test_questions_about_seeds_splits_and_protocols_are_understood():
    r = G.parse_request("Which random seeds and evaluation split were used for MAE on ImageNet?", ["ImageNet"])
    assert set(r.fields) == {"training.seed", "evaluation.dataset_split"} and r.model == "MAE" and r.dataset == "ImageNet"
    assert G.parse_request("What evaluation protocol does BEiT use on ADE20K?", ["ADE20K"]).fields == ["evaluation.evaluation_protocol"]
    assert G.parse_request("What learning rate does BEiT use on ADE20K?", ["ADE20K"]).fields == ["training.learning_rate"]


# ------------------------------------------------------------------------------------------------ training stage
@pytest.mark.parametrize("text,value,section,want", [
    ("We pre-train for 800 epochs with a batch size of 4096.", "4096", None, "pretraining"),
    ("The pre-trained model is fine-tuned with a learning rate of 1e-4.", "1e-4", None, "finetuning"),
    ("We use AdamW with a learning rate of 5e-4.", "5e-4", "A.2 Fine-tuning details", "finetuning"),
    ("We use AdamW with a learning rate of 5e-4.", "5e-4", "Implementation details", None),
    ("For linear probing we train a classifier with lr 0.1.", "0.1", None, "finetuning"),
])
def test_stage_of(text, value, section, want):
    assert stage_of(text, value, section)[0] == want


def test_one_sentence_two_stages_each_value_takes_its_nearest_stage():
    s = "During pre-training we use a batch size of 4096; for fine-tuning, a batch size of 1024."
    assert stage_of(s, "4096")[0] == "pretraining" and stage_of(s, "1024")[0] == "finetuning"
