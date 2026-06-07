import trainer_app.art_ruler_training as art_ruler_training
import trainer_app.train_policy_qlora_grpo as train_policy_qlora_grpo


def test_trainer_wrappers_import():
    assert art_ruler_training is not None
    assert train_policy_qlora_grpo is not None
