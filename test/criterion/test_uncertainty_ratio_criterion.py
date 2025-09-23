import numpy as np

from core.base_criterion import SampleCriterion
from core.module_registry import CRITERION
from data_types.criterion_context import CriterionContext


def test_ratio_computation_basic_true_and_false():
    from modules.criterion.uncertainty_ratio_criterion import (
        UncertaintyRatioCriterion,
    )

    # uncertainties in [0, 1]
    uncertainties = np.array([0.1, 0.6, 0.2, 0.8])
    context = CriterionContext(uncertainty=uncertainties)

    # uncer_thres = 0.5 -> valid: [0.1, 0.2] => 2/4 = 0.5
    crit_true = UncertaintyRatioCriterion(uncer_thres=0.5, ratio_thres=0.4)
    assert crit_true.check_sample_criterion(context) is True

    # Requires strictly greater than ratio_thres; 0.5 > 0.5 is False
    crit_false = UncertaintyRatioCriterion(uncer_thres=0.5, ratio_thres=0.5)
    assert crit_false.check_sample_criterion(context) is False


def test_boundary_values_for_uncertainty_threshold():
    from modules.criterion.uncertainty_ratio_criterion import (
        UncertaintyRatioCriterion,
    )

    # Values straddling threshold 0.5; equality should be excluded (strict <)
    uncertainties = np.array([0.5, 0.49, 0.51, 0.5])
    context = CriterionContext(uncertainty=uncertainties)

    # valid (< 0.5) -> only 0.49 -> 1/4 = 0.25
    criterion = UncertaintyRatioCriterion(uncer_thres=0.5, ratio_thres=0.2)
    assert criterion.check_sample_criterion(context) is True

    criterion = UncertaintyRatioCriterion(uncer_thres=0.5, ratio_thres=0.25)
    assert criterion.check_sample_criterion(context) is False


def test_all_low_uncertainty_triggers_true():
    from modules.criterion.uncertainty_ratio_criterion import (
        UncertaintyRatioCriterion,
    )

    uncertainties = np.array([0.01, 0.02, 0.03, 0.04])
    context = CriterionContext(uncertainty=uncertainties)

    # All < 0.5 -> ratio = 1.0
    criterion = UncertaintyRatioCriterion(uncer_thres=0.5, ratio_thres=0.9)
    assert criterion.check_sample_criterion(context) is True


def test_none_uncertainty_returns_false():
    from modules.criterion.uncertainty_ratio_criterion import (
        UncertaintyRatioCriterion,
    )

    context = CriterionContext(uncertainty=None)
    criterion = UncertaintyRatioCriterion(uncer_thres=0.5, ratio_thres=0.4)
    assert criterion.check_sample_criterion(context) is False


def test_empty_uncertainty_returns_false():
    from modules.criterion.uncertainty_ratio_criterion import (
        UncertaintyRatioCriterion,
    )

    context = CriterionContext(uncertainty=np.array([]))
    criterion = UncertaintyRatioCriterion(uncer_thres=0.5, ratio_thres=0.4)
    assert criterion.check_sample_criterion(context) is False


def test_registry_registration_and_usage():
    from modules.criterion.uncertainty_ratio_criterion import (
        UncertaintyRatioCriterion,
    )

    registered_cls = CRITERION.get("uncertainty_ratio")
    assert registered_cls is UncertaintyRatioCriterion
    assert issubclass(registered_cls, SampleCriterion)

    instance = registered_cls(uncer_thres=0.5, ratio_thres=0.25)
    context = CriterionContext(uncertainty=np.array([0.1, 0.6, 0.7, 0.2]))
    # valid: 0.1, 0.2 -> 2/4 = 0.5 > 0.25
    assert instance.check_sample_criterion(context) is True
