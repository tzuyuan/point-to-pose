from point2pose.core.base_criterion import SampleCriterion
from point2pose.core.module_registry import CRITERION
from point2pose.data_types.criterion_context import CriterionContext


def test_default_returns_true_every_time():
    # Default return_every_n_iterations is 1
    from point2pose.modules.criterion.iteration_criterion import IterationCriterion

    criterion = IterationCriterion()
    # Create empty context - should increment internal counter
    context = CriterionContext()
    assert criterion.check_sample_criterion(context) is True
    assert criterion.check_sample_criterion(context) is True
    assert criterion.check_sample_criterion(context) is True


def test_every_n_iterations_behavior():
    from point2pose.modules.criterion.iteration_criterion import IterationCriterion

    criterion = IterationCriterion(return_every_n_iterations=3)
    context = CriterionContext()
    results = [criterion.check_sample_criterion(context) for _ in range(6)]
    assert results == [False, False, True, False, False, True]


def test_registry_registration_and_usage():
    # Import ensures decorator runs and registers the class
    from point2pose.modules.criterion.iteration_criterion import IterationCriterion

    registered_cls = CRITERION.get("iteration")
    assert registered_cls is IterationCriterion
    assert issubclass(registered_cls, SampleCriterion)

    # Basic sanity check via instance from registry-retrieved class
    instance = registered_cls(return_every_n_iterations=2)
    context = CriterionContext()
    assert instance.check_sample_criterion(context) is False
    assert instance.check_sample_criterion(context) is True


def test_check_sample_criterion_with_cur_iter_argument():
    from point2pose.modules.criterion.iteration_criterion import IterationCriterion

    criterion = IterationCriterion(return_every_n_iterations=4)
    # Directly set cur_iter via context; 4 % 4 == 0 -> True
    context_with_iter = CriterionContext(cur_iter=4)
    assert criterion.check_sample_criterion(context_with_iter) is True
    # Next call without cur_iter in context increments to 5 -> 5 % 4 == 1 -> False
    context_empty = CriterionContext()
    assert criterion.check_sample_criterion(context_empty) is False
