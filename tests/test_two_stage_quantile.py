import os
import tempfile
import unittest

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig-codex"))

from lassonet import LassoNetRegressor, QuantileLassoNetRegressor

from twostage_lassonet_v2 import TwoStageLassoNetRegressorV2, TwoStageQuantileLassoNetRegressor
from twostage_lassonet_v2.utils import mean_pinball_loss

from _twostage_lassonet_v2_utils import make_small_group_data, tiny_estimator_kwargs


def tiny_quantile_kwargs(**overrides):
    params = tiny_estimator_kwargs(**overrides)
    params.setdefault("alpha_values", [0.0, 0.5, 1.0])
    return params


class TwoStageQuantileTest(unittest.TestCase):
    def test_invalid_tau(self):
        with self.assertRaises(ValueError):
            TwoStageQuantileLassoNetRegressor(tau=0.0, **tiny_quantile_kwargs())
        with self.assertRaises(ValueError):
            TwoStageQuantileLassoNetRegressor(tau=1.0, **tiny_quantile_kwargs())

    def test_same_tau_in_both_stages(self):
        X, y, groups = make_small_group_data()
        model = TwoStageQuantileLassoNetRegressor(tau=0.7, **tiny_quantile_kwargs()).fit(X, y, groups)
        self.assertAlmostEqual(model.stage1_model_.tau, 0.7)
        for group in model.groups_:
            self.assertAlmostEqual(model.group_models_[group].estimator.tau, 0.7)

    def test_fit_smoke_and_prediction_shape(self):
        X, y, groups = make_small_group_data()
        model = TwoStageQuantileLassoNetRegressor(tau=0.5, **tiny_quantile_kwargs()).fit(X, y, groups)
        prediction = model.predict(X, groups)
        self.assertEqual(prediction.shape, y.shape)
        self.assertTrue(np.isfinite(prediction).all())

    def test_unknown_groups_raise(self):
        X, y, groups = make_small_group_data()
        model = TwoStageQuantileLassoNetRegressor(tau=0.5, **tiny_quantile_kwargs()).fit(X, y, groups)
        bad_groups = groups.copy()
        bad_groups[0] = "BAD"
        with self.assertRaisesRegex(ValueError, "Unknown group"):
            model.predict(X, bad_groups)

    def test_score_is_negative_pinball_loss(self):
        X, y, groups = make_small_group_data()
        model = TwoStageQuantileLassoNetRegressor(tau=0.3, **tiny_quantile_kwargs()).fit(X, y, groups)
        prediction = model.predict(X, groups)
        expected = mean_pinball_loss(y, prediction, 0.3)
        self.assertAlmostEqual(-model.score(X, y, groups), expected, places=6)

    def test_no_test_data_needed_for_fit(self):
        X, y, groups = make_small_group_data()
        model = TwoStageQuantileLassoNetRegressor(tau=0.5, **tiny_quantile_kwargs()).fit(X, y, groups)
        self.assertTrue(model.is_fitted_)

    def test_final_state_matches_selected_alpha(self):
        X, y, groups = make_small_group_data()
        model = TwoStageQuantileLassoNetRegressor(tau=0.5, **tiny_quantile_kwargs()).fit(X, y, groups)
        selected = next(result for result in model.tuning_results_ if result.alpha == model.alpha_)
        expected_lambdas = {group: result.selected_lambda for group, result in selected.group_results.items()}
        self.assertEqual(model.group_lambdas_, expected_lambdas)

    def test_stage1_support_uses_support_tol(self):
        X, y, groups = make_small_group_data()
        model = TwoStageQuantileLassoNetRegressor(
            tau=0.5,
            support_tol=1e-6,
            **tiny_quantile_kwargs(),
        ).fit(X, y, groups)
        expected = np.abs(model.stage1_theta_) > model.support_tol
        self.assertTrue(np.array_equal(model.stage1_support_, expected))

    def test_stage2_strict_support_identity(self):
        X, y, groups = make_small_group_data()
        model = TwoStageQuantileLassoNetRegressor(tau=0.5, **tiny_quantile_kwargs()).fit(X, y, groups)
        for group in model.groups_:
            expected = model.group_supports_[group] & ~model.stage1_support_
            self.assertTrue(np.array_equal(model.new_supports_[group], expected))

    def test_original_classes_unchanged(self):
        X, y, groups = make_small_group_data()
        base = TwoStageLassoNetRegressorV2(**tiny_estimator_kwargs()).fit(X, y, groups)
        self.assertTrue(np.isfinite(base.predict(X, groups)).all())

        regressor = LassoNetRegressor(
            hidden_dims=(4,),
            lambda_seq=[0.001],
            n_iters=(6, 2),
            patience=(2, 1),
            verbose=0,
            random_state=123,
            torch_seed=123,
        ).fit(X, y)
        self.assertTrue(np.isfinite(np.asarray(regressor.predict(X))).all())

        quantile = QuantileLassoNetRegressor(
            tau=0.5,
            hidden_dims=(4,),
            lambda_seq=[0.001],
            n_iters=(6, 2),
            patience=(2, 1),
            verbose=0,
            random_state=123,
            torch_seed=123,
        ).fit(X, y)
        self.assertTrue(np.isfinite(np.asarray(quantile.predict(X))).all())

    def test_alpha_endpoints(self):
        X, y, groups = make_small_group_data()
        alpha_zero = TwoStageQuantileLassoNetRegressor(
            tau=0.5,
            alpha=0.0,
            **tiny_quantile_kwargs(alpha_values=None),
        ).fit(X, y, groups)
        self.assertTrue(np.allclose(alpha_zero.predict_offset(X), alpha_zero.predict_common(X)))
        for group in alpha_zero.groups_:
            self.assertFalse(np.any(alpha_zero.group_supports_[group] & ~alpha_zero.stage1_support_))

        alpha_one = TwoStageQuantileLassoNetRegressor(
            tau=0.5,
            alpha=1.0,
            **tiny_quantile_kwargs(alpha_values=None),
        ).fit(X, y, groups)
        self.assertTrue(np.allclose(alpha_one.predict_offset(X), 0.0))

    def test_reproducibility(self):
        X, y, groups = make_small_group_data()
        kwargs = tiny_quantile_kwargs()
        model_a = TwoStageQuantileLassoNetRegressor(tau=0.5, **kwargs).fit(X, y, groups)
        model_b = TwoStageQuantileLassoNetRegressor(tau=0.5, **kwargs).fit(X, y, groups)
        self.assertEqual(model_a.alpha_, model_b.alpha_)
        self.assertEqual(model_a.group_lambdas_, model_b.group_lambdas_)
        self.assertTrue(np.array_equal(model_a.stage1_support_, model_b.stage1_support_))
        self.assertTrue(np.allclose(model_a.predict(X, groups), model_b.predict(X, groups), atol=1e-6))


if __name__ == "__main__":
    unittest.main()

