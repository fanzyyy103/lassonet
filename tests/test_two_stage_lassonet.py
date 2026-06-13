import unittest

import numpy as np
from lassonet import LassoNetRegressor as Stage1LassoNetRegressor

from twostage_lassonet import (
    TwoStagePretrainedLassoNetRegressor,
    extract_stage1_artifacts,
)


class TwoStageLassoNetTests(unittest.TestCase):
    def test_penalty_factor_matches_formula(self):
        model = TwoStagePretrainedLassoNetRegressor(alpha=0.25)
        common_support = np.array([True, False, True, False])
        penalty_factor, zero_mask = model._build_penalty_factor(common_support)

        np.testing.assert_allclose(penalty_factor, np.array([1.0, 4.0, 1.0, 4.0]))
        np.testing.assert_array_equal(zero_mask, np.array([False, False, False, False]))

    def test_alpha_zero_locks_non_common_features(self):
        model = TwoStagePretrainedLassoNetRegressor(alpha=0.0)
        common_support = np.array([True, False, True, False])
        penalty_factor, zero_mask = model._build_penalty_factor(common_support)

        np.testing.assert_allclose(penalty_factor, np.ones(4))
        np.testing.assert_array_equal(zero_mask, np.array([False, True, False, True]))

    def test_end_to_end_fit_predict_and_supports(self):
        rng = np.random.default_rng(1234)
        n_per_group = 40
        p = 6
        X0 = rng.normal(size=(n_per_group, p)).astype(np.float32)
        X1 = rng.normal(size=(n_per_group, p)).astype(np.float32)
        X = np.vstack([X0, X1]).astype(np.float32)
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        noise = 0.05 * rng.normal(size=2 * n_per_group)
        y = (
            2.0 * X[:, 0]
            - 1.5 * X[:, 1]
            + (groups == 0) * 2.2 * X[:, 2]
            + (groups == 1) * 2.2 * X[:, 3]
            + noise
        ).astype(np.float32)

        model = TwoStagePretrainedLassoNetRegressor(
            alpha=0.5,
            stage1_lambda=1e-2,
            stage2_lambda=1e-2,
            common_model_kwargs={
                "hidden_dims": (8,),
                "dense_epochs": 30,
                "sparse_epochs": 20,
                "dense_patience": 10,
                "sparse_patience": 8,
                "random_state": 1234,
                "torch_seed": 1234,
                "verbose": 0,
            },
            group_model_kwargs={
                "hidden_dims": (8,),
                "dense_epochs": 30,
                "sparse_epochs": 20,
                "dense_patience": 10,
                "sparse_patience": 8,
                "random_state": 1234,
                "torch_seed": 1234,
                "verbose": 0,
            },
        )

        model.fit(X, y, groups)
        predictions = model.predict(X, groups)
        components = model.predict_components(X, groups)

        self.assertEqual(predictions.shape, y.shape)
        self.assertEqual(components["final"].shape, y.shape)
        self.assertIn(0, model.group_models_)
        self.assertIn(1, model.group_models_)

        common_support = model.get_common_support()
        group0_final = model.get_group_final_support(0)
        group1_final = model.get_group_final_support(1)

        self.assertTrue(np.all(np.isin(common_support, np.arange(p))))
        self.assertTrue(np.all(np.isin(group0_final, np.arange(p))))
        self.assertTrue(np.all(np.isin(group1_final, np.arange(p))))
        self.assertTrue(set(common_support).issubset(set(group0_final)))
        self.assertTrue(set(common_support).issubset(set(group1_final)))

    def test_can_reuse_external_stage1_model(self):
        rng = np.random.default_rng(1234)
        n_per_group = 30
        p = 5
        X = rng.normal(size=(2 * n_per_group, p)).astype(np.float32)
        groups = np.array([0] * n_per_group + [1] * n_per_group)
        y = (
            1.8 * X[:, 0]
            - 1.2 * X[:, 1]
            + (groups == 0) * 1.5 * X[:, 2]
            + (groups == 1) * 1.5 * X[:, 3]
        ).astype(np.float32)

        stage1_model = Stage1LassoNetRegressor(
            hidden_dims=(6,),
            lambda_seq=[1e-2],
            M=10.0,
            n_iters=(20, 10),
            patience=(5, 4),
            verbose=0,
            random_state=1234,
            torch_seed=1234,
        )
        stage1_model.fit(X, y)

        artifacts = extract_stage1_artifacts(stage1_model, X, alpha=0.5)
        self.assertEqual(artifacts["offset"].shape, y.shape)
        self.assertEqual(artifacts["common_support"].shape, (p,))

        model = TwoStagePretrainedLassoNetRegressor(
            alpha=0.5,
            stage2_lambda=1e-2,
            group_model_kwargs={
                "hidden_dims": (6,),
                "dense_epochs": 20,
                "sparse_epochs": 10,
                "dense_patience": 5,
                "sparse_patience": 4,
                "random_state": 1234,
                "torch_seed": 1234,
                "verbose": 0,
            },
        )
        model.fit(X, y, groups, stage1_model=stage1_model)

        predictions = model.predict(X, groups)
        self.assertEqual(predictions.shape, y.shape)
        self.assertIn(0, model.group_models_)
        self.assertIn(1, model.group_models_)


if __name__ == "__main__":
    unittest.main()
