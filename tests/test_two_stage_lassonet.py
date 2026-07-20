import unittest

import numpy as np
from lassonet import LassoNetRegressor as Stage1LassoNetRegressor

from twostage_lassonet import (
    PTLassoOrientedTwoStageLassoNetRegressor,
    PTLassoOrientedTwoStageLassoNetRegressorCV,
    TwoStagePretrainedLassoNetRegressor,
    LassoNetRegressor,
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

    def test_ptlasso_oriented_summary_is_self_consistent(self):
        rng = np.random.default_rng(4321)
        n_per_group = 36
        p = 6
        X0 = rng.normal(size=(n_per_group, p)).astype(np.float32)
        X1 = rng.normal(size=(n_per_group, p)).astype(np.float32)
        X = np.vstack([X0, X1]).astype(np.float32)
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        y = (
            1.7 * X[:, 0]
            - 1.1 * X[:, 1]
            + (groups == 0) * 1.9 * X[:, 2]
            + (groups == 1) * 1.9 * X[:, 3]
            + 0.05 * rng.normal(size=2 * n_per_group)
        ).astype(np.float32)

        model = PTLassoOrientedTwoStageLassoNetRegressor(
            alpha=0.5,
            stage1_lambda=1e-2,
            stage2_lambda=1e-2,
            common_model_kwargs={
                "hidden_dims": (8,),
                "dense_epochs": 25,
                "sparse_epochs": 15,
                "dense_patience": 8,
                "sparse_patience": 6,
                "random_state": 4321,
                "torch_seed": 4321,
                "verbose": 0,
            },
            group_model_kwargs={
                "hidden_dims": (8,),
                "dense_epochs": 25,
                "sparse_epochs": 15,
                "dense_patience": 8,
                "sparse_patience": 6,
                "random_state": 4321,
                "torch_seed": 4321,
                "verbose": 0,
            },
        )

        feature_names = [f"x{i}" for i in range(p)]
        model.fit(X, y, groups, feature_names=feature_names)
        components = model.predict_components(X, groups)

        self.assertIn("group_correction", components)
        self.assertEqual(components["group_correction"].shape, y.shape)

        common = set(model.get_common_support().tolist())
        summary = model.summarize_group_correction(0)
        correction = set(model.get_group_correction_support(0).tolist())
        component = set(model.get_group_specific_component_support(0).tolist())
        reused = set(model.get_group_common_reused_support(0).tolist())
        strict = set(model.get_group_strict_specific_support(0).tolist())
        final_union = set(model.get_group_final_support(0).tolist())

        self.assertEqual(component, correction)
        self.assertEqual(reused, correction & common)
        self.assertEqual(strict, correction - common)
        self.assertEqual(final_union, correction | common)
        self.assertEqual(summary["common_feature_count"], len(common))
        self.assertEqual(summary["group_correction_feature_count"], len(correction))
        self.assertEqual(summary["group_specific_component_feature_count"], len(correction))
        self.assertEqual(summary["strict_group_only_count"], len(strict))
        self.assertEqual(summary["common_reused_count"], len(reused))
        self.assertEqual(summary["final_union_count"], len(final_union))

        structure = model.describe_ptlasso_structure()
        self.assertEqual(structure["common_feature_count"], len(common))
        self.assertIn(0, structure["groups"])
        self.assertIn("pretrained_overlap", structure)

        component_summary = model.summarize_group_specific_component(0)
        self.assertEqual(
            component_summary["group_specific_component_feature_count"],
            len(correction),
        )

        overlap = model.summarize_pretrained_overlap()
        self.assertIn("pretrained_common_count", overlap)
        self.assertIn(0, overlap["groups"])
        self.assertIn("pretrained_individual_count", overlap["groups"][0])

        pretrained_common = set(model.get_pretrained_common_support().tolist())
        pretrained_final_group0 = set(model.get_group_pretrained_final_support(0).tolist())
        pretrained_individual_group0 = set(model.get_group_pretrained_individual_support(0).tolist())
        self.assertEqual(pretrained_individual_group0, pretrained_final_group0 - pretrained_common)

        importance_rows = model.get_group_correction_importance(0)
        self.assertEqual(len(importance_rows), p)
        self.assertGreaterEqual(
            importance_rows[0]["skip_importance"],
            importance_rows[-1]["skip_importance"],
        )

        role_rows = model.summarize_feature_roles()
        self.assertTrue(any(row["role"] == "common_only" for row in role_rows) or any(
            row["role"] == "common_and_group_corrected" for row in role_rows
        ))

    def test_skip_only_stage2_keeps_hidden_layers_zero(self):
        rng = np.random.default_rng(2468)
        X = rng.normal(size=(80, 5)).astype(np.float32)
        y = (1.5 * X[:, 0] - 2.0 * X[:, 2] + 0.1 * rng.normal(size=80)).astype(np.float32)

        model = LassoNetRegressor(
            hidden_dims=(6,),
            skip_only=True,
            dense_epochs=15,
            sparse_epochs=10,
            dense_patience=5,
            sparse_patience=4,
            random_state=2468,
            torch_seed=2468,
            verbose=0,
        )
        model.fit(X, y, lambda_=10.0)

        for layer in model.model.layers:
            self.assertTrue(np.allclose(layer.weight.detach().cpu().numpy(), 0.0))
            self.assertTrue(np.allclose(layer.bias.detach().cpu().numpy(), 0.0))

    def test_official_cv_two_stage_selects_alpha_and_both_stage_lambdas(self):
        rng = np.random.default_rng(9753)
        n_per_group = 24
        p = 5
        X = rng.normal(size=(2 * n_per_group, p)).astype(np.float32)
        groups = np.array([0] * n_per_group + [1] * n_per_group)
        y = (
            1.6 * X[:, 0]
            - 1.1 * X[:, 1]
            + (groups == 0) * 1.8 * X[:, 2]
            + (groups == 1) * 1.8 * X[:, 3]
            + 0.05 * rng.normal(size=2 * n_per_group)
        ).astype(np.float32)

        cv_kwargs = {
            "hidden_dims": (4,),
            "lambda_seq": [0.01, 0.1, 1.0],
            "path_multiplier": 2.0,
            "n_iters": (6, 4),
            "patience": (3, 2),
            "val_size": 0,
            "verbose": 0,
            "random_state": 9753,
            "torch_seed": 9753,
        }
        model = PTLassoOrientedTwoStageLassoNetRegressorCV(
            alpha_grid=(0.5,),
            alpha_cv=2,
            stage1_cv=2,
            stage2_cv=2,
            common_model_kwargs=cv_kwargs,
            group_model_kwargs=cv_kwargs,
            random_state=9753,
            verbose=0,
        )
        model.fit(X, y, groups, feature_names=[f"x{i}" for i in range(p)])

        predictions = model.predict(X, groups)
        self.assertEqual(predictions.shape, y.shape)
        self.assertEqual(model.best_alpha_, 0.5)
        self.assertIsInstance(model.best_stage1_lambda_, float)
        self.assertEqual(set(model.best_stage2_lambda_by_group_), {0, 1})
        self.assertEqual(len(model.alpha_cv_results_), 1)
        self.assertEqual(len(model.alpha_cv_details_), 2)
        alpha_result = model.alpha_cv_results_[0]
        detail_mses = [
            row["validation_mse"] for row in model.alpha_cv_details_
        ]
        self.assertIn("mean_fold_mse", alpha_result)
        self.assertNotIn("pooled_oof_mse", alpha_result)
        self.assertAlmostEqual(alpha_result["mean_fold_mse"], np.mean(detail_mses))
        self.assertEqual(alpha_result["n_folds"], 2)
        self.assertEqual(model.get_feature_names().tolist(), [f"x{i}" for i in range(p)])

    def test_official_cv_alpha_zero_uses_only_stage1_support(self):
        rng = np.random.default_rng(8642)
        X = rng.normal(size=(32, 4)).astype(np.float32)
        y = (1.5 * X[:, 0] - 0.8 * X[:, 2]).astype(np.float32)
        common_support = np.array([True, False, True, False])

        cv_kwargs = {
            "hidden_dims": (4,),
            "lambda_seq": [0.01, 0.1],
            "n_iters": (5, 3),
            "patience": (2, 2),
            "val_size": 0,
            "verbose": 0,
            "random_state": 8642,
            "torch_seed": 8642,
        }
        model = PTLassoOrientedTwoStageLassoNetRegressorCV(
            alpha_grid=(0.0,),
            alpha_cv=2,
            stage1_cv=2,
            stage2_cv=2,
            group_model_kwargs=cv_kwargs,
            random_state=8642,
            verbose=0,
        )

        stage2_model = model._fit_stage2_group(
            X,
            y,
            np.zeros_like(y),
            common_support,
            alpha=0.0,
        )

        np.testing.assert_array_equal(stage2_model.feature_indices, np.array([0, 2]))
        self.assertFalse(stage2_model.selected_mask()[~common_support].any())


if __name__ == "__main__":
    unittest.main()
