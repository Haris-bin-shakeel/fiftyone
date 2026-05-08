"""
Tests for YOLOE visual prompt support in fiftyone/utils/ultralytics.py.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import fiftyone.core.labels as fol


class TestDetectionsToVisualPrompts:
    def test_single_class_normalized_to_absolute(self):
        from fiftyone.utils.ultralytics import _detections_to_visual_prompts

        original_bbox_a = [0.1, 0.2, 0.3, 0.4]
        original_bbox_b = [0.5, 0.0, 0.5, 1.0]
        dets = fol.Detections(
            detections=[
                fol.Detection(label="person", bounding_box=list(original_bbox_a)),
                fol.Detection(label="person", bounding_box=list(original_bbox_b)),
            ]
        )

        boxes, cls_indices, classes = _detections_to_visual_prompts(
            dets, img_width=100, img_height=200
        )

        assert classes == ["person"]
        assert isinstance(classes, list)
        assert all(isinstance(c, str) for c in classes)

        assert cls_indices == [0, 0]
        assert isinstance(cls_indices, list)
        assert all(isinstance(c, int) for c in cls_indices)

        assert isinstance(boxes, list)
        assert len(boxes) == 2
        np.testing.assert_allclose(
            boxes,
            [[10.0, 40.0, 40.0, 120.0], [50.0, 0.0, 100.0, 200.0]],
        )

        assert dets.detections[0].bounding_box == original_bbox_a
        assert dets.detections[1].bounding_box == original_bbox_b

    def test_multiple_classes_dedup_preserves_first_seen_order(self):
        from fiftyone.utils.ultralytics import _detections_to_visual_prompts

        bbox = [0.0, 0.0, 0.1, 0.1]
        dets = fol.Detections(
            detections=[
                fol.Detection(label="dog", bounding_box=list(bbox)),
                fol.Detection(label="cat", bounding_box=list(bbox)),
                fol.Detection(label="dog", bounding_box=list(bbox)),
                fol.Detection(label="bird", bounding_box=list(bbox)),
                fol.Detection(label="cat", bounding_box=list(bbox)),
            ]
        )

        boxes, cls_indices, classes = _detections_to_visual_prompts(
            dets, img_width=10, img_height=10
        )

        assert classes == ["dog", "cat", "bird"]
        assert cls_indices == [0, 1, 0, 2, 1]
        assert len(boxes) == len(dets.detections)


class TestYOLOEVPGetItem:
    def test_required_keys(self):
        from fiftyone.utils.ultralytics import YOLOEVPGetItem

        item = YOLOEVPGetItem(transform=lambda x: {"img": x})

        keys = item.required_keys
        assert isinstance(keys, list)
        assert keys == ["filepath", "prompt_field"]

    def test_call_without_transform_raises_typeerror(self, tmp_path):
        from PIL import Image

        from fiftyone.utils.ultralytics import YOLOEVPGetItem

        path = tmp_path / "x.png"
        Image.new("RGB", (4, 4)).save(path)

        item = YOLOEVPGetItem(transform=None)
        with pytest.raises(TypeError, match="requires a transform"):
            item({"filepath": str(path), "prompt_field": fol.Detections()})

    def test_call_loads_image_force_rgb_and_attaches_prompt(self, tmp_path):
        from PIL import Image

        from fiftyone.utils.ultralytics import YOLOEVPGetItem

        # "L" source verifies the loader forces RGB before the transform.
        path = tmp_path / "gray.png"
        Image.new("L", (4, 4), color=128).save(path)

        prompt = fol.Detections(
            detections=[fol.Detection(label="x", bounding_box=[0, 0, 1, 1])]
        )

        captured: dict[str, Any] = {}

        def fake_transform(img):
            captured["mode"] = img.mode
            captured["size"] = img.size
            return {"img": "TENSOR", "orig_img": "ARRAY"}

        item = YOLOEVPGetItem(transform=fake_transform)
        result = item({"filepath": str(path), "prompt_field": prompt})

        assert captured == {"mode": "RGB", "size": (4, 4)}
        assert set(result.keys()) == {"img", "orig_img", "prompt"}
        assert result["img"] == "TENSOR"
        assert result["orig_img"] == "ARRAY"
        assert result["prompt"] is prompt


class TestFiftyOneYOLOEVPCollate:
    def test_collate_pops_prompt_and_stacks_images(self):
        import torch

        from fiftyone.utils.ultralytics import FiftyOneYOLOEVPModel

        prompt_a = fol.Detections()
        prompt_b = None

        img_a_t = torch.full((3, 5, 7), 1.0)
        img_b_t = torch.full((3, 5, 7), 2.0)
        img_a_o = np.zeros((5, 7, 3), dtype=np.uint8)
        img_b_o = np.zeros((5, 7, 3), dtype=np.uint8)

        batch = [
            {"img": img_a_t, "orig_img": img_a_o, "prompt": prompt_a},
            {"img": img_b_t, "orig_img": img_b_o, "prompt": prompt_b},
        ]

        out = FiftyOneYOLOEVPModel.collate_fn(batch)

        assert set(out.keys()) == {
            "orig_imgs",
            "images",
            "orig_shapes",
            "prompts",
        }
        assert out["prompts"] == [prompt_a, prompt_b]
        assert out["orig_imgs"][0] is img_a_o
        assert out["orig_imgs"][1] is img_b_o
        assert out["orig_shapes"] == [(7, 5), (7, 5)]

        assert isinstance(out["images"], torch.Tensor)
        assert out["images"].shape == (2, 3, 5, 7)
        assert torch.equal(out["images"][0], img_a_t)
        assert torch.equal(out["images"][1], img_b_t)

        assert all("prompt" not in item for item in batch)

    def test_collate_uses_none_for_missing_prompt_key(self):
        import torch

        from fiftyone.utils.ultralytics import FiftyOneYOLOEVPModel

        batch = [
            {
                "img": torch.zeros(3, 4, 4),
                "orig_img": np.zeros((4, 4, 3), dtype=np.uint8),
            }
        ]

        out = FiftyOneYOLOEVPModel.collate_fn(batch)

        assert out["prompts"] == [None]
        assert out["images"].shape == (1, 3, 4, 4)


class TestFiftyOneYOLOEVPDispatch:
    @staticmethod
    def _bare_model():
        from fiftyone.utils.ultralytics import FiftyOneYOLOEVPModel

        return FiftyOneYOLOEVPModel.__new__(FiftyOneYOLOEVPModel)

    def test_dispatch_with_prompts_calls_visual_prompts(self, monkeypatch):
        from fiftyone.utils import ultralytics as fu

        model = self._bare_model()

        captured: dict[str, Any] = {}

        def fake_vp(orig_imgs, width_height, prompts):
            captured["orig_imgs"] = orig_imgs
            captured["width_height"] = width_height
            captured["prompts"] = prompts
            return ["VP_RESULT"]

        model._predict_all_visual_prompts = fake_vp

        monkeypatch.setattr(
            fu.FiftyOneYOLOModel,
            "_predict_all",
            lambda self, imgs: pytest.fail(
                "super()._predict_all called when prompts present"
            ),
        )

        prompt = fol.Detections(
            detections=[fol.Detection(label="x", bounding_box=[0, 0, 1, 1])]
        )

        batch = {
            "orig_imgs": ["A"],
            "images": "stacked-tensor",
            "orig_shapes": [(7, 5)],
            "prompts": [prompt],
        }

        out = model._predict_all(batch)

        assert out == ["VP_RESULT"]
        assert captured == {
            "orig_imgs": ["A"],
            "width_height": [(7, 5)],
            "prompts": [prompt],
        }

    def test_dispatch_with_mixed_none_and_non_none_prompts_uses_vp_path(self):
        model = self._bare_model()

        prompts_seen = []

        def fake_vp(orig_imgs, width_height, prompts):
            prompts_seen.extend(prompts)
            return ["MIXED"]

        model._predict_all_visual_prompts = fake_vp

        prompt = fol.Detections()
        batch = {
            "orig_imgs": ["A", "B", "C"],
            "images": "T",
            "orig_shapes": [(1, 1), (1, 1), (1, 1)],
            "prompts": [None, prompt, None],
        }

        out = model._predict_all(batch)

        assert out == ["MIXED"]
        assert prompts_seen == [None, prompt, None]

    def test_dispatch_with_all_none_prompts_falls_through_to_super(
        self, monkeypatch
    ):
        from fiftyone.utils import ultralytics as fu

        model = self._bare_model()

        super_calls = []

        def fake_super_predict_all(self, imgs):
            super_calls.append(imgs)
            return ["SUPER"]

        monkeypatch.setattr(
            fu.FiftyOneYOLOModel,
            "_predict_all",
            fake_super_predict_all,
        )

        model._predict_all_visual_prompts = lambda *a, **kw: pytest.fail(
            "_predict_all_visual_prompts called for all-None prompts"
        )

        batch = {
            "orig_imgs": ["A"],
            "images": "T",
            "orig_shapes": [(1, 1)],
            "prompts": [None],
        }

        out = model._predict_all(batch)

        assert out == ["SUPER"]
        assert super_calls == [batch]

    def test_dispatch_with_empty_prompts_falls_through_to_super(
        self, monkeypatch
    ):
        from fiftyone.utils import ultralytics as fu

        model = self._bare_model()

        super_calls = []
        monkeypatch.setattr(
            fu.FiftyOneYOLOModel,
            "_predict_all",
            lambda self, imgs: super_calls.append(imgs) or ["SUPER"],
        )

        model._predict_all_visual_prompts = lambda *a, **kw: pytest.fail(
            "_predict_all_visual_prompts called for empty prompts list"
        )

        batch = {
            "orig_imgs": [],
            "images": "T",
            "orig_shapes": [],
            "prompts": [],
        }

        assert model._predict_all(batch) == ["SUPER"]
        assert super_calls == [batch]

    def test_dispatch_with_non_dict_falls_through_to_super(self, monkeypatch):
        from fiftyone.utils import ultralytics as fu

        model = self._bare_model()

        super_calls = []
        monkeypatch.setattr(
            fu.FiftyOneYOLOModel,
            "_predict_all",
            lambda self, imgs: super_calls.append(imgs) or ["SUPER"],
        )

        model._predict_all_visual_prompts = lambda *a, **kw: pytest.fail(
            "_predict_all_visual_prompts called for non-dict input"
        )

        out = model._predict_all(["raw", "list"])

        assert out == ["SUPER"]
        assert super_calls == [["raw", "list"]]


class TestFiftyOneYOLOEVPVisualPrompts:
    @staticmethod
    def _make_model(
        monkeypatch,
        *,
        predict,
        vp_predictor_cls=object,
        confidence_thresh=None,
        filter_classes=None,
        rect=False,
        retina_masks=True,
        predictor_default_conf=0.25,
    ):
        from fiftyone.utils import ultralytics as fu

        monkeypatch.setattr(
            fu, "_get_yoloe_vp_predictor", lambda: vp_predictor_cls
        )

        model = fu.FiftyOneYOLOEVPModel.__new__(fu.FiftyOneYOLOEVPModel)
        model.config = SimpleNamespace(
            confidence_thresh=confidence_thresh,
            filter_classes=filter_classes,
        )
        model._device = "cpu"
        model._model = SimpleNamespace(
            predict=predict,
            predictor=SimpleNamespace(
                args=SimpleNamespace(
                    rect=rect,
                    retina_masks=retina_masks,
                    conf=predictor_default_conf,
                )
            ),
        )
        return model

    def test_visual_prompts_built_per_image_with_full_predict_kwargs(
        self, monkeypatch
    ):
        from fiftyone.utils import ultralytics as fu

        captured_calls = []

        class _Result:
            def __init__(self):
                self.names = None

        result_a = _Result()
        result_b = _Result()

        def fake_predict(img, **kwargs):
            captured_calls.append({"img": img, **kwargs})
            return [result_a if img == "img-A" else result_b]

        to_instances_calls = []

        def fake_to_instances(results, confidence_thresh, classes):
            to_instances_calls.append(
                {
                    "results": results,
                    "confidence_thresh": confidence_thresh,
                    "classes": classes,
                }
            )
            return fol.Detections(
                detections=[
                    fol.Detection(
                        label=f"out-{r.names[0]}",
                        bounding_box=[0, 0, 1, 1],
                    )
                    for r in results
                ]
            )

        monkeypatch.setattr(fu, "to_instances", fake_to_instances)

        sentinel_predictor = object
        model = self._make_model(
            monkeypatch,
            predict=fake_predict,
            vp_predictor_cls=sentinel_predictor,
            rect=True,
            retina_masks=False,
            predictor_default_conf=0.31,
        )

        prompt_a = fol.Detections(
            detections=[
                fol.Detection(label="dog", bounding_box=[0.0, 0.0, 0.5, 0.5]),
                fol.Detection(label="cat", bounding_box=[0.5, 0.5, 0.5, 0.5]),
            ]
        )
        prompt_b = fol.Detections(
            detections=[
                fol.Detection(label="bird", bounding_box=[0.0, 0.0, 1.0, 1.0])
            ]
        )

        labels = model._predict_all_visual_prompts(
            orig_images=["img-A", "img-B"],
            width_height=[(20, 10), (4, 4)],
            prompts=[prompt_a, prompt_b],
        )

        assert len(captured_calls) == 2

        for call, expected_img in zip(
            captured_calls, ["img-A", "img-B"], strict=True
        ):
            assert call["img"] == expected_img
            assert call["predictor"] is sentinel_predictor
            assert call["mode"] == "predict"
            assert call["save"] is False
            assert call["verbose"] is False
            assert call["device"] == "cpu"
            assert call["rect"] is True
            assert call["retina_masks"] is False
            assert call["conf"] == 0.31

        np.testing.assert_array_equal(
            captured_calls[0]["visual_prompts"]["bboxes"],
            np.array([[0.0, 0.0, 10.0, 5.0], [10.0, 5.0, 20.0, 10.0]]),
        )
        np.testing.assert_array_equal(
            captured_calls[0]["visual_prompts"]["cls"], np.array([0, 1])
        )
        np.testing.assert_array_equal(
            captured_calls[1]["visual_prompts"]["bboxes"],
            np.array([[0.0, 0.0, 4.0, 4.0]]),
        )
        np.testing.assert_array_equal(
            captured_calls[1]["visual_prompts"]["cls"], np.array([0])
        )

        assert result_a.names == {0: "dog", 1: "cat"}
        assert result_b.names == {0: "bird"}

        assert len(to_instances_calls) == 2
        for call in to_instances_calls:
            assert call["confidence_thresh"] is None
            assert call["classes"] is None

        assert len(labels) == 2
        assert isinstance(labels[0], fol.Detections)
        assert labels[0].detections[0].label == "out-dog"
        assert labels[1].detections[0].label == "out-bird"

    def test_visual_prompts_forwards_filter_classes_to_to_instances(
        self, monkeypatch
    ):
        from fiftyone.utils import ultralytics as fu

        captured_to_instances = []

        class _Result:
            names = None

        def fake_predict(img, **kwargs):
            return [_Result()]

        def fake_to_instances(results, confidence_thresh, classes):
            captured_to_instances.append(
                {
                    "confidence_thresh": confidence_thresh,
                    "classes": classes,
                }
            )
            return fol.Detections()

        monkeypatch.setattr(fu, "to_instances", fake_to_instances)

        model = self._make_model(
            monkeypatch,
            predict=fake_predict,
            confidence_thresh=0.42,
            filter_classes=["dog", "cat"],
        )

        prompt = fol.Detections(
            detections=[
                fol.Detection(label="dog", bounding_box=[0, 0, 0.5, 0.5])
            ]
        )

        model._predict_all_visual_prompts(["A"], [(4, 4)], [prompt])

        assert captured_to_instances == [
            {"confidence_thresh": 0.42, "classes": ["dog", "cat"]}
        ]

    def test_predictor_restored_on_success(self, monkeypatch):
        from fiftyone.utils import ultralytics as fu

        monkeypatch.setattr(
            fu, "to_instances", lambda *a, **kw: fol.Detections()
        )

        class _Result:
            names = None

        # Simulate ultralytics installing the VP predictor mid-call.
        def fake_predict(img, **kwargs):
            model_ref._model.predictor = SimpleNamespace(name="VP_PRED")
            return [_Result()]

        model = self._make_model(monkeypatch, predict=fake_predict)
        model_ref = model
        original_predictor = model._model.predictor

        prompt = fol.Detections(
            detections=[fol.Detection(label="x", bounding_box=[0, 0, 1, 1])]
        )

        model._predict_all_visual_prompts(
            ["A", "B"], [(4, 4), (4, 4)], [prompt, prompt]
        )

        assert model._model.predictor is original_predictor

    def test_predictor_restored_on_exception(self, monkeypatch):
        from fiftyone.utils import ultralytics as fu

        monkeypatch.setattr(
            fu, "to_instances", lambda *a, **kw: fol.Detections()
        )

        def fake_predict(img, **kwargs):
            model_ref._model.predictor = SimpleNamespace(name="VP_PRED")
            raise RuntimeError("ultralytics blew up")

        model = self._make_model(monkeypatch, predict=fake_predict)
        model_ref = model
        original_predictor = model._model.predictor

        prompt = fol.Detections(
            detections=[fol.Detection(label="x", bounding_box=[0, 0, 1, 1])]
        )

        with pytest.raises(RuntimeError, match="ultralytics blew up"):
            model._predict_all_visual_prompts(["A"], [(4, 4)], [prompt])

        assert model._model.predictor is original_predictor

    def test_empty_or_missing_prompts_yield_empty_detections(self, monkeypatch):
        from fiftyone.utils import ultralytics as fu

        to_instances_calls = []

        def fake_to_instances(results, confidence_thresh, classes):
            to_instances_calls.append(results)
            return fol.Detections(
                detections=[
                    fol.Detection(
                        label="should-not-appear",
                        bounding_box=[0, 0, 1, 1],
                    )
                ]
            )

        monkeypatch.setattr(fu, "to_instances", fake_to_instances)

        predict_calls = []

        class _Result:
            names = None

        def fake_predict(img, **kwargs):
            predict_calls.append(img)
            return [_Result()]

        model = self._make_model(monkeypatch, predict=fake_predict)

        prompt_real = fol.Detections(
            detections=[
                fol.Detection(label="dog", bounding_box=[0, 0, 0.5, 0.5])
            ]
        )

        labels = model._predict_all_visual_prompts(
            orig_images=["img-empty", "img-none", "img-real"],
            width_height=[(4, 4), (4, 4), (4, 4)],
            prompts=[fol.Detections(), None, prompt_real],
        )

        assert predict_calls == ["img-real"]
        assert len(to_instances_calls) == 1

        assert len(labels) == 3
        assert isinstance(labels[0], fol.Detections)
        assert labels[0].detections == []
        assert isinstance(labels[1], fol.Detections)
        assert labels[1].detections == []
        assert labels[2].detections[0].label == "should-not-appear"

    def test_confidence_thresh_overrides_predictor_default(self, monkeypatch):
        from fiftyone.utils import ultralytics as fu

        monkeypatch.setattr(
            fu, "to_instances", lambda *a, **kw: fol.Detections()
        )

        captured = {}

        class _Result:
            names = None

        def fake_predict(img, **kwargs):
            captured.update(kwargs)
            return [_Result()]

        model = self._make_model(
            monkeypatch,
            predict=fake_predict,
            confidence_thresh=0.42,
            predictor_default_conf=0.25,
        )

        prompt = fol.Detections(
            detections=[fol.Detection(label="x", bounding_box=[0, 0, 1, 1])]
        )

        model._predict_all_visual_prompts(["A"], [(4, 4)], [prompt])

        assert captured["conf"] == 0.42

    def test_confidence_thresh_zero_overrides_predictor_default(
        self, monkeypatch
    ):
        # Falsy guard: an explicit 0.0 must not collapse to the predictor
        # default.
        from fiftyone.utils import ultralytics as fu

        monkeypatch.setattr(
            fu, "to_instances", lambda *a, **kw: fol.Detections()
        )

        captured = {}

        class _Result:
            names = None

        def fake_predict(img, **kwargs):
            captured.update(kwargs)
            return [_Result()]

        model = self._make_model(
            monkeypatch,
            predict=fake_predict,
            confidence_thresh=0.0,
            predictor_default_conf=0.25,
        )

        prompt = fol.Detections(
            detections=[fol.Detection(label="x", bounding_box=[0, 0, 1, 1])]
        )

        model._predict_all_visual_prompts(["A"], [(4, 4)], [prompt])

        assert captured["conf"] == 0.0

    def test_confidence_thresh_none_falls_back_to_predictor_default(
        self, monkeypatch
    ):
        from fiftyone.utils import ultralytics as fu

        monkeypatch.setattr(
            fu, "to_instances", lambda *a, **kw: fol.Detections()
        )

        captured = {}

        class _Result:
            names = None

        def fake_predict(img, **kwargs):
            captured.update(kwargs)
            return [_Result()]

        model = self._make_model(
            monkeypatch,
            predict=fake_predict,
            confidence_thresh=None,
            predictor_default_conf=0.31,
        )

        prompt = fol.Detections(
            detections=[fol.Detection(label="x", bounding_box=[0, 0, 1, 1])]
        )

        model._predict_all_visual_prompts(["A"], [(4, 4)], [prompt])

        assert captured["conf"] == 0.31

    def test_multiple_results_per_predict_call_share_names_map(
        self, monkeypatch
    ):
        # Every Result returned by one predict() call receives the same
        # names_map.
        from fiftyone.utils import ultralytics as fu

        class _Result:
            def __init__(self):
                self.names = None

        result_first = _Result()
        result_second = _Result()

        def fake_predict(img, **kwargs):
            return [result_first, result_second]

        monkeypatch.setattr(
            fu, "to_instances", lambda *a, **kw: fol.Detections()
        )

        model = self._make_model(monkeypatch, predict=fake_predict)

        prompt = fol.Detections(
            detections=[
                fol.Detection(label="dog", bounding_box=[0.0, 0.0, 0.5, 0.5]),
                fol.Detection(label="cat", bounding_box=[0.5, 0.5, 0.5, 0.5]),
            ]
        )

        model._predict_all_visual_prompts(["A"], [(4, 4)], [prompt])

        expected = {0: "dog", 1: "cat"}
        assert result_first.names == expected
        assert result_second.names == expected

    def test_zip_strict_raises_on_length_mismatch(self, monkeypatch):
        # strict=True trips on the next advance, so matched tuples run before
        # the error surfaces. The finally block must still restore.
        from fiftyone.utils import ultralytics as fu

        monkeypatch.setattr(
            fu, "to_instances", lambda *a, **kw: fol.Detections()
        )

        class _Result:
            names = None

        predict_calls = []

        def fake_predict(img, **kwargs):
            predict_calls.append(img)
            return [_Result()]

        model = self._make_model(monkeypatch, predict=fake_predict)

        prompt = fol.Detections(
            detections=[fol.Detection(label="x", bounding_box=[0, 0, 1, 1])]
        )
        original_predictor = model._model.predictor

        with pytest.raises(ValueError):
            model._predict_all_visual_prompts(
                orig_images=["A", "B", "C"],
                width_height=[(4, 4), (4, 4)],
                prompts=[prompt, prompt],
            )

        assert predict_calls == ["A", "B"]
        assert model._model.predictor is original_predictor


class TestGetYOLOEVPPredictor:
    def test_returns_predictor_class_when_available(self):
        from fiftyone.utils.ultralytics import _get_yoloe_vp_predictor

        cls = _get_yoloe_vp_predictor()

        assert isinstance(cls, type)
        assert cls.__name__ == "YOLOEVPSegPredictor"

    def test_raises_import_error_with_attribute_error_cause(self, monkeypatch):
        from fiftyone.utils import ultralytics as fu

        class _BrokenYoloe:
            def __getattr__(self, name):
                raise AttributeError(name)

        monkeypatch.setattr(fu, "_yoloe", _BrokenYoloe())

        with pytest.raises(ImportError, match="ultralytics>=8.4.0") as exc:
            fu._get_yoloe_vp_predictor()

        # `from e` chains the underlying AttributeError as __cause__.
        assert isinstance(exc.value.__cause__, AttributeError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
