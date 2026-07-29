"""
Adapter: PDR cascade (proliferative diabetic retinopathy).

The default inference implementation ships with OphAgent. The external
``deploy`` and ``legacy`` backends remain available only for compatibility.

Pipeline:
  1. Main model (ConvNeXt-Base, 384×384) → 4-class category
     {无PDR, 非活动性PDR, 活动性PDR, 无法判断}
  2. EfficientNet-B0 active-reasons multi-label head
  3. EfficientNet-B0 inactive-reasons multi-label head
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...components.pdr import PDRCascadePipeline as BuiltinPDRCascadePipeline
from ...utils.paths import checkpoint_file, external_dir


PDR_SRC = external_dir("OPHAGENT_PDR_SRC", "pdr-label")
_PDR_DEPLOY_OVERRIDE = os.environ.get("OPHAGENT_PDR_DEPLOY", "").strip()
PDR_DEPLOY = (
    Path(os.path.expandvars(os.path.expanduser(_PDR_DEPLOY_OVERRIDE))).resolve()
    if _PDR_DEPLOY_OVERRIDE
    else (PDR_SRC / "deploy").resolve()
)
PDR_MAIN_CKPT = checkpoint_file("OPHAGENT_PDR_MAIN_CKPT", "cfp", "pdr_main.pth")
PDR_ACTIVE_CKPT = checkpoint_file("OPHAGENT_PDR_ACTIVE_CKPT", "cfp", "pdr_active.pth")
PDR_ACTIVE_THR = checkpoint_file("OPHAGENT_PDR_ACTIVE_THR", "cfp", "pdr_active_thr.pth")
PDR_INACTIVE_CKPT = checkpoint_file("OPHAGENT_PDR_INACTIVE_CKPT", "cfp", "pdr_inactive.pth")
PDR_INACTIVE_THR = checkpoint_file("OPHAGENT_PDR_INACTIVE_THR", "cfp", "pdr_inactive_thr.pth")


@register
class PDRCascadeAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="cfp_pdr_cascade",
        modality="CFP",
        task="classification",
        description=(
            "Proliferative diabetic retinopathy (PDR) staging on a colour fundus "
            "photograph. Outputs one of four categories — none / inactive PDR / "
            "active PDR / undetermined — together with the specific signs "
            "(neovascularisation, scarring, laser scars, etc.) when relevant."
        ),
        input_size=(392, 392),
        labels=["no_PDR", "inactive_PDR", "active_PDR", "ungradable"],
        confidence_threshold=0.55,
        limitations=[
            "May misclassify severe NPDR with neovascular-elsewhere features",
            "Trained on Chinese clinic data; CFP from other devices may underperform",
            "**TRAINING DISTRIBUTION CAVEAT** — the negative class was 'no PDR' "
            "drawn predominantly from diabetic populations. The model has NOT "
            "been shown enough non-DR pathology negatives (pathological myopia, "
            "retinal detachment, AMD, RP, large chorioretinal atrophy, cataract). "
            "On such cases it tends to misread atrophy patches / staphyloma "
            "borders as 'inactive PDR with laser scars'. ALWAYS pair this tool "
            "with `cfp_clip_multi_disease`: if CLIP top-1 is a non-DR class, "
            "the PDR label here is likely a false positive — trust CLIP. The "
            "composite tool `cfp_dr_workup` does this cross-check automatically.",
        ],
        cost_class="medium",
        source_dir="ophagent.components.pdr",
    )

    def _load_impl(self) -> None:
        backend = os.environ.get("OPH_PDR_BACKEND", "builtin").strip().lower()
        if backend == "builtin":
            self._load_builtin_impl()
            return
        if backend == "legacy":
            self._load_legacy_impl()
            return
        if backend == "deploy":
            self._load_deploy_impl()
            return
        raise ValueError(
            "OPH_PDR_BACKEND must be one of: builtin, deploy, legacy"
        )

    def _load_builtin_impl(self) -> None:
        paths = (
            PDR_MAIN_CKPT,
            PDR_ACTIVE_CKPT,
            PDR_ACTIVE_THR,
            PDR_INACTIVE_CKPT,
            PDR_INACTIVE_THR,
        )
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "PDR model asset(s) not found:\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        self._impl = BuiltinPDRCascadePipeline(
            main_model_path=PDR_MAIN_CKPT,
            active_model_path=PDR_ACTIVE_CKPT,
            inactive_model_path=PDR_INACTIVE_CKPT,
            active_thresholds_path=PDR_ACTIVE_THR,
            inactive_thresholds_path=PDR_INACTIVE_THR,
            device=self.device,
        )
        self._impl_backend = "builtin"

    def _load_deploy_module(self):
        predict_py = PDR_DEPLOY / "predict.py"
        if not predict_py.exists():
            raise FileNotFoundError(
                "PDR deploy predict.py not found. Set OPHAGENT_PDR_DEPLOY "
                f"or OPHAGENT_PDR_SRC. Expected: {predict_py}"
            )
        spec = importlib.util.spec_from_file_location("ophagent_pdr_deploy_predict", predict_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import PDR deploy module: {predict_py}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _load_deploy_impl(self) -> None:
        mod = self._load_deploy_module()
        import torch

        device_name = self.device
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        device = torch.device(device_name)
        model_dir = PDR_DEPLOY / "models"
        img_size = int(os.environ.get("OPH_PDR_IMG_SIZE", "392"))
        main_models = mod._load_ensemble(
            str(model_dir / "fold_*" / "model_best.pth"),
            len(mod.CATEGORY_NAMES),
            img_size,
            device,
        )
        if not main_models:
            raise FileNotFoundError(
                "No PDR deploy main models found. Set OPHAGENT_PDR_DEPLOY "
                f"or place models under: {model_dir / 'fold_*' / 'model_best.pth'}"
            )
        self._impl = {
            "backend": "deploy",
            "module": mod,
            "device": device,
            "model_dir": model_dir,
            "img_size": img_size,
            "main_models": main_models,
            "main_tf": mod._tf_main(img_size),
            "review_threshold": float(os.environ.get("OPH_PDR_REVIEW_THRESHOLD", "0.6")),
        }

    def _load_legacy_impl(self) -> None:
        self._ensure_path(PDR_SRC)
        # Import the user's modules
        config = self._import_fresh("config", PDR_SRC)
        inference_mod = self._import_fresh("inference", PDR_SRC)

        OUTPUT_DIR = config.OUTPUT_DIR

        def _first_existing(paths: list[Path]) -> Path:
            for p in paths:
                if p.exists():
                    return p
            raise FileNotFoundError(
                "None of these weights exist:\n  " + "\n  ".join(str(p) for p in paths)
            )

        # Prefer the model trained on the latest (post-Dec-2025) annotations.
        # `main_model_phaseNEW/model_final.pth` is the ConvNeXt-Base retrained
        # on all 502 newly-labeled images; it picks up the corrected reasons
        # taxonomy that the active/inactive heads were also retrained on.
        main_path = _first_existing([
            PDR_MAIN_CKPT,
            OUTPUT_DIR / "main_model_phaseNEW" / "model_final.pth",
            OUTPUT_DIR / "main_model_phaseNEW" / "fold_1" / "model_best.pth",
            OUTPUT_DIR / "main_model_phase1" / "fold_1" / "model_best.pth",
            OUTPUT_DIR / "main_model_phase1" / "model_final.pth",
            OUTPUT_DIR / "main_model_phase2" / "model_final.pth",
        ])
        active_path = _first_existing([
            PDR_ACTIVE_CKPT,
            OUTPUT_DIR / "active_reasons_loocv" / "model_final.pth",
            OUTPUT_DIR / "active_reasons" / "model_final.pth",
        ])
        inactive_path = _first_existing([
            PDR_INACTIVE_CKPT,
            OUTPUT_DIR / "inactive_reasons_5fold" / "model_final.pth",
            OUTPUT_DIR / "inactive_reasons" / "model_final.pth",
        ])
        active_thresh = _first_existing([
            PDR_ACTIVE_THR,
            OUTPUT_DIR / "active_reasons_loocv" / "thresholds.pth",
            OUTPUT_DIR / "active_reasons" / "thresholds.pth",
        ])
        inactive_thresh = _first_existing([
            PDR_INACTIVE_THR,
            OUTPUT_DIR / "inactive_reasons_5fold" / "thresholds.pth",
            OUTPUT_DIR / "inactive_reasons" / "thresholds.pth",
        ])

        self._impl = inference_mod.PDRCascadePipeline(
            main_model_path=str(main_path),
            active_model_path=str(active_path),
            inactive_model_path=str(inactive_path),
            active_thresholds_path=str(active_thresh),
            inactive_thresholds_path=str(inactive_thresh),
            device=self.device,
        )
        self._impl_backend = "legacy"

    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        if isinstance(self._impl, dict) and self._impl.get("backend") == "deploy":
            return self._predict_deploy(image_path)
        return self._predict_legacy(image_path)

    def _deploy_reason_head(
        self,
        image_path: str,
        branch: str,
        labels: list[str],
    ) -> list[dict[str, Any]]:
        state = self._impl
        mod = state["module"]
        device = state["device"]
        model_dir = state["model_dir"]
        img_size = state["img_size"]
        rdir = model_dir / f"reasons_{branch}"
        rmodels = mod._load_ensemble(
            str(rdir / "fold_*" / "model_best.pth"),
            len(labels),
            img_size,
            device,
        )
        if not rmodels:
            return []
        try:
            thresholds = mod._load_thresholds(rdir, len(labels))
            probs = mod._ensemble_probs(
                rmodels,
                [image_path],
                mod._tf_reason(img_size),
                device,
                1,
                softmax=False,
            )[0]
            return [
                {
                    "name": labels[i],
                    "score": round(float(probs[i]), 4),
                    "threshold": round(float(thresholds[i]), 4),
                    "predicted": bool(probs[i] >= thresholds[i]),
                }
                for i in range(len(labels))
            ]
        finally:
            del rmodels
            try:
                import torch
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _predict_deploy(self, image_path: str) -> AdapterResult:
        state = self._impl
        mod = state["module"]
        device = state["device"]
        probs_arr = mod._ensemble_probs(
            state["main_models"],
            [image_path],
            state["main_tf"],
            device,
            1,
        )[0]
        cat_id = int(probs_arr.argmax())
        top_conf = float(probs_arr[cat_id])
        cat = mod.CATEGORY_NAMES[cat_id]
        cat_en = mod.CATEGORY_EN[cat_id]
        probs = {
            mod.CATEGORY_NAMES[i]: round(float(probs_arr[i]), 4)
            for i in range(len(mod.CATEGORY_NAMES))
        }
        probs_en = {
            mod.CATEGORY_EN[i]: round(float(probs_arr[i]), 4)
            for i in range(len(mod.CATEGORY_EN))
        }

        active: list[dict[str, Any]] = []
        inactive: list[dict[str, Any]] = []
        # Match the shipped deploy behavior: explain only the selected PDR
        # branch. This avoids surfacing off-branch reason false positives on
        # non-PDR and NPDR images.
        if cat_id == getattr(mod, "ACTIVE_ID", 2):
            active = self._deploy_reason_head(image_path, "active", list(mod.ACTIVE_REASONS))
        elif cat_id == getattr(mod, "INACTIVE_ID", 1):
            inactive = self._deploy_reason_head(image_path, "inactive", list(mod.INACTIVE_REASONS))

        reasons_predicted = [
            r["name"] for r in active + inactive if r.get("predicted")
        ]
        has_active = any(r.get("predicted") for r in active)
        has_inactive = any(r.get("predicted") for r in inactive)
        mixed_pattern = bool(has_active and has_inactive)
        clinical_note = None
        if mixed_pattern:
            clinical_note = (
                "Mixed active + inactive signs detected on the same eye. "
                "Treat this as possible treated PDR with residual or recurrent "
                "active disease; review the union of reasons."
            )

        needs_review = bool(
            top_conf < state["review_threshold"]
            or cat_id == len(mod.CATEGORY_NAMES) - 1
        )
        raw = {
            "backend": "deploy",
            "image": Path(image_path).name,
            "category": cat,
            "category_en": cat_en,
            "category_id": cat_id,
            "category_probs": probs,
            "category_probs_en": probs_en,
            "needs_review": needs_review,
            "reasons": {
                r["name"]: {**r, "head": "active"} for r in active
            } | {
                r["name"]: {**r, "head": "inactive"} for r in inactive
            },
            "has_active_signs": has_active,
            "has_inactive_signs": has_inactive,
            "mixed_pattern": mixed_pattern,
        }

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="CFP",
            task="classification",
            predictions={
                "category": cat,
                "category_en": cat_en,
                "category_id": cat_id,
                "probabilities": probs,
                "probabilities_en": probs_en,
                "needs_review": needs_review,
                "active_reasons": active + inactive,
                "by_head": {"active": active, "inactive": inactive},
                "predicted_reasons": reasons_predicted,
                "has_active_signs": has_active,
                "has_inactive_signs": has_inactive,
                "mixed_pattern": mixed_pattern,
                "clinical_note": clinical_note,
            },
            confidence=top_conf,
            raw_output=raw,
            metadata={
                "source": "PDR_label/deploy/predict.py",
                "backend": "deploy",
                "model_dir": str(state["model_dir"]),
                "review_threshold": state["review_threshold"],
            },
        )

    def _predict_legacy(self, image_path: str) -> AdapterResult:
        raw = self._impl.predict(image_path)
        cat = raw["category"]
        cat_en = raw.get("category_en")
        cat_id = raw["category_id"]
        probs = raw["category_probs"]
        probs_en = raw.get("category_probs_en")
        top_conf = float(max(probs.values())) if probs else 0.0

        reasons_dict = raw.get("reasons", {}) or {}
        reasons_predicted = [n for n, i in reasons_dict.items() if i.get("predicted")]

        # Both reasons heads are now run on every image (see inference.py).
        # Split the output by head so the agent can clearly see
        # `active_reasons` (NVD / NVE / vitreous-hem / fibrovascular) and
        # `inactive_reasons` (laser scars / scarred-FVP) separately.
        active = [
            {"name": n, "score": i["score"], "predicted": i["predicted"]}
            for n, i in reasons_dict.items() if i.get("head") == "active"
        ]
        inactive = [
            {"name": n, "score": i["score"], "predicted": i["predicted"]}
            for n, i in reasons_dict.items() if i.get("head") == "inactive"
        ]

        # Build a clinical-context note for the LLM when the cascade picks
        # one category but the OTHER head also fires — this is the "treated
        # PDR with ongoing NV" pattern the strict taxonomy can't express.
        mixed_pattern = bool(raw.get("mixed_pattern"))
        clinical_note: str | None = None
        if mixed_pattern:
            clinical_note = (
                "Mixed active + inactive signs detected on the same eye "
                "(e.g. laser scars + neovascularisation). The category head "
                f"chose '{cat}', but the other reasons head also reported "
                "positives — likely a TREATED PDR eye with residual or "
                "recurrent active disease. Trust the union of reasons over "
                "the single-class category label."
            )

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="CFP",
            task="classification",
            predictions={
                "category": cat,
                "category_en": cat_en,
                "category_id": cat_id,
                "probabilities": probs,
                "probabilities_en": probs_en,
                # `active_reasons` kept as the legacy key for backward compat
                # with the agent's prompts — it now contains BOTH active-head
                # and inactive-head reasons (the older code only routed one).
                "active_reasons": active + inactive,
                "by_head": {"active": active, "inactive": inactive},
                "predicted_reasons": reasons_predicted,
                "has_active_signs": bool(raw.get("has_active_signs")),
                "has_inactive_signs": bool(raw.get("has_inactive_signs")),
                "mixed_pattern": mixed_pattern,
                "clinical_note": clinical_note,
            },
            confidence=top_conf,
            raw_output=raw,
            metadata={
                "source": (
                    "ophagent.components.pdr"
                    if self._impl_backend == "builtin"
                    else "PDR_label/inference.py"
                ),
                "backend": self._impl_backend,
            },
        )
