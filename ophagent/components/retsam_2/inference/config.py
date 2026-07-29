"""
Configuration for the mask-generating inference pipeline.

Trimmed from the original `inference_modules/config.py`:
- dropped `supported_diseases`, `disease_names`, `disease_grade_descriptions`,
  `classification_thresholds`, and their accessor methods. Disease classification
  is explicitly NOT part of this project.
"""

from __future__ import annotations

from typing import Dict, List


class InferenceConfig:
    """Central configuration for mask-only inference."""

    def __init__(self):
        # ---- Task configuration ----------------------------------------------------
        # Task ordering must match the head ordering in the checkpoint.
        self.task_names = [
            'artery_vein', 'od_oc', 'tessellation', 'myopia',
            'lesion_s1', 'lesion_s2', 'lesion_s3', 'possible_lesions',
        ]

        # Which tasks are lesion-type (multi-class binary per category).
        self.lesion_task_names = ['lesion_s1', 'lesion_s2', 'lesion_s3', 'possible_lesions']

        # Grouped lesion buckets that the visualizer consolidates into.
        self.grouped_lesion_names = ['lesion_dr', 'lesion_amd', 'lesion_others', 'possible_lesions']

        # Per-lesion-task class index -> human-readable name (saved mask filenames).
        self.lesion_type_mappings = {
            'lesion_s1': {
                1: 'hemorrhage',
                2: 'exudate',
                3: 'cotton_wool_spot',
                4: 'drusen',
                5: 'laser_spot',
            },
            'lesion_s2': {
                1: 'epiretinal_membrane',
                2: 'macular_hole',
                3: 'lesion_3',
                4: 'lesion_4',
                5: 'artifact',
                6: 'lesion_6',
                7: 'lesion_7',
                8: 'lesion_8',
                9: 'myelinated_nerve_fiber',
                10: 'lesion_10',
            },
            'lesion_s3': {
                1: 'venous_tortuosity',
                2: 'CNV',
                3: 'laser_spot',
                4: 'retinal_scar',
                5: 'patch_hemorrhage',
            },
            'possible_lesions': {
                1: 'other_lesions',
            },
        }

        # Non-lesion task class mappings.
        self.task_class_mappings = {
            'artery_vein': {1: 'artery', 2: 'vein'},
            'od_oc': {1: 'optic_disc', 2: 'optic_cup'},
            'tessellation': {1: 'tessellation_region'},
            'myopia': {
                1: 'arc_lesion',
                2: 'diffuse_chorioretinal_atrophy',
                3: 'patchy_chorioretinal_atrophy',
            },
        }

        # Grouped lesion buckets for final saved outputs.
        self.grouped_lesion_type_mappings = {
            'lesion_dr': {
                1: 'hemorrhage',
                2: 'exudate',
                3: 'cotton_wool_spot',
                4: 'laser_spot',
            },
            'lesion_amd': {
                1: 'drusen',
                2: 'patch_hemorrhage',
            },
            'lesion_others': {
                1: 'epiretinal_membrane',
                2: 'artifact',
                3: 'retinal_scar',
                4: 'macular_hole',
            },
            'possible_lesions': {
                1: 'other_lesions',
                2: 'myelinated_nerve_fiber',
                3: 'venous_tortuosity',
            },
        }

        # How each task's logits should be interpreted.
        self.task_output_types = {
            'artery_vein': 'multilabel',
            'od_oc': 'multiclass',
            'lesion_s1': 'multiclass',
            'tessellation': 'multiclass',
            'myopia': 'multiclass',
            'lesion_s2': 'multiclass',
            'lesion_s3': 'multiclass',
            'possible_lesions': 'multiclass',
            'lesion_dr': 'multiclass',
            'lesion_amd': 'multiclass',
            'lesion_others': 'multiclass',
        }

        # ---- Image processing ------------------------------------------------------
        self.image_size = 640
        self.analysis_size = 640
        self.normalization_mean = [0.42575, 0.29737, 0.21294]
        self.normalization_std = [0.2767, 0.2024, 0.1686]

        # ---- Model defaults --------------------------------------------------------
        # Default channels for the deployed retsam.ckpt. A coordinate head (2 scalars)
        # may sit after the segmentation heads when --has_coordinate_head is set.
        self.default_output_channels = (3, 3, 2, 4, 6, 11, 6, 2)
        self.model_params = {
            'in_channels': 3,
            'feature_size': 128,
            'depths': (2, 2, 18, 2),
            'num_heads': (4, 8, 16, 32),
            'window_size': 10,
            'patch_size': 4,
            'norm_name': 'instance',
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'dropout_path_rate': 0.0,
            'normalize': True,
            'use_checkpoint': False,
            'img_size': 640,
            'spatial_dims': 2,
            'downsample': 'merging',
            'use_v2': False,
        }

        # ---- Visualization (optional overlay rendering) ----------------------------
        self.visualization_colors = {
            'artery': [255, 0, 0],
            'vein': [0, 0, 255],
            'disc': [0, 255, 0],
            'cup': [255, 255, 0],
            'lesion': [255, 0, 255],
            'tessellation': [0, 255, 255],
            'myopia': [255, 128, 0],
        }

        self.overlay_alpha_default: float = 0.6
        self.overlay_alpha_per_task = {
            'artery_vein': 0.6,
            'od_oc': 0.55,
            'lesion_s1': 0.6,
            'tessellation': 0.6,
            'myopia': 0.55,
            'lesion_s2': 0.6,
            'lesion_s3': 0.6,
            'possible_lesions': 0.55,
            'lesion_dr': 0.6,
            'lesion_amd': 0.6,
            'lesion_others': 0.6,
        }

        # ---- Probability gating (used by NoiseFilter in postprocessing.py) --------
        self.prob_threshold_default: float = 0.7
        self.prob_threshold_per_task = {
            'lesion_s1': 0.7,
            'lesion_s2': 0.7,
            'lesion_s3': 0.7,
            'possible_lesions': 0.7,
            'artery_vein': 0.3,
            'od_oc': 0.0,
            'tessellation': 0.7,
            'myopia': 0.5,
            'lesion_dr': 0.7,
            'lesion_amd': 0.7,
            'lesion_others': 0.7,
        }

        # ---- Minimum connected-component sizes (mask cleaning) --------------------
        self.analysis_min_component_size_default: int = 5
        self.analysis_min_component_size_per_task = {
            'lesion_s1': 15,
            'lesion_s2': 15,
            'lesion_s3': 15,
            'possible_lesions': 15,
            'artery_vein': 10,
            'od_oc': 50,
            'tessellation': 20,
            'myopia': 15,
            'lesion_dr': 15,
            'lesion_amd': 15,
            'lesion_others': 15,
        }

    # ---- Accessors -----------------------------------------------------------------
    def get_task_name(self, task_idx: int) -> str:
        if task_idx < len(self.task_names):
            return self.task_names[task_idx]
        return f'task_{task_idx}'

    def get_display_name(self, task_name: str) -> str:
        """No display mapping: internal names are final names."""
        return task_name

    def to_internal_task_name(self, name: str) -> str:
        return name

    def get_lesion_types(self, task_name: str) -> Dict[int, str]:
        return self.lesion_type_mappings.get(task_name, {})

    def get_grouped_lesion_types(self, group_name: str) -> Dict[int, str]:
        return self.grouped_lesion_type_mappings.get(group_name, {})

    def get_task_class_mapping(self, task_name: str) -> Dict[int, str]:
        if task_name in self.lesion_type_mappings:
            return self.lesion_type_mappings[task_name]
        if task_name in self.grouped_lesion_type_mappings:
            return self.grouped_lesion_type_mappings[task_name]
        return self.task_class_mappings.get(task_name, {})

    def get_grouped_lesion_names(self) -> List[str]:
        return self.grouped_lesion_names.copy()

    def is_lesion_task(self, task_name: str) -> bool:
        return task_name in self.lesion_task_names or task_name in self.grouped_lesion_names

    def get_output_type(self, task_name: str) -> str:
        return self.task_output_types.get(task_name, 'multiclass')

    def get_overlay_alpha(self, task_name: str) -> float:
        return self.overlay_alpha_per_task.get(task_name, self.overlay_alpha_default)

    def get_prob_threshold(self, task_name: str) -> float:
        return float(self.prob_threshold_per_task.get(task_name, self.prob_threshold_default))

    def get_analysis_min_component_size(self, task_name: str) -> int:
        return int(self.analysis_min_component_size_per_task.get(
            task_name, self.analysis_min_component_size_default
        ))
