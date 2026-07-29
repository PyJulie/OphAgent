from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    tmp_input: Path
    npy_seg: Path
    enface: Path
    centroid_out: Path
    results: Path
    logs: Path
    manifests: Path
    work_root: Path
    work_segmentation: Path

    @classmethod
    def create(cls, root: Path) -> "OutputLayout":
        return cls(
            root=root,
            tmp_input=root / "tmp_input",
            npy_seg=root / "npy_seg",
            enface=root / "enface",
            centroid_out=root / "centroid_out",
            results=root / "results",
            logs=root / "logs",
            manifests=root / "manifests",
            work_root=root / "_work",
            work_segmentation=root / "_work" / "segmentation",
        )

    def ensure(self, *, include_tmp_input: bool) -> None:
        dirs = [
            self.root,
            self.npy_seg,
            self.enface,
            self.centroid_out,
            self.results,
            self.logs,
            self.manifests,
            self.work_root,
            self.work_segmentation,
        ]
        if include_tmp_input:
            dirs.append(self.tmp_input)
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
