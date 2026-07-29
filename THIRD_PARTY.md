# Third-Party Source Notice

OphAgent does not include model weights in the source repository. Public source
components are installed from their original repositories at the revisions
listed in `ophagent/resources/components.yaml`.

| Component | Upstream | Locked revision | Declared license |
|---|---|---|---|
| Chinese-CLIP | https://github.com/OFA-Sys/Chinese-CLIP | `31863c707501bf1605d36842f43deb78793dbc5d` | MIT |
| FLAIR | https://github.com/jusiro/FLAIR | `9afac72a5d58851923f7b81f10c0d861f14c78f5` | Apache-2.0 |
| EFIQA | https://github.com/penway/EFIQA | `e1c64d4bfc6059ebc1395052867bec651fd67d34` | MIT |
| OCTCubeM | https://github.com/ZucksLiu/OCTCubeM | `5b2392287c67155de841e75e701450dda89309c5` | BSD-2-Clause |
| RetiZero | https://github.com/LooKing9218/RetiZero | `d72aadc692fbe33b182c79711bccb397edffb419` | Not declared upstream |
| FMUE | https://github.com/yuanyuanpeng0129/FMUE | `b07ba8a797d6440826f3870f73faf567963ffc15` | Not declared upstream |

The installer preserves each upstream repository and its notices. RetiZero and
FMUE are excluded from automatic installation because no upstream license was
declared at the audited revision. Local cloning requires explicit operator
acknowledgement and does not authorise redistribution.

The OphAgent repository contains inference-only implementations for its
glaucoma and PDR checkpoints and the integration logic for EFIQA. These files
do not bundle the external training repositories or any learned parameters.

The repository also includes the checkpoint-compatible inference portions of
ReT-SAM 2.0 and G-DISC OCT, with redistribution approved for this project.
ReT-SAM model files derived from MONAI retain their Apache-2.0 headers. The
G-DISC W-Net implementation records its source lineage in the file header.
Learned parameters, histogram calibration assets, local metadata, and
operational scripts are not bundled.

Model repositories, pretrained backbones, datasets, and separately distributed
weights remain subject to their own licenses, access controls, and use terms.
The OphAgent project license does not replace those terms.
