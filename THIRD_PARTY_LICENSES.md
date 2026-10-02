# Third-party licences

This repository is CC BY-NC-SA 4.0 as a whole (see `LICENSE`), because it vendors and adapts RADAR.

| Component | Where | Licence |
|---|---|---|
| RADAR code and weights, Alibaba DAMO Academy | `worker/vendor/damo-radar/`, `worker/radar_worker/infer.py` (adapted from `evaluate()`), weights on the Modal Volume | CC BY-NC-SA 4.0. Paper: "An expert-level generalist AI for abdominal CT diagnosis", Science 393(6817), eaec6129, doi 10.1126/science.aec6129. Weights at huggingface.co/radar-generalist/RADAR. Research use only. |
| LAVIS (inside the vendored code) | `worker/vendor/damo-radar/RADAR_inference/dynamic_network_architectures/med.py` | BSD-3-Clause |
| nnU-Net dynamic network architectures (inside the vendored code) | `worker/vendor/damo-radar/RADAR_inference/dynamic_network_architectures/` | Apache-2.0 |
| MONAI | worker runtime dependency | Apache-2.0 |
| persona (`@runtypelabs/persona`) | `web/` dependency; `src/radar_desk/chat/wire.py` ports `persona-wire/src/index.ts` | MIT |
| NiiVue (`@niivue/niivue`) | `web/` dependency | BSD-2-Clause |

`worker/vendor/damo-radar/THIRD_PARTY_LICENSES.md` is upstream's own list and applies to everything under that folder.
