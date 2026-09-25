# Frozen Teacher-direct versus Student-B: efficiency and perturbation robustness

Protocol locked before measurements in `AUDIT_PROTOCOL.md`. All inputs are the same 16 frozen V4 **val** pilot plants (Core 12 and Diagnostic 4), 518×518 RGB `phenotype_roi_v1`. Runs used `cv` / `neaucs2-OMEN`, RTX 3090 24 GB, driver 560.35.05, Intel Core i9-12900K, PyTorch 2.9.1+cu128, CUDA 12.8, 62.6 GiB system RAM reported by Linux. The GPU was otherwise idle at preflight. Each method ran in an isolated process, models loaded once, 2 complete 16-image warm-up passes and 10 measured passes per stage (`N=160` image timings per method/stage). CUDA synchronization surrounds timing. No image-file reading, ROI construction, model initialization, checkpoint I/O or batching is included in the latency numbers. Paths are mapped back to standardized-source pixels before the frozen 240-point phenotype geometry. This is **in-memory image-to-result latency**, not whole scanner-file processing time. A preliminary timing that omitted this coordinate conversion was discarded, and both methods were fully rerun at the same preregistered N.

## Same-hardware latency

All times in milliseconds; parentheses are Q1–Q3 and IQR. The ratio is the median of 160 image/repetition-paired Teacher/Student time ratios, not a quotient of medians.

| Timed stage | Teacher-direct median (Q1–Q3; IQR) | Student-B median (Q1–Q3; IQR) | Paired T/S ratio |
| --- | ---: | ---: | ---: |
| Point generation only | 685.19 (595.42–859.33; 263.91) | 13.14 (13.11–13.27; 0.17) | 50.94× |
| Shared point-conditioned graph + local decoder | 27.10 (24.39–29.78; 5.39) | 24.86 (23.54–27.83; 4.29) | 1.07× |
| End-to-end phenotype geometry | 720.26 (627.22–899.80; 272.58) | 42.33 (38.80–46.62; 7.81) | 17.75× |

Every one of the 16 plant-wise median end-to-end ratios exceeds 14.97× (range 14.97–24.42×). The large difference is in **point generation**: Teacher repeats DINOv2 representation, structure proposal and consensus over nine views; Student predicts with one learned heatmap forward/decode. Shared graph/decoder is nearly the same runtime and should not be advertised as accelerated. No performance claim is made for other hardware, batch sizes or raw-file preprocessing.

The preregistered peak GPU metric is `torch.cuda.max_memory_allocated` (tensor allocation, **not** total driver/device occupancy). RAM is 5-ms-sampled process peak RSS. Values are MiB; stages share a loaded process, but methods use separate processes.

| Stage | Teacher peak CUDA allocated / RAM RSS | Student peak CUDA allocated / RAM RSS |
| --- | ---: | ---: |
| Point generation | 137.3 / 1298.5 | 122.3 / 1336.2 |
| Shared graph + decoder | 93.3 / 1299.1 | 96.4 / 1338.9 |
| End-to-end | 137.3 / 1301.2 | 122.3 / 1338.9 |

Student has a modest lower peak CUDA *allocated* footprint in point/end-to-end timing, but approximately 37 MiB **higher** process RAM peak. Neither result should be recast as total GPU occupancy or universal memory efficiency. GPU driver context and library caching are outside the allocation counter; 5-ms RSS polling may miss shorter transients.

## Pre-fixed mild perturbations

The only views are the existing `g1.make_transforms` horizontal flip and +10° rotation, plus brightness ×0.85 from the locked training augmentation bound. Image and ROI/basal masks receive the same geometry; points and complete paths are inverse-mapped to the identity image. Each frozen method runs once per image/view. Point F1 is one-to-one within `0.05D`; path matching and 240-point PCHIP geometry use the unchanged phenotype protocol. Length/angle medians are **conditional on a matched path** and must be read alongside path coverage. `N` is 12 Core or 4 Diagnostic plants per row; these are perturbation diagnostics, not independent external generalization.

| Group / view | Method | Median point-count absolute change | Median point F1 | Exact path-count agreement | Original paths matched | Median matched length difference | Median branch-angle difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Core / flip | Teacher | 1 | .857 | 10/12 | 23/24 | .61% | 3.11° (11) |
| Core / flip | Student | 1 | .857 | 10/12 | 21/22 | .54% | 2.25° (10) |
| Core / rotate +10° | Teacher | 2 | .586 | 5/12 | 16/24 | 2.03% | 7.34° (7) |
| Core / rotate +10° | Student | .5 | .857 | 11/12 | 20/22 | 2.41% | 6.73° (8) |
| Core / brightness ×.85 | Teacher | 1 | .866 | 11/12 | 21/24 | .17% | .40° (11) |
| Core / brightness ×.85 | Student | 0 | 1.000 | 11/12 | 21/22 | .22% | .15° (9) |
| Diagnostic / flip | Teacher | 0 | .900 | 4/4 | 12/12 | .37% | 3.32° (7) |
| Diagnostic / flip | Student | .5 | .708 | 4/4 | 9/9 | 1.10% | 5.02° (6) |
| Diagnostic / rotate +10° | Teacher | 2 | .667 | 2/4 | 9/12 | 1.28% | 11.22° (5) |
| Diagnostic / rotate +10° | Student | .5 | .873 | 1/4 | 7/9 | 8.23% | 11.57° (4) |
| Diagnostic / brightness ×.85 | Teacher | 1 | .856 | 2/4 | 10/12 | .13% | .08° (6) |
| Diagnostic / brightness ×.85 | Student | 0 | 1.000 | 3/4 | 8/9 | .16% | .89° (5) |

Core rotation gives Student better point repeatability and path-count agreement, while Diagnostic rotation gives Student worse path-count agreement and larger *conditional* length change. A count agreement does not guarantee matching the same paths; the separate matched-path numerator remains essential. Brightness stabilizes Student's point count but does not erase its original Diagnostic path loss. This is mixed robustness evidence, not a claim of universal superiority. No GT was read by the benchmark or perturbation scripts, and no threshold or model parameter was selected from these outcomes.

The local private measurement source is `experiment/phenotype_pilot_protocol/runtime/method_gate_20260925/measurements/`, excluded from Git; the corrected aggregate JSON is SHA-256 `0fe2e09e1688e40fe490b506a3784cea70fa3c0fc98608acc6a9ef2a16a4aecb`. Reproduction scripts and preregistration are public under `experiment/method_gate_20260925/`. A non-GT unit test confirms the end-to-end coordinate conversion equals the frozen pilot evaluator on every saved Student path in these 16 plants. The two independent models use the same frozen graph, decoder and geometry. V4 test remains locked.
