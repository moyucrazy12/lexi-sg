# Precomputed room transitions

LEXI-SG supports two values of `room_segmentation.source`:

- `native` (the default when omitted): the original DINO room-label hysteresis,
  including its corridor/doorway handling and existing carry behavior.
- `external`: precomputed RADIO–TCN transitions. Each boundary starts a new
  room directly; `method`, hysteresis settings, and `transitions_as_nodes` do
  not influence cuts. Every node has the label `room`, so no artificial
  corridor or doorway nodes are introduced.

Both modes use the same MapAnything room reconstruction, boundary-pair
estimation, room graph, feature-based loop closure, overlap checks, and Sim(3)
optimization. External mode still extracts DINO image features for every RGB
frame, including features used by loop closure and object segmentation. It
skips DINO text-prompt classification for transition decisions. No RADIO–TCN
model code is run or changed.

## Configuration and JSON

Use an existing dataset YAML and change only the source and boundary file:

```yaml
room_segmentation:
  source: external
  boundary_file: /path/to/room_boundaries.json
```

The minimal format is:

```json
{
  "boundaries": [
    {"frame_index": 120},
    {"frame_index": 245},
    {"frame_index": 390}
  ]
}
```

Indices are zero-based positions in **the exact lexicographically sorted RGB
list used by LEXI-SG**, before reconstruction subsampling. `120` assigns
frames `[0, 120)` to the first room and frame `120` to the second. The example
therefore requires at least 391 input images; it is illustrative, not a set of
RADIO–TCN predictions for a particular sequence. An empty `boundaries` list
creates one room. Indices must be integers, strictly increasing, unique, and
between `1` and `num_input_frames - 1`. Frame `0` already starts the first room;
the sequence endpoint is not an additional cut.

LEXI-SG also accepts the existing richer RADIO–TCN/VGGT-SLAM export produced by
`tcnn_pipeline/room_transition_detection/room_transition/visualize_rerun_json.py`
(`export_boundaries_json`) and consumed by `VGGT-SLAM/main.py`:

```json
{
  "sequence": "example_sequence",
  "fps": 20.0,
  "num_frames": 500,
  "boundaries": [
    {
      "frame_index": 120,
      "image_name": "000120.png",
      "time_seconds": 6.0,
      "emitted_frame_index": 160,
      "emitted_time_seconds": 8.0,
      "score": 0.8
    }
  ]
}
```

A top-level list of boundary objects is accepted too, matching VGGT-SLAM.
`frame_index` is the physical cut. `emitted_frame_index` and emitted times are
detector latency metadata and never shift a cut. `sequence` and `model` are
retained as provenance; detector sequence aliases need not equal directory
names. `image_name` must exactly match an RGB basename, including extension.

### Sampling and timestamp alignment

Prefer exporting **both `num_frames` and `image_name`**. Minimal index-only
input asserts that the detector ran on this exact sorted sequence; without
identity metadata, a sampling mismatch cannot be detected. LEXI-SG logs that
assumption. Providing `num_frames` detects differing sequence lengths:

- With the same sequence length (or no declared length), an index and an
  accompanying filename must agree.
- With a different declared length, every boundary must match an exact
  `image_name` or an explicitly configured filename timestamp. The source
  index is retained and the cut is mapped to that image's current sorted
  position. Index-only input is rejected even when the index is in range.
- If a boundary image was removed by subsampling, loading fails. There is no
  nearest-frame rounding, clamping, silent dropping, or scaling by FPS.
  Re-export cuts for the actual input sequence or retain the boundary images.
- VGGT-SLAM can sort unpadded names numerically; LEXI-SG keeps its existing
  lexicographic order. Prefer padded filenames and inspect the image-name
  checks when reusing exports.

Timestamp-only cuts require explicit knowledge that **numeric filename stems
are timestamps**. For example, AOD's device timestamps can be described as:

```yaml
room_segmentation:
  source: external
  boundary_file: /path/to/timestamp_boundaries.json
  frame_timestamp_unit: ns
```

```json
{
  "timestamp_unit": "ns",
  "boundaries": [{"timestamp": "883601046500"}]
}
```

This is accepted only if `883601046500` exactly matches an input filename
stem and that frame is not the first frame. Units `s`, `ms`, `us`, and `ns`
are supported, with exact decimal conversion between the declared JSON and
filename units. Use strings or integers for large timestamps to retain
precision. If JSON `timestamp_unit` is omitted, it defaults to the configured
filename unit. Duplicate numeric timestamps, nonnumeric stems, unmatched
timestamps, and contradictory filename/timestamp identities are rejected.
No unit is inferred from the size of a filename number.

The VGGT-SLAM export's `time_seconds = frame_index / fps` is relative detector
time, not a dataset timestamp. It is ignored when an index or image identity
is available and rejected as the only alignment field. No origin or frame
rate is guessed. For HM3D frame-number filenames, use indices or filenames
rather than interpreting their stems as timestamps.

## Run the same sequence with both sources

Run from the LEXI-SG repository with its virtual environment active. Replace
the boundary-file paths with actual detector exports for the chosen scene.
Use matching reconstruction, loop closure, priors, and object-segmentation
settings for both runs. `output_folder` is deleted and recreated at startup,
so keep separate directories as below.

### AOD

For the workspace's extracted layout (`Aria-Office-Dataset` is an extra level):

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python main.py \
  --config config/datasets/aod.yaml \
  --set data_path=data/aod/Aria-Office-Dataset/gf-2 \
  --set room_segmentation.source=native \
  --set output_folder=output/gf-2_native

HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python main.py \
  --config config/datasets/aod.yaml \
  --set data_path=data/aod/Aria-Office-Dataset/gf-2 \
  --set room_segmentation.source=external \
  --set room_segmentation.boundary_file=/path/to/gf-2_radio_tcn.json \
  --set output_folder=output/gf-2_radio_tcn
```

The current workspace's `gf-2/rgb` has 230 images, while some existing detector
exports describe 917 frames. Such exports require exact filename alignment;
do not copy their raw indices into a 230-frame JSON. Some detected images may
be missing from the sampled input and require a new export.

For a 16 GB GPU, select `config/datasets/aod_16gb.yaml` in **both** commands.
The environment setting bypasses expired implicit Hugging Face credentials
for public weights; authenticated/private downloads require valid credentials.

### HM3D

Point both runs at the same scene directory containing `rgb/`:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python main.py \
  --config config/datasets/hm3d.yaml \
  --set data_path=/path/to/hm3d/00824 \
  --set room_segmentation.source=native \
  --set output_folder=output/hm3d_00824_native

HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python main.py \
  --config config/datasets/hm3d.yaml \
  --set data_path=/path/to/hm3d/00824 \
  --set room_segmentation.source=external \
  --set room_segmentation.boundary_file=/path/to/00824_radio_tcn.json \
  --set output_folder=output/hm3d_00824_radio_tcn
```

Add `--set log_level=INFO` for per-batch frame ranges and sampling-remap logs.
For room mapping alone, add `--set object_segmentation.enabled=false` to both
commands. `--visualise` works with either source.

## Batch semantics and run summaries

External mode emits `[previous_cut, next_cut)` without native carry frames,
midpoint splitting, or overlap. The last room includes the final RGB image,
even when it contains a single frame. Every image is assigned exactly once;
the existing reconstruction `batch_size` can still subsample frames *within*
each room. Boundary-pair reconstruction connects adjacent rooms without
changing their membership.

Native hysteresis and carry semantics remain unchanged for comparison. The
original `main.py` also removes the last image from its final native batch;
that existing behavior is deliberately retained only in native mode. Its
coverage is visible in the summary rather than silently applied to external
cuts. Native corridor/doorway nodes still follow `transitions_as_nodes`.

Each run writes `<output_folder>/room_segmentation_summary.json` at sequence
setup, after each successfully reconstructed batch, and at completion. It
contains the source, ordered-filename SHA-256, input frame count, actual batch
ranges (end exclusive), actual boundary indices and filenames, room-batch
count, assigned/unassigned counts, and overlap counts. External runs also
record the boundary file, detector metadata, original source indices, and
resolved identities. `segmentation_complete` is false for partial runs.
Batch counts describe segmentation before any loop-closure merges of graph
nodes, not the number of nodes remaining in the optimized graph.

Invalid external boundaries are rejected before models are loaded or the
output directory is deleted. Sequence identity is checked again when the
validated plan is bound to the frame loop.

## Lightweight tests

```bash
python -m pytest -q tests
```

Tests cover parsing, malformed/out-of-range/duplicate cuts, sorted-frame
alignment, mismatched sampling, filename and timestamp checks, exact external
partitions, empty boundaries, one-frame final rooms, final-frame preservation,
native hysteresis/carry/folding regressions, summaries, and early validation.
Pipeline tests use stand-ins for neural models and optimization, while
exercising the real batching, subsampling, boundary-pair estimation, and graph
construction paths. They require no model downloads or GPU.
