# Training dataset workflow

Local frames and labels are listed in `session_manifest.csv`. See [EXTERNAL_DATASETS.md](EXTERNAL_DATASETS.md) for approved sources, class mapping, access requirements, and split rules.

After accepting the Hugging Face dataset conditions, install `datasets`, set `HF_TOKEN` if needed, and run:

```powershell
python tools/import_hf_sidewalk_semantic.py --max-samples 0
```

The importer places converted sidewalk-semantic images and approximate segmentation polygons in **train only**. It does not alter `val1`. Review the generated masks visually before training. The masks are semantic regions converted into polygon pseudo-instances, not manually annotated object instances.

SANPO is not downloaded automatically because the complete release is about 6 TB. Select a small set of official training sessions with panoptic annotations; keep each session entirely in train and preserve source metadata. A SANPO-specific conversion still needs to be checked against the exact downloaded annotation files before merging.

ROD contains bounding boxes and must be trained as a separate YOLO detection model. Do not copy its box labels into the YOLO segmentation dataset. The app can be evaluated separately with a detection checkpoint.
