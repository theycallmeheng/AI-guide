"""Fine-tune both local ROD checkpoints; validate only, reserve test for final acceptance."""
import argparse
import gc
import json
from pathlib import Path
from audit_detection_dataset import audit


def train_candidates(data, manifest, models, output, epochs=50, batch=4):
    import torch
    from ultralytics import YOLO
    checked = audit(data, manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, path in enumerate(models):
        assert Path(path).is_file(), f'Missing local weights: {path}'
        model = YOLO(str(path))
        expected = [str(n).strip().lower() for n in checked['names']]
        actual = [str(model.names[i]).strip().lower() for i in range(len(model.names))]
        assert actual == expected, 'Class order differs from ROD checkpoint; explicitly remap labels before training'
        model.train(data=str(Path(data).resolve()), epochs=epochs, batch=batch, imgsz=640,
                    device=0 if torch.cuda.is_available() else 'cpu', workers=0,
                    seed=42, deterministic=True, patience=15, cache=False,
                    project=str(output), name=f'candidate_{index}_{Path(path).stem}', exist_ok=False)
        best = str(model.trainer.best)
        metrics = YOLO(best).val(data=str(Path(data).resolve()), split='val', imgsz=640, batch=batch, workers=0)
        summary.append(dict(initial_weights=str(path), best=best, validation=metrics.results_dict))
        (output/'candidate_results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        del model, metrics
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--models', nargs=2, required=True)
    p.add_argument('--output', default='runs/detection_candidates')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=4)
    args=p.parse_args()
    train_candidates(args.data,args.manifest,args.models,args.output,args.epochs,args.batch)
