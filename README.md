# multitask-learning-framework

Sandbox for multi-task learning (MTL) with a shared backbone and task-specific
heads. Runnable toy demos for NLP (BERT trunk with sentiment + topic + NER
heads) and CV (ResNet trunk with classification + segmentation heads).

## Why

Training one model per task is expensive and ignores structure that tasks in
the same domain often share. MTL lets a single backbone amortise feature
extraction across related tasks. The tricky bit is loss balancing: a naive sum
of task losses lets one task dominate. This repo implements two loss-weighting
strategies (uniform, learnable uncertainty) so the trade-off is easy to
inspect on toy data.

## Scope and honest limits

This is a scaffold, not a training platform.

- Data is synthetic in both demos: NLP uses 8 hand-written sentences repeated
  32x; CV draws coloured shapes on a black background. No AG News, CoNLL,
  Pascal VOC, or any other public dataset is downloaded or used.
- The training scripts do not save model weights, only a `history.pt` with
  logged metrics. There is no exported checkpoint to serve.
- The FastAPI app under `src/api/main.py` is an endpoint-shape stub. It uses
  simple keyword heuristics for the NLP response and a constant response for
  vision. It does not run the trained model.
- Task lists are hardcoded in `examples/run_nlp.py` and `examples/run_vision.py`.
  The YAML configs only drive the loss weighter, seed, batch sizes, learning
  rate, and epoch count.

## What's in here

- shared backbone (BERT for NLP, ResNet for CV) with a `ModuleDict` of heads
- loss-weighting strategies: uniform, learnable uncertainty (Kendall et al 2018)
- per-task training curve and task-weight visualisation
- FastAPI stub with `/predict_nlp` and `/predict_vision` returning heuristic /
  constant outputs (no model inference)

## Layout

```
src/
  core/
    mtl_model.py        shared backbone + ModuleDict of heads
    trainer.py          multi-task loss aggregation, train/eval loops
    loss_weighting.py   uniform, uncertainty
    tasks.py            base Task abstraction
  nlp/
    bert_backbone.py
    heads.py            sentiment, ner, topic
    data.py
  vision/
    resnet_backbone.py
    heads.py            classifier, segmenter
    data.py
  api/
    main.py             FastAPI stub
  visualize.py
configs/
  nlp.yaml
  vision.yaml
tests/
  test_mtl_model.py
  test_loss_weighting.py
  test_trainer.py
  test_api.py
notebooks/
  walkthrough.ipynb
examples/
  run_nlp.py
  run_vision.py
ci/
  test.yml.example      copy to .github/workflows/test.yml when ready
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Or use docker:

```bash
docker compose up --build
# api on http://localhost:8000
```

## Run

NLP demo (BERT shared, 3 heads):

```bash
python examples/run_nlp.py --config configs/nlp.yaml
```

Vision demo (ResNet shared, classification + segmentation):

```bash
python examples/run_vision.py --config configs/vision.yaml
```

Serve the stub endpoints:

```bash
uvicorn src.api.main:app --reload
```

## Loss weighting

| strategy     | extra params         | notes                                       |
|--------------|----------------------|---------------------------------------------|
| uniform      | none                 | sum of losses, scaled by 1/N                |
| uncertainty  | log-sigma per task   | Kendall, Gal, Cipolla (2018)                |

Switch by changing `loss_weighting.strategy` in the YAML.

## Tests

```bash
make test
```

The test suite covers:
- `test_mtl_model.py`: forward shapes, head dict structure
- `test_loss_weighting.py`: uniform and uncertainty combine + weight readback
- `test_trainer.py`: single train step under uniform and uncertainty
- `test_api.py`: FastAPI stub endpoint contracts

## Design notes

The shared backbone is wrapped in an `MTLModel` that holds a `nn.ModuleDict`
of heads keyed by task name. Forward returns a dict `{task_name: head_output}`
so the trainer can compute per-task losses without bespoke routing.

`Trainer` accepts a `LossWeighter` strategy. At each step it:
1. computes raw losses `{task: scalar}`,
2. asks the weighter for a combined scalar loss,
3. backprops once through the shared trunk.

## References

- Kendall, Gal, Cipolla. *Multi-Task Learning Using Uncertainty to Weigh Losses
  for Scene Geometry and Semantics*. CVPR 2018.
- Caruana. *Multitask Learning*. Machine Learning 1997.

## License

MIT.
