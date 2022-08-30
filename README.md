# multitask-learning-framework

Multi-task learning (MTL) framework for jointly training models with shared
backbones and task-specific heads. Includes runnable demos for NLP (BERT shared
trunk -> sentiment + NER + topic) and CV (ResNet trunk -> classification +
segmentation).

## Why

Training one model per task is expensive and ignores the structure that tasks
in the same domain often share. MTL lets a single backbone amortise feature
extraction across related tasks, which can improve sample efficiency and
sometimes accuracy on the harder/smaller tasks.

The tricky bit is loss balancing: a naive sum of task losses lets one task
dominate. This repo implements three loss-weighting strategies side by side so
the trade-off is easy to inspect.

## What's in here

- shared backbone (BERT for NLP, ResNet for CV) with a `ModuleDict` of heads
- pluggable loss-weighting: uniform, learnable uncertainty (Kendall et al 2018),
  GradNorm (Chen et al 2018)
- per-task training curve and gradient-norm visualisation tools
- FastAPI app exposing `/predict_nlp` and `/predict_vision`
- demos: AG News + CoNLL-style toy NER (NLP), Pascal VOC subset (CV)

## Layout

```
src/
  core/
    mtl_model.py        shared backbone + ModuleDict of heads
    trainer.py          multi-task loss aggregation, train/eval loops
    loss_weighting.py   uniform, uncertainty, GradNorm
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
    main.py             FastAPI app
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
# mlflow on http://localhost:5000
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

Serve:

```bash
uvicorn src.api.main:app --reload
```

## Loss weighting

| strategy     | extra params         | notes                                       |
|--------------|----------------------|---------------------------------------------|
| uniform      | none                 | sum of losses, scaled by 1/N                |
| uncertainty  | log-sigma per task   | Kendall, Gal, Cipolla (2018)                |
| gradnorm     | one weight per task  | Chen et al (2018), balances grad magnitudes |

To switch weighters change `loss_weighting.strategy` in the YAML config.

## Tests

```bash
make test
```

The test suite covers:
- `test_mtl_model.py`: forward shapes, head dict structure
- `test_loss_weighting.py`: each strategy returns scalars, gradnorm grads flow
- `test_trainer.py`: single train step under all three strategies
- `test_api.py`: FastAPI endpoint contracts

## Design notes

The shared backbone is wrapped in an `MTLModel` that holds a `nn.ModuleDict` of
heads keyed by task name. Forward returns a dict `{task_name: head_output}` so
the trainer can compute per-task losses without bespoke routing.

`Trainer` accepts a `LossWeighter` strategy. At each step it:
1. computes raw losses `{task: scalar}`,
2. asks the weighter for a combined scalar loss (and possibly side-effects like
   updating GradNorm weights),
3. backprops once through the shared trunk.

GradNorm needs the gradient norm of each task loss w.r.t. the *last* layer of
the shared backbone. The implementation hooks into `model.backbone.last_layer`
to grab those.

## References

- Kendall, Gal, Cipolla. *Multi-Task Learning Using Uncertainty to Weigh Losses
  for Scene Geometry and Semantics*. CVPR 2018.
- Chen, Badrinarayanan, Lee, Rabinovich. *GradNorm: Gradient Normalization for
  Adaptive Loss Balancing in Deep Multitask Networks*. ICML 2018.
- Caruana. *Multitask Learning*. Machine Learning 1997.

## License

MIT.
