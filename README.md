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
scripts/
  smoke.py              tiny-CPU offline smoke (no GPU, no download)
ci/
  test.yml.example      copy to .github/workflows/test.yml when ready
```

## Quick start (tiny-CPU smoke, no GPU/download)

The two headline demos below (`examples/run_nlp.py`, `examples/run_vision.py`)
need a GPU, pretrained BERT / ResNet weights, and real datasets. To verify the
core multi-task machinery on a laptop with no GPU and nothing to download, run
the smoke instead. It builds a tiny from-scratch shared MLP encoder with three
heads (a multi-class classifier, a scalar regressor, and a binary classifier),
generates tiny synthetic multi-task data whose targets are learnable functions
of one shared input, trains a few steps through the real `MTLTrainer` with the
real learnable-uncertainty task weighter, and runs inference:

```bash
make smoke
# or: python scripts/smoke.py
```

Real output (CPU, torch 2.5.1, a couple of seconds):

```
Multi-task learning framework: tiny-CPU offline smoke
device=cpu (pinned) torch=2.5.1+cu121 cuda_available=True (ignored)
shared TinyEncoder trunk + 3 heads (category=classification, value=regression, positive=binary)

model parameters: 1831 (tiny; runs in well under a second on CPU)

epoch  0 | weighted_total=15.0133 | category(CE)=1.3838 value(MSE)=13.3571 positive(CE)=0.6835 | weights c=0.96 v=0.96 p=1.05
epoch  3 | weighted_total=2.6902 | category(CE)=1.2334 value(MSE)=0.9496 positive(CE)=0.5891 | weights c=0.81 v=0.82 p=1.32
epoch  7 | weighted_total=0.9818 | category(CE)=0.9081 value(MSE)=0.1156 positive(CE)=0.2093 | weights c=0.90 v=0.83 p=1.88
epoch 11 | weighted_total=-0.5717 | category(CE)=0.3467 value(MSE)=0.0661 positive(CE)=0.0493 | weights c=1.36 v=0.88 p=2.98

per-task loss change (first epoch -> last epoch):
  category : 1.3838 -> 0.3467
  value    : 13.3571 -> 0.0661
  positive : 0.6835 -> 0.0493
  weighted total: 15.0133 -> -0.5717

inference on 5 fresh examples -> per-task output shapes:
  category  (logits) : (5, 4)  expected (5, 4)
  value     (scalar) : (5,)  expected (5,)
  positive  (logits) : (5, 2)  expected (5, 2)

SMOKE PASS
```

Every per-task loss drops and the weighted total falls (it goes negative because
the uncertainty weighter adds a `log_sigma` regularizer per task, which turns
negative once the raw losses are small; that is expected, not a bug). The task
weights shift as training proceeds, which is the whole point of the weighter.

**The headline NLP / CV demos need a GPU, pretrained weights, and real data.**
This smoke uses a tiny from-scratch trunk, tiny synthetic data, and CPU only, so
it does not download BERT / ResNet or touch the network. It exercises the same
`MTLModel`, `MTLTrainer`, and `UncertaintyWeighter` the real demos use.

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

These two demos need a GPU, pretrained BERT / ResNet weights, and a download on
first run. For a no-GPU, no-download check use `make smoke` (see Quick start).

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
