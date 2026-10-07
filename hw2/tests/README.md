# Student Tests

Run these checks from the `handout` directory:

```bash
python -m unittest discover -s tests
```

The tests use small synthetic inputs and do not train a model. They are meant
to catch common shape mistakes, missing class weighting, missing input
dependence, and incorrect confusion-matrix or IoU calculations.
