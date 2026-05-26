# Output Filtering Notes

Filter generated augmentation outputs before using them in downstream training.

## Suggested Filters

- Drop empty outputs.
- Drop exact duplicates when diversity is required.
- Flag outputs with extreme length changes.
- Flag examples where numbers or named entities changed unexpectedly.
- Keep method-specific failure examples for later prompt or model changes.

## Review Samples

Review BART, MLM replacement, MLM insertion, and T5 outputs separately. Each method has different failure patterns and should not share one quality threshold blindly.

## Metadata

Save the generation method, model revision, random seed, filter counts, and final row count with every filtered dataset.
