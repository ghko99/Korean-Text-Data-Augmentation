# Augmentation Quality Checks

Use lightweight checks before promoting generated Korean text augmentation outputs into downstream experiments.

## Manual Spot Checks

- Confirm the augmented sentence preserves the source meaning.
- Check for awkward word order or broken spacing.
- Verify named entities and numbers are not changed unexpectedly.
- Inspect examples from BART, MLM replacement, MLM insertion, and T5 separately.

## Automated Checks

- Track source and augmented sentence length ratios.
- Count duplicate outputs.
- Count unchanged outputs when diversity is expected.
- Flag outputs with missing punctuation or empty strings.

## Reporting

When sharing an augmented dataset, include the method, model name, generation settings, sample count, and any filters applied after generation.
