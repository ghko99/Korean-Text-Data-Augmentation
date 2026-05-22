# Model Cache Notes

The augmentation examples load pretrained models from Hugging Face. Keep cache and artifact handling explicit so runs are portable.

## Before Running

- Confirm the target model name and revision.
- Check whether GPU execution is required for the batch size.
- Set a local Hugging Face cache directory when running on shared machines.
- Record package versions for `transformers`, `torch`, and tokenizer dependencies.

## Generated Artifacts

Keep model caches, generated mapping files, and large augmented datasets outside normal Git history unless they are curated examples. If an output is committed, keep it small and document the generation settings.

## Reproducibility

Save the model id, revision, random seed, and generation parameters with each augmentation output.
