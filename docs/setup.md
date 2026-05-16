# Setup Notes

This project contains BART, BERT, and T5 based Korean text augmentation examples.

## Environment

The README was developed around Python 3.8 on Windows 11. A virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Model downloads

The examples load Hugging Face models at runtime. Make sure the machine has network access or a populated local cache before running augmentation scripts.

## Generated files

Keep generated model checkpoints, logs, and output datasets out of Git unless they are small curated examples intended for documentation.
