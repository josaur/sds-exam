# Quick Start Guide

## Installation

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn pyreadstat
```

## Running the Notebook

```bash
jupyter notebook replication_python.ipynb
```

Then: **Cell → Run All**

## What Gets Replicated

✅ Simple 2×2 DiD (ATT ≈ 14.3 %pts)
✅ Synthetic Control Method (Gap ≈ 12.8 %pts)
✅ Placebo Tests (p < 0.01)
✅ All three main figures (inline display)

## Troubleshooting

**Q: Import errors?**
A: Run first cell which installs dependencies with `%pip install`

**Q: File not found errors?**
A: Make sure you're in the `exam_causal_inference/` directory

**Q: Figures not displaying?**
A: Make sure you're using Jupyter (not just Python)

**Q: Still getting RuntimeError?**
A: Run `python test_notebook.py` to verify functions work

## Files

- `replication_python.ipynb` - Main notebook
- `test_notebook.py` - Test suite
- `README.md` - Full documentation
- `SOLUTION.md` - Details on bug fixes
- `replication/` - Original R code and data
