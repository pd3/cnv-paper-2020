This is a collection of scripts and pipelines used for the paper XY. Note that this repository is only intended for inspiration and transparency. It is NOT complete, it is NOT maintained and it is NOT supported.

```
# Pipelines used to process and pre-format input data and resources
# The pipelines use the vr-runner framework
#   https://github.com/VertebrateResequencing/vr-runner
run-prep-1000gp
run-prep-canoes
run-prep-clamms
run-prep-cnsolidate
run-prep-convex
run-prep-curated
run-prep-decipher
run-prep-dgv
run-prep-xhmm

# The main random forest merging and filtering
run-merge-cnvsets
run-learn-cnvs
random-forest.py
run-prep-learned

# Analyses for the paper
run-cnv-paper
percentage-of-small-dels-accessible.v2
run-parental-origin
run-permutation-test
run-recurrence-test
```

