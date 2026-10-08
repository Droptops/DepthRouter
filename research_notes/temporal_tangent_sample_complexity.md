# Address-head data bottleneck

The rank-16 temporal address head has 111,361 trainable parameters but only 160 training states (20 prompts x 8 decode steps). Held-out 30% selection yields 57.4% oracle-set recall, KL 0.139 and 78% top-1. Prior-oracle-mask training versus prior-predicted-mask evaluation creates an exposure-bias risk. Next compare teacher-forced, predicted-history, no-history and copy-previous baselines, then scale training data before interpreting higher-rank sweeps.
