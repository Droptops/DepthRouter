# Temporal tangent distillation: held-out failure

CI run 37749237107 passed, but the quality gate did not. At 30% active MLP width, mean KL was 0.138947 and top-1 agreement 78% across 20 prompts and 10 dense-reference decode states each. The rank-16 address head recovered 57.4% of oracle-selected rows. At 40% width, mean KL was 0.104816. No hardware speedup is established: the hook still executes the dense MLP. Next step is teacher-forced previous-mask versus predicted-mask ablation to isolate exposure bias.
