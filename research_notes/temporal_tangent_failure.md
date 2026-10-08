# Temporal address head

On the held-out 20-prompt 10-step Qwen2.5-0.5B-Instruct experiment, the rank-16 learned address head fails the quality gate: 30% active rows yielded mean KL 0.139 and 78% top-1 agreement; 40% active rows yielded mean KL 0.105 and 83% top-1 agreement. The oracle frontier remains promising but non-deployable. Next isolate previous oracle-mask teacher forcing from previous predicted-mask exposure and compare a persistence-only baseline.
