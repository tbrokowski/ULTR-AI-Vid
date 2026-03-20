# SA Finetuning Result Integrity Notes

These are the issues that could genuinely distort the reported results or make the run behave differently from what the config said.

## Issues that could affect SA full-finetune AUROC

| Problem | Why it mattered | Fix | Could explain a higher SA AUROC now? |
| --- | --- | --- | --- |
| No real validation split | The "best" checkpoint was chosen from training AUROC, so model selection could overfit and the reported `best_metric` was misleading. | Added a real internal SA train/val split and select the checkpoint from validation AUROC only. | Yes |
| Oversampling was not actually happening | Positive SA patients were duplicated in a list, but the dataset handling removed duplicates, so class balancing silently failed. | Replaced that logic with a real `WeightedRandomSampler` in the trainer. | Yes |
| Missing pathology labels were treated like true negatives | Unknown pathology labels were fed to the model as `0`, which added wrong supervision and could push training in the wrong direction. | Kept real `0/1` labels, masked missing labels, and stopped treating "missing" as "negative". | Yes |
| The model ignored the configured pathology-loss weight | The run was not actually following the experiment config, so the auxiliary pathology task could influence training more than intended. | Made the model read the pathology settings from config; for SA finetuning the pathology loss is now disabled. | Yes |
| Evaluation still had RL exploration noise | The same checkpoint could give slightly different predictions on different eval passes, so AUROC was not fully reproducible. | Disabled exploration noise during `eval()` and kept it only for training. | Mostly a correctness fix, not the main reason |

## Issue that could corrupt Benin transfer evaluation

| Problem | Why it mattered | Fix | Affects the SA AUROC directly? |
| --- | --- | --- | --- |
| Benin re-evaluation could keep using the SA dataset module | The Benin transfer score could be computed with the wrong loader behavior. | Explicitly restore the original Benin dataset module before Benin evaluation. | No |

## Short takeaway

The most likely reasons the SA full-finetune AUROC improved are:

- real class balancing instead of broken oversampling
- validation-based checkpoint selection instead of train-based selection
- removing wrong pathology supervision from missing labels

The deterministic-eval and Benin-loader fixes are still important, but they are mainly about making the reported numbers trustworthy rather than directly boosting SA performance.
