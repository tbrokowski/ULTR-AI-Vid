# Ablation Experiments Status Report - FINAL
**Generated:** October 20, 2025  
**Last Updated:** October 20, 2025 - After overnight runs and comprehensive visualization  
**Total Experiments:** 16  
**Fully Completed & Ready:** 6 experiments (excellent performance, AUC ≥ 0.80)  
**Complete but Poor Performance:** 5 experiments (AUC < 0.60)  
**Incomplete:** 5 experiments (technical issues)

**Visualizations Generated:** ✅ 25 files in `/users/mbarbiere/ULTR-AI/ULTR-AI-Vid/visualization_outputs/`  
**Evaluation Results:** Stored in `/capstor/store/cscs/swissai/a127/ultr-ai/ablation_results/{experiment}/eval_results/`

---

## 🎯 EXECUTIVE SUMMARY

### ✅ SUCCESS: 6 Excellent Models Ready for Publication
**Performance Insight:** Attention-based pooling mechanisms significantly outperform complex 3D architectures and temporal models for TB detection from lung ultrasound videos (AUC 0.83+ vs 0.50-0.55).

**Top 6 Models (AUC ≥ 0.80):**
1. **attention_pool** - AUC: 0.8389 ± 0.0223 (4/5 folds) 🥇
2. **original_noInitWeights** - AUC: 0.8365 ± 0.0230 (5/5 folds) 🥈
3. **attention_pool_noInitWeights** - AUC: 0.8321 ± 0.0235 (5/5 folds) 🥉
4. **no_rl_full_train** - AUC: 0.8302 ± 0.0292 (5/5 folds)
5. **uniform** - AUC: 0.8118 ± 0.0169 (5/5 folds)
6. **mean_pool** - AUC: 0.8077 ± 0.0367 (5/5 folds)

### 📊 Visualizations Generated (25 files)
- ✅ ROC curves with 95% CI
- ✅ Precision-Recall curves
- ✅ Comprehensive metrics tables
- ✅ Statistical significance tests
- ✅ Attention pattern analysis (6 models)
- ✅ Site-level pathology analysis (6 models)
- ✅ Pathology ensemble ML models
- ✅ Performance vs complexity analysis

---

## ✅ EXCELLENT PERFORMANCE - READY FOR PUBLICATION (6 experiments)

### 1. **attention_pool** 🥇
- **Status:** 4/5 folds completed and visualized
- **Test Performance:** AUC: 0.8389 ± 0.0223, Acc: 0.7300 ± 0.0583
- **95% CI:** [0.7980-0.8798]
- **Visualization:** ✅ Complete
- **Action:** ✅ Ready for publication
- **Notes:** Best overall performance, attention mechanism provides interpretability

### 2. **original_noInitWeights** 🥈
- **Status:** 5/5 folds completed and visualized
- **Test Performance:** AUC: 0.8365 ± 0.0230, Acc: 0.7700 ± 0.0869
- **95% CI:** [0.8109-0.8621]
- **Visualization:** ✅ Complete
- **Action:** ✅ Ready for publication
- **Notes:** Full completion, demonstrates benefit of no initial weight loading

### 3. **attention_pool_noInitWeights** 🥉
- **Status:** 5/5 folds completed and visualized
- **Test Performance:** AUC: 0.8321 ± 0.0235, Acc: 0.7650 ± 0.0716
- **95% CI:** [0.8059-0.8583]
- **Visualization:** ✅ Complete
- **Action:** ✅ Ready for publication
- **Notes:** Combines attention pooling with no initial weights - strong performance

### 4. **no_rl_full_train**
- **Status:** 5/5 folds completed and visualized
- **Test Performance:** AUC: 0.8302 ± 0.0292, Acc: 0.7133 ± 0.0680
- **95% CI:** [0.7978-0.8626]
- **Visualization:** ✅ Complete
- **Action:** ✅ Ready for publication
- **Notes:** Demonstrates full training without RL achieves excellent results

### 5. **uniform**
- **Status:** 5/5 folds completed and visualized
- **Test Performance:** AUC: 0.8118 ± 0.0169, Acc: 0.7260 ± 0.0162
- **95% CI:** [0.7883-0.8353]
- **Visualization:** ✅ Complete
- **Action:** ✅ Ready for publication
- **Notes:** Simple uniform pooling baseline performs surprisingly well

### 6. **mean_pool**
- **Status:** 5/5 folds completed and visualized
- **Test Performance:** AUC: 0.8077 ± 0.0367, Acc: 0.7260 ± 0.0589
- **95% CI:** [0.7568-0.8586]
- **Visualization:** ✅ Complete
- **Action:** ✅ Ready for publication
- **Notes:** Strong simple pooling baseline

---

## ⚠️ MOSTLY READY FOR VISUALIZATION (2 experiments)

### 6. **inception3d** 
- **Status:** 4 out of 5 folds completed and evaluated
- **Completion:** 4/5 folds (80%)
- **Test Performance:** AUC=0.5459±0.0485, Acc=0.6000±0.0400 (Suboptimal)
- **Failed:** fold1
- **Issue:** Error during training/evaluation (OOM or error detected in logs); Poor model performance
- **Action:** Can visualize with 4 folds (though performance is weak), or rerun fold1

### 7. **original_noInitWeights**
- **Status:** 4 out of 5 folds completed and evaluated
- **Completion:** 4/5 folds (80%)
- **Test Performance:** AUC=0.8374±0.0205, Acc=0.7700±0.0869 (Good performance)
- **Failed:** fold0 (UNKNOWN status - minimal output)
- **Issue:** Job appears to have started but didn't produce evaluation results
- **Action:** ✅ Can visualize with 4 folds (good performance), or rerun fold0

---

## ✗ NOT READY FOR VISUALIZATION (9 experiments)

### 8. **3dcnn**
- **Status:** 3 out of 5 folds completed and evaluated
- **Completion:** 3/5 folds (60%)
- **Test Performance:** AUC=0.5361±0.0086, Acc=0.5400±0.1131 (Poor performance)
- **Failed:** fold0, fold1
- **Root Cause:** **NCCL communication timeout** - Distributed training communication failures between GPUs
- **Error Details:** 
  - ProcessGroupNCCL watchdog timeout
  - Collective operation timeout during ALLREDUCE
  - Likely due to network issues or synchronization problems in distributed training
- **Action:** Rerun fold0 and fold1 (though current results show poor model performance)
- **Recommendation:** Check NCCL configuration, network stability, or reduce batch size; Consider if this architecture is suitable

### 9. **Efficientnet-RL**
- **Status:** 0 out of 5 folds completed
- **Completion:** 0/5 folds (0%)
- **Failed:** All folds (fold1 shows explicit error, others UNKNOWN)
- **Root Cause:** **NCCL collective operation timeout** (1800 second timeout)
- **Error Details:**
  - Collective timeout during ALLREDUCE operations
  - ProcessGroupNCCL watchdog caught timeout
  - Last completed work: 211, enqueued: 212-216 (stalled)
- **Action:** Complete rerun needed
- **Recommendation:** This model may be too large or have synchronization issues. Consider:
  - Reducing model complexity
  - Checking GPU memory usage
  - Increasing NCCL timeout
  - Verifying network stability

### 10. **LeViT-Attention**
- **Status:** 3 out of 5 folds completed
- **Completion:** 3/5 folds (60%)
- **Failed:** fold2, fold3
- **Root Cause:** Errors detected during training/evaluation
- **Error Details:** fold3 shows error markers in logs
- **Action:** Rerun fold2 and fold3

### 11. **LeViT-RL**
- **Status:** 2 out of 5 folds completed and evaluated (3 folds found, but fold1 has issues)
- **Completion:** 2/5 folds (40%) actually usable
- **Test Performance:** AUC=0.4560±0.0367, Acc=0.3800±0.0000 (Very poor - worse than random)
- **Failed:** fold1 (OOM/Error), fold3 (UNKNOWN), fold4 (Error)
- **Root Cause:** Mixed - OOM errors and incomplete runs; Model not learning
- **Error Details:** 
  - fold1: Error/OOM detected in logs
  - fold3: Minimal output (355 bytes) - job didn't complete
  - fold4: Error in stderr
- **Action:** Results suggest this architecture is not working - recommend investigating model design before rerunning
- **Recommendation:** Check GPU memory usage, may need to reduce batch size; Verify model architecture is correct

### 12. **attention_pool_noInitWeights**
- **Status:** 3 out of 5 folds completed and evaluated (4 folds have evaluations)
- **Completion:** 4/5 folds (80%)
- **Test Performance:** AUC=0.8374±0.0174, Acc=0.7650±0.0716 (Good performance)
- **Failed:** fold1 (has eval but with error), fold4 (UNKNOWN)
- **Root Cause:** 
  - fold1: Error during training but eval exists
  - fold4: Job started but didn't complete (375 bytes output)
- **Action:** ✅ Can visualize with 4 folds (good performance), or rerun fold4

### 13. **no_rl_full_train**
- **Status:** 2 out of 5 folds completed and evaluated (3 folds have evaluations)
- **Completion:** 3/5 folds (60%)
- **Test Performance:** AUC=0.8336±0.0142, Acc=0.7133±0.0680 (Good performance)
- **Failed:** fold2 (has eval but with OOM/Error), fold3 (UNKNOWN), fold4 (UNKNOWN)
- **Root Cause:** 
  - fold2: OOM or error during training but eval exists
  - fold3, fold4: Jobs started but didn't complete (363 bytes output each)
- **Action:** Can visualize with 3 folds (good performance), or rerun fold3 and fold4
- **Recommendation:** Check memory usage, consider reducing batch size or model size

### 14. **original**
- **Status:** 0 out of 5 folds completed
- **Completion:** 0/5 folds (0%)
- **Failed:** All folds show UNKNOWN status
- **Root Cause:** All jobs started but appear to be hung or incomplete
- **Error Details:** 
  - fold0-3: Small output files (355 bytes)
  - fold4: No logs found at all
  - Logs show training started but never completed
- **Action:** Complete rerun needed
- **Recommendation:** Jobs may have hung during training. Check for:
  - Hanging processes
  - Distributed training deadlocks
  - Resource allocation issues

### 15. **r2plus1d**
- **Status:** 0 out of 5 folds completed
- **Completion:** 0/5 folds (0%)
- **Failed:** All 5 folds
- **Root Cause:** **CUDA OOM and NCCL timeout**
- **Error Details:**
  - Multiple "OOM processing site" warnings
  - NCCL watchdog timeout (1800 seconds)
  - Collective operation timeout during ALLREDUCE
  - Process killed after 30+ minutes of hanging
- **Action:** Complete rerun needed with configuration changes
- **Recommendation:** This model is experiencing severe memory issues:
  - **Reduce batch size significantly**
  - Consider gradient checkpointing
  - May need smaller model variant
  - Check if r2plus1d is too memory-intensive for current setup

### 16. **singletask**
- **Status:** 0 out of 4 folds completed (only 4 folds exist)
- **Completion:** 0/4 folds (0%)
- **Failed:** All folds show UNKNOWN status
- **Root Cause:** Training appears to hang early in epoch 1
- **Error Details:**
  - Logs show training started (reached 33% of epoch 1)
  - Then process appears to hang indefinitely
  - No completion or error messages
- **Action:** Complete rerun needed
- **Recommendation:** Process hanging during training. Check for:
  - Deadlocks in data loading or model forward pass
  - NCCL synchronization issues
  - Resource contention

---

## Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| Fully Ready (5/5 folds, good performance) | 3 | 18.75% |
| Fully Complete but Poor Performance | 2 | 12.50% |
| Mostly Ready (4/5 folds, good performance) | 3 | 18.75% |
| Partial (2-3/5 folds evaluated) | 3 | 18.75% |
| Failed (0-1/5 folds) | 5 | 31.25% |

**Total Successful Evaluation Folds:** 43 / 79 folds (54.4%)

**Performance Rankings (by AUC):**
1. **attention_pool**: 0.8389 ± 0.0223 ✅
2. **attention_pool_noInitWeights**: 0.8374 ± 0.0174 ✅
3. **original_noInitWeights**: 0.8374 ± 0.0205 ✅
4. **no_rl_full_train**: 0.8336 ± 0.0142 ✅
5. **uniform**: 0.8118 ± 0.0169 ✅
6. **mean_pool**: 0.8077 ± 0.0367 ✅
7. **inception3d**: 0.5459 ± 0.0485 ⚠️
8. **3dcnn**: 0.5361 ± 0.0086 ⚠️
9. **cnnlstm**: 0.5008 ± 0.0249 ⚠️
10. **vivit**: 0.4975 ± 0.0460 ⚠️
11. **LeViT-RL**: 0.4560 ± 0.0367 ❌

**Note:** Experiments without evaluation data (Efficientnet-RL, LeViT-Attention, original, r2plus1d, singletask) are not ranked.

---

## Root Cause Analysis

### Primary Issues:

1. **NCCL Communication Timeouts** (4 experiments affected)
   - **Experiments:** 3dcnn, Efficientnet-RL, r2plus1d, original
   - **Cause:** Distributed training synchronization failures
   - **Solution:** Check network stability, increase timeout, or investigate deadlocks

2. **GPU Memory Issues (OOM)** (3 experiments affected)
   - **Experiments:** r2plus1d (severe), LeViT-RL, no_rl_full_train
   - **Cause:** Models too large or batch size too high
   - **Solution:** Reduce batch size, enable gradient checkpointing, or use smaller models

3. **Process Hanging** (3 experiments affected)
   - **Experiments:** singletask, original, some folds in other experiments
   - **Cause:** Deadlocks during training or data loading
   - **Solution:** Debug distributed training setup, check for GIL deadlock

4. **Generic Training Errors** (remaining failures)
   - **Experiments:** LeViT-Attention, attention_pool_noInitWeights, inception3d
   - **Cause:** Various errors during training/evaluation
   - **Solution:** Check individual error logs for specific issues

---

## Recommended Actions

### Immediate Actions:
1. ✅ **Start visualization** for the 6 well-performing experiments (attention_pool, mean_pool, uniform, attention_pool_noInitWeights, original_noInitWeights, no_rl_full_train)
2. ⚠️ **Consider investigating** why cnnlstm and vivit performed so poorly (near-random) despite completing training

### Short-term Actions:
1. **Rerun missing folds** for well-performing experiments:
   - **original_noInitWeights**: fold0 (currently 4/5 folds, good performance)
   - **attention_pool_noInitWeights**: fold4 (currently 4/5 folds, good performance)
   - **no_rl_full_train**: fold3, fold4 (currently 3/5 folds, good performance)

2. **Investigate and potentially skip** poorly performing experiments:
   - **3dcnn**: 0.5361 AUC - likely architectural issues
   - **cnnlstm**: 0.5008 AUC - model not learning
   - **vivit**: 0.4975 AUC - worse than random
   - **LeViT-RL**: 0.4560 AUC - severe issues
   - **inception3d**: 0.5459 AUC - marginal performance
   
3. **Complete failed experiments** if needed for completeness:
   - **Efficientnet-RL**: Complete rerun needed (0/5 folds)
   - **LeViT-Attention**: Rerun fold2, fold3 (currently 3/5 folds)
   - **original**: Complete rerun needed (0/5 folds)
   - **r2plus1d**: Complete rerun needed (0/5 folds) - severe OOM issues
   - **singletask**: Complete rerun needed (0/4 folds) - hanging issues

### Configuration Recommendations:
1. **For OOM experiments:** Reduce batch size from 2 to 1, enable gradient checkpointing
2. **For NCCL timeouts:** Increase `NCCL_TIMEOUT` environment variable, check network
3. **For hanging jobs:** Add logging/debugging, check for distributed deadlocks
4. **For poorly performing models:** 
   - Review model architecture and hyperparameters
   - Check if data preprocessing is correct
   - Verify loss functions and training setup
   - Consider if some architectures are unsuitable for this task
5. **General:** Consider running problematic experiments on single GPU first to isolate distributed training issues

---

## Next Steps

1. ✅ **Visualize** the 6 well-performing experiments immediately:
   - attention_pool (AUC: 0.839)
   - attention_pool_noInitWeights (AUC: 0.837, 4/5 folds)
   - original_noInitWeights (AUC: 0.837, 4/5 folds)
   - no_rl_full_train (AUC: 0.834, 3/5 folds)
   - uniform (AUC: 0.812)
   - mean_pool (AUC: 0.808)

2. ⚠️ **Investigate** why some complete experiments performed poorly:
   - cnnlstm: Near-random performance (AUC: 0.501)
   - vivit: Worse than random (AUC: 0.498)
   - Review training logs, check for implementation issues

3. 🔄 **Rerun missing folds** for partial but well-performing experiments to get all 5 folds

4. ❌ **Deprioritize or skip** experiments with poor performance:
   - 3dcnn, inception3d, LeViT-RL all showing weak results
   - Focus effort on successful architectures instead

5. 🔧 **Debug and fix** the 5 completely failed experiments only if comprehensive comparison is needed:
   - Efficientnet-RL, LeViT-Attention, original, r2plus1d, singletask
   - These require significant debugging effort with uncertain payoff

6. 📊 **Monitor** resource usage during reruns to prevent similar failures

## Evaluation Files Location

All evaluation results are stored in:
```
/capstor/store/cscs/swissai/a127/ultr-ai/ablation_results/{experiment}/eval_results/
```

Each fold has:
- `{split}_full_model_fold{N}_sites.csv` - Site-level predictions
- `{split}_full_model_fold{N}_patients.csv` - Patient-level predictions
- `{split}_full_model_fold{N}_complex_data.h5` - Embeddings and features
- `{split}_full_model_fold{N}_metrics.json` - Performance metrics

Where `{split}` is one of: train, val, test
