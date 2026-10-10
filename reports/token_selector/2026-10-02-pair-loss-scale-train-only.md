# Pair-loss 尺度稳定性审计（train-only，2026-10-02-pair-loss-scale-train-only）

本轮只做**零更新、CPU-only、train-only** 的 pair-loss 尺度稳定性审计：先用三 seed 初始 concat pair/base 梯度范数比求 `r_med`，再对**预先固定**的四个 `lambda` 候选逐一实际调用 `autograd.grad`，报告 scaled pair/base、total/base 比值与 base/pair/total 梯度 cosine。不训练、不调参、不保存 `Wt/Wv`、不读 val/test 特征缓存、不加载模型、不用 GPU、不联网、不设阈值、不给 promotion 结论、不宣布 winner。

## 1. 范围与硬性隔离

- split 仅由 index metadata 用冻结 `split_sample_ids` 计算：train/val/test = 318/76/78 条记录。
- 只打开 train NPZ：opened_npz_count=318，val_test_npz_opened=[]，opened_only_train_npz=True。
- train 记录/图片 = 318/159（期望 318/159，matches_expected=True）；status=COMPUTED。
- draft 合同校验（只读）：全部通过 = True；draft 状态保持 `DRAFT_REQUIRES_PI_APPROVAL`，本审计不修改 DRAFT。

## 2. 输入与 provenance

- python 3.10.20 / numpy 2.2.6 / torch 2.5.1+cu124（cuda_available=False）
- seeds=[20261002, 20261003, 20261004]，student temperature=0.1，init_std=0.02，score_dim=128。
- 训练 config SHA256 `44399c8e6fb8f333a150a6016faa00d6ae70524b5fcd030bcabf4e5fb07ab28b`
- index SHA256 `02a50868a84675418e9bd9e5fdd0bb55f522ec5ce0ce1d6da7c69f85d2aeee26`
- draft config SHA256 `787f8f401bcb10b5fe8007a6a0f5b415858d893cb2797b3c1797d5f1fa785772`（只读）
- script SHA256 `b1b80810b340fc94cad6c3ba9ac237c69d775b681d73b2cc5a498e0f482853e9`
- 输入 NPZ aggregate SHA256 `33997394c4e148e3cfcd45eb0eb37fc97616cece8db3da5c731171d4b53d3951`（仅 train，318 个文件）

## 3. r_med 与预先固定的候选尺度

每个 seed 先算初始 concat `r_seed = ||grad(L_pair)|| / ||grad(L_base)||`，再取三 seed 中位数 `r_med`；四个候选 `lambda` 在任何 scaled 梯度数值之前固定，不新增其它 lambda。

- seed 20261002: r_seed(concat) = 0.010005
- seed 20261003: r_seed(concat) = 0.017507
- seed 20261004: r_seed(concat) = 0.022286
- **r_med (concat median) = 0.017507**

| candidate | lambda | target pair/base ratio |
| --- | --- | --- |
| lambda_current | 1.000000 | N/A (current) |
| lambda_target_0p1 | 5.712084 | 0.100000 |
| lambda_target_0p25 | 14.280210 | 0.250000 |
| lambda_target_1p0 | 57.120839 | 1.000000 |

## 4. 逐 seed / 逐 lambda 梯度诊断

### seed 20261002

| lambda | lambda value | scaled_pair/base (Wt) | (Wv) | (concat) | total/base (Wt) | (Wv) | (concat) | cos(total,base) | cos(total,pair) | cos(base,pair) | scaled_pair_loss/base_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lambda_current | 1.000000 | 0.010874 | 0.009353 | 0.010005 | 1.000258 | 0.999999 | 1.000107 | 0.999964 | 0.015560 | 0.005557 | 0.017115 |
| lambda_target_0p1 | 5.712084 | 0.062114 | 0.053423 | 0.057147 | 1.003065 | 1.001172 | 1.001950 | 0.998386 | 0.062583 | 0.005557 | 0.097762 |
| lambda_target_0p25 | 14.280210 | 0.155285 | 0.133557 | 0.142869 | 1.014800 | 1.008248 | 1.010942 | 0.989978 | 0.146820 | 0.005557 | 0.244406 |
| lambda_target_1p0 | 57.120839 | 0.621140 | 0.534228 | 0.571475 | 1.186863 | 1.131502 | 1.154530 | 0.868917 | 0.499803 | 0.005557 | 0.977622 |

### seed 20261003

| lambda | lambda value | scaled_pair/base (Wt) | (Wv) | (concat) | total/base (Wt) | (Wv) | (concat) | cos(total,base) | cos(total,pair) | cos(base,pair) | scaled_pair_loss/base_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lambda_current | 1.000000 | 0.017444 | 0.017572 | 0.017507 | 1.001268 | 1.000366 | 1.000828 | 0.999858 | 0.056056 | 0.038595 | 0.019143 |
| lambda_target_0p1 | 5.712084 | 0.099643 | 0.100372 | 0.100000 | 1.011287 | 1.006232 | 1.008820 | 0.995093 | 0.137385 | 0.038595 | 0.109347 |
| lambda_target_0p25 | 14.280210 | 0.249108 | 0.250930 | 0.250000 | 1.045939 | 1.033943 | 1.040095 | 0.970737 | 0.277473 | 0.038595 | 0.273368 |
| lambda_target_1p0 | 57.120839 | 0.996432 | 1.003721 | 1.000001 | 1.456232 | 1.425397 | 1.441246 | 0.720630 | 0.720631 | 0.038595 | 1.093471 |

### seed 20261004

| lambda | lambda value | scaled_pair/base (Wt) | (Wv) | (concat) | total/base (Wt) | (Wv) | (concat) | cos(total,base) | cos(total,pair) | cos(base,pair) | scaled_pair_loss/base_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lambda_current | 1.000000 | 0.023844 | 0.020821 | 0.022286 | 1.000463 | 1.000315 | 1.000385 | 0.999768 | 0.028380 | 0.006104 | 0.019188 |
| lambda_target_0p1 | 5.712084 | 0.136199 | 0.118933 | 0.127301 | 1.010240 | 1.007609 | 1.008841 | 0.992023 | 0.132238 | 0.006104 | 0.109604 |
| lambda_target_0p25 | 14.280210 | 0.340497 | 0.297331 | 0.318253 | 1.058786 | 1.044624 | 1.051273 | 0.953091 | 0.308541 | 0.006104 | 0.274010 |
| lambda_target_1p0 | 57.120839 | 1.361988 | 1.189325 | 1.273012 | 1.695688 | 1.557506 | 1.623612 | 0.620706 | 0.787831 | 0.006104 | 1.096041 |

## 5. 跨 seed 聚合（mean/median/min/max/CV）

### lambda_current（lambda = 1.000000，target = N/A (current)）

| param | scaled_pair/base mean | median | min | max | CV | total/base mean | median | min | max | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wt | 0.017387 | 0.017444 | 0.010874 | 0.023844 | 0.372976 | 1.000663 | 1.000463 | 1.000258 | 1.001268 | 0.000534 |
| Wv | 0.015915 | 0.017572 | 0.009353 | 0.020821 | 0.371409 | 1.000227 | 1.000315 | 0.999999 | 1.000366 | 0.000199 |
| concat | 0.016599 | 0.017507 | 0.010005 | 0.022286 | 0.372964 | 1.000440 | 1.000385 | 1.000107 | 1.000828 | 0.000363 |

- concat cos(total,base) mean=0.999863，cos(total,pair) mean=0.033332，cos(base,pair) mean=0.016752；scaled_pair_loss/base_loss mean=0.018482；all_finite=True，all_nonzero=True。

### lambda_target_0p1（lambda = 5.712084，target = 0.100000）

| param | scaled_pair/base mean | median | min | max | CV | total/base mean | median | min | max | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wt | 0.099319 | 0.099643 | 0.062114 | 0.136199 | 0.372976 | 1.008197 | 1.010240 | 1.003065 | 1.011287 | 0.004439 |
| Wv | 0.090909 | 0.100372 | 0.053423 | 0.118933 | 0.371409 | 1.005004 | 1.006232 | 1.001172 | 1.007609 | 0.003373 |
| concat | 0.094816 | 0.100000 | 0.057147 | 0.127301 | 0.372963 | 1.006537 | 1.008820 | 1.001950 | 1.008841 | 0.003947 |

- concat cos(total,base) mean=0.995167，cos(total,pair) mean=0.110735，cos(base,pair) mean=0.016752；scaled_pair_loss/base_loss mean=0.105571；all_finite=True，all_nonzero=True。

### lambda_target_0p25（lambda = 14.280210，target = 0.250000）

| param | scaled_pair/base mean | median | min | max | CV | total/base mean | median | min | max | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wt | 0.248297 | 0.249108 | 0.155285 | 0.340497 | 0.372976 | 1.039842 | 1.045939 | 1.014800 | 1.058786 | 0.021751 |
| Wv | 0.227273 | 0.250930 | 0.133557 | 0.297331 | 0.371409 | 1.028938 | 1.033943 | 1.008248 | 1.044624 | 0.018171 |
| concat | 0.237041 | 0.250000 | 0.142869 | 0.318253 | 0.372963 | 1.034103 | 1.040095 | 1.010942 | 1.051273 | 0.020136 |

- concat cos(total,base) mean=0.971269，cos(total,pair) mean=0.244278，cos(base,pair) mean=0.016752；scaled_pair_loss/base_loss mean=0.263928；all_finite=True，all_nonzero=True。

### lambda_target_1p0（lambda = 57.120839，target = 1.000000）

| param | scaled_pair/base mean | median | min | max | CV | total/base mean | median | min | max | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wt | 0.993187 | 0.996432 | 0.621140 | 1.361988 | 0.372976 | 1.446261 | 1.456232 | 1.186863 | 1.695688 | 0.176012 |
| Wv | 0.909091 | 1.003721 | 0.534228 | 1.189325 | 0.371409 | 1.371468 | 1.425397 | 1.131502 | 1.557506 | 0.158999 |
| concat | 0.948163 | 1.000001 | 0.571475 | 1.273012 | 0.372963 | 1.406463 | 1.441246 | 1.154530 | 1.623612 | 0.168129 |

- concat cos(total,base) mean=0.736751，cos(total,pair) mean=0.669422，cos(base,pair) mean=0.016752；scaled_pair_loss/base_loss mean=1.055711；all_finite=True，all_nonzero=True。

## 6. 解释与限制

- lambda=1 的 pair 梯度约为 base 的 1–2%（初始、concat 实测），不足以说明该成对监督会产生可见训练影响。
- 0.1 / 0.25 / 1.0 只是预先固定的诊断目标 ratio，不是文献标准，也不是精度门槛。
- 不能从零更新初始梯度推断收敛、验证集改进或任务精度。
- 本审计不修改 DRAFT、不训练、不给出 promotion / no-promotion 结论，也不授权重训。
- 所有数值都是 internal teacher/math 初始梯度诊断，不是 CV-Bench 任务精度。
- 本审计只描述初始点的梯度尺度与方向，不改变 DRAFT 的任何字段，也不构成重训授权。

