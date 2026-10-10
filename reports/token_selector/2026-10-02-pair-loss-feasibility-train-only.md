# Pair-loss 数学与初始化梯度可行性审计（train-only，2026-10-02-pair-loss-feasibility-train-only）

本轮只做**零更新、CPU-only、train-only** 的数学与梯度可行性审计：判断草案自适应 margin 在 teacher optimum 是否可满足、初始化梯度是否存在且有限、以及 KL 与 pair 梯度同向/冲突程度。不训练、不调参、不保存新 `Wt/Wv`、不读 val/test 特征缓存、不加载模型、不用 GPU、不联网、不设阈值、不声称重训会成功。

## 1. 范围与硬性隔离

- split 仅由 index metadata 用冻结 `split_sample_ids` 计算：train/val/test = 318/76/78 条记录。
- 只打开 train NPZ：opened_npz_count=318，val_test_npz_opened=[]，opened_only_train_npz=True。
- train 记录/图片 = 318/159（期望 318/159，matches_expected=True）；status=COMPUTED。
- draft 合同校验（只读）：全部通过 = True；draft 状态保持 `DRAFT_REQUIRES_PI_APPROVAL`。

## 2. 输入与 provenance

- python 3.10.20 / numpy 2.2.6 / torch 2.5.1+cu124（cuda_available=False）
- seeds=[20261002, 20261003, 20261004]，student temperature=0.1，init_std=0.02，score_dim=128。
- 训练 config SHA256 `44399c8e6fb8f333a150a6016faa00d6ae70524b5fcd030bcabf4e5fb07ab28b`
- index SHA256 `02a50868a84675418e9bd9e5fdd0bb55f522ec5ce0ce1d6da7c69f85d2aeee26`
- draft config SHA256 `787f8f401bcb10b5fe8007a6a0f5b415858d893cb2797b3c1797d5f1fa785772`
- script SHA256 `2913e43fadef1397f33a6d2f38ea093727ce2fc129378c330c0698e8c5a73dea`
- 输入 NPZ aggregate SHA256 `33997394c4e148e3cfcd45eb0eb37fc97616cece8db3da5c731171d4b53d3951`（仅 train，318 个文件）

## 3. Teacher optimum margin 可行性（159 image pairs）

teacher target 用 cached `teacher_soft`；`JS=JS_nats(y_g,y_p)`；teacher-optimal 时 `L_base=0`，`D_swap=0.5(KL(y_g||y_p)+KL(y_p||y_g))=symmetric_KL`，`slack=D_swap-JS`；`L_pair=relu(JS+L_base-D_swap)` 在 teacher optimum 归零当且仅当 slack≥0。

- slack：{'count': '159', 'mean': '0.334143', 'median': '0.041600', 'std': '0.741316', 'p5': '0.003287', 'p25': '0.014292', 'p75': '0.217549', 'p95': '1.839294', 'min': '0.000549', 'max': '4.538504'}
- slack≥0 比例 = **1.000000**；slack<0 图像数 = **0**（group_counts=[]）。
- `D_swap/JS` 比值（JS>0，n=159）：{'count': '159', 'mean': '5.593916', 'median': '5.203301', 'std': '1.310464', 'p5': '4.329114', 'p25': '4.673726', 'p75': '5.958517', 'p95': '8.462220', 'min': '4.114361', 'max': '10.334379'}
- min slack ≥ 0 = True；min ratio ≥ 2 = True。
- 不等式依据：JS(p,q) = 0.5 KL(p||m) + 0.5 KL(q||m) with m=(p+q)/2; convexity of KL in its second argument gives KL(p||m) <= 0.5 KL(p||q) and KL(q||m) <= 0.5 KL(q||p), hence JS <= 0.5(KL(p||q)+KL(q||p)) = symmetric_KL and the adaptive-margin slack D_swap - JS is non-negative at the teacher optimum.

**数学解释（仅此）**：在 teacher optimum，`L_base=0` 且 hinge 归零，因此自适应 margin **在 teacher optimum 是可满足的**；这不代表可训练成功。

## 4. 初始化前向/梯度审计（zero-update）

固定 seeds 20261002/20261003/20261004；scorer 严格复用冻结 `prompt_mean_vector`；`student distribution = softmax(score/0.1)`；只对 train-image mean loss 调 `autograd.grad`，不更新参数。

### 4.1 逐 seed

| seed | L_base mean | L_pair mean | L_total mean | hinge active frac | own-better frac | query cos mean | exact query eq frac | exact score eq frac | score finite frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20261002 | 2.900621 | 0.049644 | 2.950266 | 0.987421 | 0.559748 | 0.965567 | 0.000000 | 0.000000 | 1.000000 |
| 20261003 | 2.710857 | 0.051894 | 2.762752 | 0.974843 | 0.433962 | 0.954835 | 0.000000 | 0.000000 | 1.000000 |
| 20261004 | 2.697297 | 0.051756 | 2.749053 | 0.968553 | 0.402516 | 0.963547 | 0.000000 | 0.000000 | 1.000000 |

### 4.2 梯度范数 / 比值 / cosine

| seed | param | base norm | pair norm | total norm | pair/base | cos(base,pair) | finite | nonzeros |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20261002 | Wt | 6.249533 | 0.067958 | 6.251148 | 0.010874 | 0.018372 | true | b=1/p=1/t=1 |
| 20261002 | Wv | 7.494977 | 0.070098 | 7.494972 | 0.009353 | -0.004775 | true | b=1/p=1/t=1 |
| 20261002 | concat | 9.758619 | 0.097632 | 9.759663 | 0.010005 | 0.005557 | true | b=1/p=1/t=1 |
| 20261003 | Wt | 4.703732 | 0.082053 | 4.709699 | 0.017444 | 0.064096 | true | b=1/p=1/t=1 |
| 20261003 | Wv | 4.598096 | 0.080797 | 4.599782 | 0.017572 | 0.012103 | true | b=1/p=1/t=1 |
| 20261003 | concat | 6.577796 | 0.115156 | 6.583240 | 0.017507 | 0.038595 | true | b=1/p=1/t=1 |
| 20261004 | Wt | 3.082652 | 0.073503 | 3.084077 | 0.023844 | 0.007465 | true | b=1/p=1/t=1 |
| 20261004 | Wv | 3.288190 | 0.068464 | 3.289225 | 0.020821 | 0.004762 | true | b=1/p=1/t=1 |
| 20261004 | concat | 4.507190 | 0.100449 | 4.508925 | 0.022286 | 0.006104 | true | b=1/p=1/t=1 |

### 4.3 三 seed aggregate

- pooled L_base/L_pair/L_total mean = 2.769592/0.051098/2.820690；pooled p95 = 2.900622/0.051894/2.950266。
- hinge active image fraction（seed 均值）= 0.976939；own-better-than-swap fraction = 0.465409；exact query equality = 0.000000；exact score equality = 0.000000；score finite = 1.000000。
- Wt：base norm mean=4.678639，pair norm mean=0.074505，pair/base mean=0.017387，cos(base,pair) mean=0.029978（[0.007465, 0.064096]），pair 非零 seed 数=3/3，all_finite=True。
- Wv：base norm mean=5.127088，pair norm mean=0.073120，pair/base mean=0.015915，cos(base,pair) mean=0.004030（[-0.004775, 0.012103]），pair 非零 seed 数=3/3，all_finite=True。
- concat：base norm mean=6.947868，pair norm mean=0.104412，pair/base mean=0.016599，cos(base,pair) mean=0.016752（[0.005557, 0.038595]），pair 非零 seed 数=3/3，all_finite=True。

## 5. 解释与限制

- 本审计只回答：teacher optimum 下 margin 是否可满足、初始化梯度是否存在且有限、KL 与 pair 梯度同向/冲突程度。
- `slack≥0` 与该结论都是**数学可行性**，不是重训成功的证据；`gradient nonzero` 不是准确率。
- 未自创通过阈值，未据此声称任何方法会成功；草案公式若有歧义只在本报告指出，不改草案。
- 所有数值是 internal teacher/math 诊断，不是 CV-Bench 任务精度，也不代表 student 可学。

