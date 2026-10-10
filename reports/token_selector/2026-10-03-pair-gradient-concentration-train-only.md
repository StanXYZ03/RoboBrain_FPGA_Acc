# Pair 梯度贡献集中度审计（train-only，2026-10-03-pair-gradient-concentration-train-only）

本轮只做**零更新、CPU-only、train-only** 的成对梯度贡献集中度审计：先用三 seed 初始 concat pair/base 比对 `r_med` 与冻结 `lambda_target_0p25=14.280210`，再逐 image pair 计算 `L_base_i`/`L_pair_i` 的 Wt/Wv/concat 梯度并**只保留标量**，报告 signed projection contribution、负贡献、top-share、ESS、Gini、rank 稳定性与相关性。不训练、不调参、不保存 `Wt/Wv`、不读 val/test 特征缓存、不加载模型、不用 GPU、不联网、不设阈值、不给 promotion 结论。

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
- script SHA256 `7981c4ab76518c5039afde9b38b016abd75b12a4d5eb18446d8fa3d030190607`
- 输入 NPZ aggregate SHA256 `33997394c4e148e3cfcd45eb0eb37fc97616cece8db3da5c731171d4b53d3951`（仅 train，318 个文件）

## 3. lambda 与 r_med 校验

- 冻结 `r_med=0.017506745782799503`；本审计三 seed 重算 `r_med=0.017506745782799503`；matches_frozen=True。
- 冻结 `lambda_target_0p25=14.280209646136903245`；本审计使用 `14.280209646136903245`；matches_frozen=True（正 lambda 只改变 pair/base 相对尺度）。
- signed contribution 和校验最大绝对误差 = 0.000002。

## 4. 逐 seed 梯度集中度

### seed 20261002

- r_seed(concat)=0.010005，L_base/L_pair/L_total mean=2.900621/0.049644/2.950266，aggregate pair/base(concat)=0.010005。

| param | pair norm concentration | signed pair contribution |
| --- | --- | --- |
| Wt | top1%=0.098808, top5%=0.349951, top10%=0.538494, top20%=0.737153, ESS=42.343062, Gini=0.677577 | sum=1.000001, |sum-expected|=0.000001, expected=1.000000, neg_count=61, neg_fraction=0.383648, neg_mass=0.299239, pos_sum=1.299240 |
| Wv | top1%=0.097215, top5%=0.353370, top10%=0.542899, top20%=0.737089, ESS=41.976588, Gini=0.678112 | sum=1.000001, |sum-expected|=0.000001, expected=1.000000, neg_count=63, neg_fraction=0.396226, neg_mass=0.317577, pos_sum=1.317577 |
| concat | top1%=0.097547, top5%=0.351683, top10%=0.539988, top20%=0.737139, ESS=42.188012, Gini=0.677772 | sum=1.000001, |sum-expected|=0.000001, expected=1.000000, neg_count=61, neg_fraction=0.383648, neg_mass=0.302686, pos_sum=1.302688 |

| param | pair norm mean | pair/base mean | cos(pair,agg) mean | cos(base,agg) mean | pair zero frac mean | pair nonfinite frac mean |
| --- | --- | --- | --- | --- | --- | --- |
| Wt | 0.386707 | 0.034284 | 0.071913 | 0.388535 | 0.012579 | 0.000000 |
| Wv | 0.386551 | 0.029783 | 0.064173 | 0.406642 | 0.012579 | 0.000000 |
| concat | 0.546961 | 0.031767 | 0.067907 | 0.398828 | 0.012579 | 0.000000 |

- Spearman（本 seed，concat）：JS vs pair norm=0.929321，JS vs signed contribution=0.393191，G vs pair norm=-0.038236，L_pair vs pair norm=0.914948，hinge margin vs pair norm=0.914947（hinge active n=157）。

### seed 20261003

- r_seed(concat)=0.017507，L_base/L_pair/L_total mean=2.710857/0.051894/2.762752，aggregate pair/base(concat)=0.017507。

| param | pair norm concentration | signed pair contribution |
| --- | --- | --- |
| Wt | top1%=0.104333, top5%=0.360875, top10%=0.547376, top20%=0.732406, ESS=40.956733, Gini=0.681387 | sum=1.000001, |sum-expected|=0.000001, expected=1.000000, neg_count=64, neg_fraction=0.402516, neg_mass=0.296482, pos_sum=1.296483 |
| Wv | top1%=0.101791, top5%=0.360359, top10%=0.549752, top20%=0.734467, ESS=40.917046, Gini=0.681999 | sum=1.000000, |sum-expected|=0.000000, expected=1.000000, neg_count=62, neg_fraction=0.389937, neg_mass=0.325698, pos_sum=1.325699 |
| concat | top1%=0.102973, top5%=0.360518, top10%=0.548547, top20%=0.733448, ESS=40.960775, Gini=0.681571 | sum=1.000002, |sum-expected|=0.000002, expected=1.000000, neg_count=62, neg_fraction=0.389937, neg_mass=0.308638, pos_sum=1.308640 |

| param | pair norm mean | pair/base mean | cos(pair,agg) mean | cos(base,agg) mean | pair zero frac mean | pair nonfinite frac mean |
| --- | --- | --- | --- | --- | --- | --- |
| Wt | 0.431490 | 0.036854 | 0.074949 | 0.293575 | 0.025157 | 0.000000 |
| Wv | 0.454509 | 0.037272 | 0.067489 | 0.273128 | 0.025157 | 0.000000 |
| concat | 0.626896 | 0.037049 | 0.070856 | 0.283288 | 0.025157 | 0.000000 |

- Spearman（本 seed，concat）：JS vs pair norm=0.924120，JS vs signed contribution=0.335016，G vs pair norm=0.001645，L_pair vs pair norm=0.895655，hinge margin vs pair norm=0.895648（hinge active n=155）。

### seed 20261004

- r_seed(concat)=0.022286，L_base/L_pair/L_total mean=2.697297/0.051756/2.749053，aggregate pair/base(concat)=0.022286。

| param | pair norm concentration | signed pair contribution |
| --- | --- | --- |
| Wt | top1%=0.102365, top5%=0.355115, top10%=0.547337, top20%=0.732884, ESS=41.123623, Gini=0.679493 | sum=1.000001, |sum-expected|=0.000001, expected=1.000000, neg_count=68, neg_fraction=0.427673, neg_mass=0.380521, pos_sum=1.380522 |
| Wv | top1%=0.101902, top5%=0.355641, top10%=0.543808, top20%=0.731771, ESS=41.415108, Gini=0.678243 | sum=1.000000, |sum-expected|=0.000000, expected=1.000000, neg_count=58, neg_fraction=0.364780, neg_mass=0.306493, pos_sum=1.306493 |
| concat | top1%=0.101444, top5%=0.355375, top10%=0.545641, top20%=0.732345, ESS=41.287924, Gini=0.678798 | sum=1.000002, |sum-expected|=0.000002, expected=1.000000, neg_count=66, neg_fraction=0.415094, neg_mass=0.334751, pos_sum=1.334753 |

| param | pair norm mean | pair/base mean | cos(pair,agg) mean | cos(base,agg) mean | pair zero frac mean | pair nonfinite frac mean |
| --- | --- | --- | --- | --- | --- | --- |
| Wt | 0.393943 | 0.037477 | 0.063087 | 0.209320 | 0.031447 | 0.000000 |
| Wv | 0.380414 | 0.033310 | 0.070125 | 0.206091 | 0.031447 | 0.000000 |
| concat | 0.547795 | 0.035268 | 0.066386 | 0.207495 | 0.031447 | 0.000000 |

- Spearman（本 seed，concat）：JS vs pair norm=0.930627，JS vs signed contribution=0.337427，G vs pair norm=-0.035373，L_pair vs pair norm=0.942756，hinge margin vs pair norm=0.942742（hinge active n=154）。

## 5. 三 seed 聚合（pooled）

### Wt

- pair norm 分布：top1%=0.086919, top5%=0.355518, top10%=0.539159, top20%=0.734543, ESS=123.976400, Gini=0.680063
- signed pair contribution：sum=3.000003, |sum-expected|=0.000003, expected=3.000000, neg_count=193, neg_fraction=0.404612, neg_mass=0.976242, pos_sum=3.976245
- 正贡献集中度：top1%=0.146601，top5%=0.439072，top10%=0.649802，top20%=0.816631，ESS=49.823158，Gini=0.767486
- base signed contribution：sum=3.000005, |sum-expected|=0.000005, expected=3.000000, neg_count=67, neg_fraction=0.140461, neg_mass=0.222494, pos_sum=3.222499

### Wv

- pair norm 分布：top1%=0.088621, top5%=0.357446, top10%=0.540536, top20%=0.734828, ESS=123.279103, Gini=0.680720
- signed pair contribution：sum=3.000001, |sum-expected|=0.000001, expected=3.000000, neg_count=183, neg_fraction=0.383648, neg_mass=0.949768, pos_sum=3.949769
- 正贡献集中度：top1%=0.141616，top5%=0.472479，top10%=0.677231，top20%=0.841595，ESS=48.373777，Gini=0.785753
- base signed contribution：sum=3.000002, |sum-expected|=0.000002, expected=3.000000, neg_count=63, neg_fraction=0.132075, neg_mass=0.202729, pos_sum=3.202731

### concat

- pair norm 分布：top1%=0.087464, top5%=0.356141, top10%=0.539848, top20%=0.734687, ESS=123.742059, Gini=0.680267
- signed pair contribution：sum=3.000005, |sum-expected|=0.000005, expected=3.000000, neg_count=189, neg_fraction=0.396226, neg_mass=0.946076, pos_sum=3.946081
- 正贡献集中度：top1%=0.144849，top5%=0.444115，top10%=0.660550，top20%=0.832844，ESS=48.937040，Gini=0.777647
- base signed contribution：sum=3.000006, |sum-expected|=0.000006, expected=3.000000, neg_count=63, neg_fraction=0.132075, neg_mass=0.206200, pos_sum=3.206205

## 6. 跨 seed 稳定性与相关性

- Wt 同图 pair norm 跨 seed pairwise Spearman = {'20261002_vs_20261003': 0.967774992203672, '20261002_vs_20261004': 0.974510736249662, '20261003_vs_20261004': 0.9456908069317979}，mean=0.962659（min=0.945691，max=0.974511，shared images=159）。
- Wv 同图 pair norm 跨 seed pairwise Spearman = {'20261002_vs_20261003': 0.9693139945689339, '20261002_vs_20261004': 0.9764766734398831, '20261003_vs_20261004': 0.9487987102913159}，mean=0.964863（min=0.948799，max=0.976477，shared images=159）。
- concat 同图 pair norm 跨 seed pairwise Spearman = {'20261002_vs_20261003': 0.9690706799369866, '20261002_vs_20261004': 0.9757541877906295, '20261003_vs_20261004': 0.9474835984470626}，mean=0.964103（min=0.947484，max=0.975754，shared images=159）。

- pooled Wt：JS vs pair norm=0.927549，JS vs signed contribution=0.361665，G vs pair norm=-0.023642，L_pair vs pair norm=0.915870，hinge margin vs pair norm=0.915864，active-only hinge margin vs pair norm=0.909771。
- pooled Wv：JS vs pair norm=0.926159，JS vs signed contribution=0.355050，G vs pair norm=-0.022930，L_pair vs pair norm=0.914207，hinge margin vs pair norm=0.914201，active-only hinge margin vs pair norm=0.907988。
- pooled concat：JS vs pair norm=0.927240，JS vs signed contribution=0.356084，G vs pair norm=-0.023203，L_pair vs pair norm=0.915456，hinge margin vs pair norm=0.915450，active-only hinge margin vs pair norm=0.909327。

## 7. Top 10 图像（仅 hash）

### seed 20261002

| rank | image_sha256 | G | JS | L_pair | hinge_active | pair norm concat | pair signed contribution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | cfe37e2cabeb42f4a3c6d20e7b9c3a50882193729550ea58bd8bac3133a10906 | 260 | 0.486214 | 0.425655 | true | 4.337597 | -0.015660 |
| 2 | ccfbf08efc80416f38a5b732d916b9f2a4b6436ae4a0398ccbc0d9dcc7db0aae | 300 | 0.411547 | 0.306640 | true | 4.145730 | 0.062923 |
| 3 | 5f7990e5edb8b8ab72faeef5ad6391976b24c2ab14d1c05d178fba70fadfd435 | 260 | 0.497356 | 0.476450 | true | 4.026680 | 0.102681 |
| 4 | 231e12c229c2f84c186085b320a6a8dcaa3d4763bbd09940204121379a5b3e11 | 300 | 0.235941 | 0.196337 | true | 3.827330 | 0.198232 |
| 5 | 5643aadb0049998a1f1e2a418af7be084c40d5ff5ac7eadf17b38cb5c61c0117 | 300 | 0.393590 | 0.381616 | true | 3.788874 | 0.108480 |
| 6 | 533fd5d3b52305fee0f1e96b258b41de137ec1de465ee5c0c3afaa599c6c8567 | 260 | 0.321647 | 0.280239 | true | 3.570955 | 0.090501 |
| 7 | 2d0e8b59bee673296d4ed71dea5345565b79cf73c0b56a20e09165850c9509df | 160 | 0.242987 | 0.274570 | true | 3.446763 | -0.086035 |
| 8 | 13ec6259276d140b0fb45d955cf497cfe59be9a8ae17a76330b24541069f9cab | 192 | 0.308571 | 0.327135 | true | 3.440819 | -0.023193 |
| 9 | f4714db3b155caf93afa783e108381c036c759c39c2836880dece65dd020cafe | 260 | 0.265168 | 0.306281 | true | 2.950006 | -0.064067 |
| 10 | 8d5a4b2e84e045a974113175429619a0a0118f6f0b95124d53057c733f4ff8d1 | 260 | 0.243381 | 0.315922 | true | 2.478354 | 0.039501 |

按 signed contribution 排序的 top 10 hash：231e12c229c2f84c186085b320a6a8dcaa3d4763bbd09940204121379a5b3e11, 5643aadb0049998a1f1e2a418af7be084c40d5ff5ac7eadf17b38cb5c61c0117, 5f7990e5edb8b8ab72faeef5ad6391976b24c2ab14d1c05d178fba70fadfd435, 6a6b5e10a1bb90c1a322cc39088a2e7174f30935e8fd4a9d6b0f9392ac2351e7, d353d745084e986af44b6bb9b2fc4cbf5f412ed913294ffc774ec3d83e49f87d, 533fd5d3b52305fee0f1e96b258b41de137ec1de465ee5c0c3afaa599c6c8567, ccfbf08efc80416f38a5b732d916b9f2a4b6436ae4a0398ccbc0d9dcc7db0aae, 980b3adbde3e49f6cc92f0611ea9ef8dde633715a7d9041bafa8048fb61c24de, 8d5a4b2e84e045a974113175429619a0a0118f6f0b95124d53057c733f4ff8d1, 5957b55666219c9181b00c4fe31c802cc2d33225f976d47656ef92f5f50f788a

### seed 20261003

| rank | image_sha256 | G | JS | L_pair | hinge_active | pair norm concat | pair signed contribution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ccfbf08efc80416f38a5b732d916b9f2a4b6436ae4a0398ccbc0d9dcc7db0aae | 300 | 0.411547 | 0.361810 | true | 5.148172 | 0.074156 |
| 2 | cfe37e2cabeb42f4a3c6d20e7b9c3a50882193729550ea58bd8bac3133a10906 | 260 | 0.486214 | 0.363652 | true | 5.115768 | -0.020485 |
| 3 | 5f7990e5edb8b8ab72faeef5ad6391976b24c2ab14d1c05d178fba70fadfd435 | 260 | 0.497356 | 0.576483 | true | 4.669818 | 0.118979 |
| 4 | 5643aadb0049998a1f1e2a418af7be084c40d5ff5ac7eadf17b38cb5c61c0117 | 300 | 0.393590 | 0.506706 | true | 4.556527 | 0.128933 |
| 5 | 231e12c229c2f84c186085b320a6a8dcaa3d4763bbd09940204121379a5b3e11 | 300 | 0.235941 | 0.274559 | true | 4.214383 | 0.181795 |
| 6 | 533fd5d3b52305fee0f1e96b258b41de137ec1de465ee5c0c3afaa599c6c8567 | 260 | 0.321647 | 0.400747 | true | 4.142965 | 0.098883 |
| 7 | 2d0e8b59bee673296d4ed71dea5345565b79cf73c0b56a20e09165850c9509df | 160 | 0.242987 | 0.217251 | true | 4.131655 | -0.080091 |
| 8 | 13ec6259276d140b0fb45d955cf497cfe59be9a8ae17a76330b24541069f9cab | 192 | 0.308571 | 0.309299 | true | 3.955861 | -0.028707 |
| 9 | f4714db3b155caf93afa783e108381c036c759c39c2836880dece65dd020cafe | 260 | 0.265168 | 0.316339 | true | 3.348229 | -0.062655 |
| 10 | 8d5a4b2e84e045a974113175429619a0a0118f6f0b95124d53057c733f4ff8d1 | 260 | 0.243381 | 0.191752 | true | 2.869882 | 0.041884 |

按 signed contribution 排序的 top 10 hash：231e12c229c2f84c186085b320a6a8dcaa3d4763bbd09940204121379a5b3e11, 5643aadb0049998a1f1e2a418af7be084c40d5ff5ac7eadf17b38cb5c61c0117, 5f7990e5edb8b8ab72faeef5ad6391976b24c2ab14d1c05d178fba70fadfd435, 533fd5d3b52305fee0f1e96b258b41de137ec1de465ee5c0c3afaa599c6c8567, d353d745084e986af44b6bb9b2fc4cbf5f412ed913294ffc774ec3d83e49f87d, 6a6b5e10a1bb90c1a322cc39088a2e7174f30935e8fd4a9d6b0f9392ac2351e7, ccfbf08efc80416f38a5b732d916b9f2a4b6436ae4a0398ccbc0d9dcc7db0aae, c76dfdb296725a9260eb9855a1aede9df5d6dd177ad9cfde06458728f9f22607, 8d5a4b2e84e045a974113175429619a0a0118f6f0b95124d53057c733f4ff8d1, 980b3adbde3e49f6cc92f0611ea9ef8dde633715a7d9041bafa8048fb61c24de

### seed 20261004

| rank | image_sha256 | G | JS | L_pair | hinge_active | pair norm concat | pair signed contribution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5f7990e5edb8b8ab72faeef5ad6391976b24c2ab14d1c05d178fba70fadfd435 | 260 | 0.497356 | 0.516227 | true | 4.452456 | 0.137748 |
| 2 | cfe37e2cabeb42f4a3c6d20e7b9c3a50882193729550ea58bd8bac3133a10906 | 260 | 0.486214 | 0.442657 | true | 4.383297 | 0.001715 |
| 3 | ccfbf08efc80416f38a5b732d916b9f2a4b6436ae4a0398ccbc0d9dcc7db0aae | 300 | 0.411547 | 0.311557 | true | 4.298377 | 0.083359 |
| 4 | 231e12c229c2f84c186085b320a6a8dcaa3d4763bbd09940204121379a5b3e11 | 300 | 0.235941 | 0.235711 | true | 3.818099 | 0.191560 |
| 5 | 5643aadb0049998a1f1e2a418af7be084c40d5ff5ac7eadf17b38cb5c61c0117 | 300 | 0.393590 | 0.425458 | true | 3.772339 | 0.114150 |
| 6 | 13ec6259276d140b0fb45d955cf497cfe59be9a8ae17a76330b24541069f9cab | 192 | 0.308571 | 0.447597 | true | 3.524579 | -0.028855 |
| 7 | 533fd5d3b52305fee0f1e96b258b41de137ec1de465ee5c0c3afaa599c6c8567 | 260 | 0.321647 | 0.347126 | true | 3.373652 | 0.085750 |
| 8 | 2d0e8b59bee673296d4ed71dea5345565b79cf73c0b56a20e09165850c9509df | 160 | 0.242987 | 0.225642 | true | 3.330136 | -0.082074 |
| 9 | f4714db3b155caf93afa783e108381c036c759c39c2836880dece65dd020cafe | 260 | 0.265168 | 0.257006 | true | 3.001198 | -0.082552 |
| 10 | 8d5a4b2e84e045a974113175429619a0a0118f6f0b95124d53057c733f4ff8d1 | 260 | 0.243381 | 0.177827 | true | 2.689861 | 0.031774 |

按 signed contribution 排序的 top 10 hash：231e12c229c2f84c186085b320a6a8dcaa3d4763bbd09940204121379a5b3e11, 5f7990e5edb8b8ab72faeef5ad6391976b24c2ab14d1c05d178fba70fadfd435, 5643aadb0049998a1f1e2a418af7be084c40d5ff5ac7eadf17b38cb5c61c0117, d353d745084e986af44b6bb9b2fc4cbf5f412ed913294ffc774ec3d83e49f87d, 6a6b5e10a1bb90c1a322cc39088a2e7174f30935e8fd4a9d6b0f9392ac2351e7, 533fd5d3b52305fee0f1e96b258b41de137ec1de465ee5c0c3afaa599c6c8567, ccfbf08efc80416f38a5b732d916b9f2a4b6436ae4a0398ccbc0d9dcc7db0aae, c76dfdb296725a9260eb9855a1aede9df5d6dd177ad9cfde06458728f9f22607, 980b3adbde3e49f6cc92f0611ea9ef8dde633715a7d9041bafa8048fb61c24de, 8d5a4b2e84e045a974113175429619a0a0118f6f0b95124d53057c733f4ff8d1

## 8. 统计解释与限制

- 梯度范数之和不是 aggregate 梯度：top-k norm share 只是模长分布，不能当作向量贡献来读；只有 signed projection contribution 才满足和约为 1。
- signed projection contribution 可为负，负贡献已被单独报告；正贡献的 share/ESS/Gini 只在 positive 子集上定义，不能掩盖负质量。
- lambda 为正时只改变 pair 与 base 的相对尺度，不改变 pair-only 的集中度、rank 或 cosine；本审计不据此宣布任何 lambda 为 winner。
- 不设通过/失败阈值，不宣布 promotion/no-promotion，不修改 DRAFT，也不授权重训。
- 所有数值都是 internal train-only 初始梯度诊断，不是 CV-Bench 任务精度，不能从初始梯度推断收敛、验证集改进或任务精度。
- 本审计只描述初始点的每图梯度贡献分布，不改变 DRAFT 的任何字段，也不构成重训授权。

