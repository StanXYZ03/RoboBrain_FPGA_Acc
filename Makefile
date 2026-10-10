PYTHON ?= /home/zhaoxiangyu/conda-envs/robobrain25/bin/python

.PHONY: help check test eval-test analyze spec roofline deploy-test llama-host llama-arm64 model-download model-download-python model-verify model-smoke ram-linux ram-linux-sim

MODEL_DIR ?= deployment/models

help:
	@echo "make check     顶层项目与工具链自检"
	@echo "make test      运行 FPGA 分析单元测试"
	@echo "make eval-test 运行空间评测指标单元测试"
	@echo "make analyze   分析模型结构（不加载权重）"
	@echo "make spec      导出 FPGA 软件规格"
	@echo "make roofline  生成 ZCU102 Roofline 估算"
	@echo "make deploy-test 运行只读部署单元测试和已有 testbench"
	@echo "make llama-host 构建服务器 CPU llama.cpp"
	@echo "make llama-arm64 交叉编译 Cortex-A53 llama.cpp"
	@echo "make model-download 下载并验证固定 Q8_0 GGUF（约 5.4GB）"
	@echo "make model-smoke 在服务器 CPU 上进行确定性文本冒烟测试"
	@echo "make ram-linux 构建 ZCU102 16GB RAM Linux/initramfs"
	@echo "make ram-linux-sim 在 QEMU 中启动真实 ARM64 Image/initramfs 自检"

check:
	$(PYTHON) scripts/check_project.py

test:
	$(PYTHON) -m unittest discover -s fpga_accel/tests -v

eval-test:
	$(PYTHON) -m unittest discover -s evaluation/tests -v

analyze:
	$(PYTHON) fpga_accel/scripts/analyze_model_structure.py

spec:
	$(PYTHON) fpga_accel/scripts/export_fpga_spec.py

roofline:
	$(PYTHON) fpga_accel/scripts/estimate_fpga_roofline.py --lanes 256,512,1024,1536,2048 --bandwidth-gbps 2,4,6,8

deploy-test:
	@for script in deployment/board/*.sh deployment/scripts/*.sh; do sh -n "$$script" || exit 1; done
	python3 -m unittest discover -s deployment/tests -v
	$(MAKE) -C fpga_accel/ddr_test sim
	$(MAKE) -C fpga_accel/eth_test sim

llama-host:
	sh deployment/scripts/build_llama_host.sh

llama-arm64:
	sh deployment/scripts/build_llama_arm64.sh

model-download:
	MODEL_DIR=$(MODEL_DIR) sh deployment/scripts/download_models_dns_safe.sh

model-download-python:
	python3 deployment/scripts/model_gate.py download --directory $(MODEL_DIR)

model-verify:
	python3 deployment/scripts/model_gate.py verify --directory $(MODEL_DIR)

model-smoke:
	MODEL_DIR=$(MODEL_DIR) sh deployment/scripts/host_model_smoke.sh

ram-linux:
	sh deployment/scripts/build_ram_linux.sh

ram-linux-sim: ram-linux
	sh deployment/scripts/run_ram_linux_qemu_smoke.sh
