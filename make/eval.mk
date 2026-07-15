# RAG evaluation, prompt-system, benchmark, and pipeline targets.
##@ Evaluation and Pipelines

include $(PROJECT_ROOT)/make/eval/rag.mk
include $(PROJECT_ROOT)/make/eval/finetune.mk
include $(PROJECT_ROOT)/make/eval/workflows.mk
include $(PROJECT_ROOT)/make/eval/prompt-system.mk
include $(PROJECT_ROOT)/make/eval/security-agentic.mk
include $(PROJECT_ROOT)/make/eval/knowledge-cutoff.mk
include $(PROJECT_ROOT)/make/eval/categories-platform.mk
