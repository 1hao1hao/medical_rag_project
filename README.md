# MedAgent-RAGBench

This repository is being refactored from script-based medical RAG demos into a modular project for interview demos and benchmarking.

## Current Status

- Legacy scripts `01_xxx.py` to `07_xxx.py` are preserved as-is.
- New modular project skeleton is created under `src/medagent/`.
- Business logic migration is intentionally not done in this step.

## Target Modules

- `medagent.config`: runtime profiles and environment loading
- `medagent.ingestion`: document loading and chunking
- `medagent.retrieval`: dense/hybrid retrieval and rerank
- `medagent.generation`: local/api model adapters
- `medagent.agent`: query analyzer, rewrite, evidence evaluator, citation guard
- `medagent.evaluation`: evaluation harness and reports
- `medagent.serving`: CLI/service entrypoints
- `medagent.utils`: common helpers

## Project Structure

```text
medical_rag_project/
  01_build_vector_db.py
  02_retrieval_and_rerank.py
  03_rag_pipeline.py
  03_rag_pipeline_api.py
  04_rag_evaluation.py
  05_build_lora_dataset.py
  06_train_lora.py
  07_rag_with_lora.py
  configs/
  scripts/
  tests/
  outputs/
  logs/
  src/
    medagent/
      config/
      ingestion/
      retrieval/
      generation/
      agent/
      evaluation/
      serving/
      utils/
```

## Compile Check

```bash
python -m compileall src scripts
```

## 注释规范

- 项目内代码注释统一使用中文。
- 面试重点代码块（例如检索重排、提示词构建、LoRA 注入、评测关键流程）必须添加注释。
