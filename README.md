# vllm-notes

vLLM 내부 구조 학습 노트. 엔진, 스케줄러, KV 캐시, 분산 서빙 등 고성능 LLM 인퍼런스 시스템의 동작 원리를 정리합니다.

- [용어집](glossary.md) — 아래 문서들에서 반복 등장하는 공통 용어 정리 (Latency 지표, Prefill/Decode, KV Cache, PagedAttention, TP/PP/DP/EP 등)
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](inside-vllm.md)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](paged-attention.md) — vLLM의 기반 논문 (SOSP '23)
- [온프레미스 LLM GPU 사이징 & Serving Optimization Guide](gpu-sizing.md) — GPU 스펙, Dense/MoE, KV 캐시, TP/PP/DP/EP, 용량 계획까지 다루는 실전 가이드
