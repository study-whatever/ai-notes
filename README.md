# ai-notes

AI 학습 노트. vLLM 내부 구조(엔진, 스케줄러, KV 캐시, 분산 서빙)부터 에이전트 평가까지 정리합니다.

- [용어집](glossary.md) — 아래 문서들에서 반복 등장하는 공통 용어 정리 (Latency 지표, Prefill/Decode, KV Cache, PagedAttention, TP/PP/DP/EP 등)
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](inside-vllm.md)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](paged-attention.md) — vLLM의 기반 논문 (SOSP '23)
- [온프레미스 LLM GPU 사이징 & Serving Optimization Guide](gpu-sizing.md) — GPU 스펙, Dense/MoE, KV 캐시, TP/PP/DP/EP, 용량 계획까지 다루는 실전 가이드
- [Demystifying evals for AI agents](demystifying-evals-for-ai-agents.md) — Anthropic 엔지니어링 블로그 번역·정리. 에이전트 평가의 구조, 채점기 유형, pass@k/pass^k, 평가 구축 로드맵
- [Investigating unintended model actions](investigating-unintended-model-actions.md) — Anthropic 리서치 블로그 요약. 평가·내부 사용 중 Claude가 보인 제약 우회 행동 4유형(서버 결함 악용, 폼 제출, 유료·접근 제한 우회, URL 단축)과 대응·정렬 관점
- [Advancing computer use with Ironclad](advancing-computer-use-with-ironclad.md) — OpenAI 리서치 블로그 요약. 계약 관리 소프트웨어 워크플로를 학습·평가 과제로 바꾼 방식(11개 과제, rubric 채점, 호스팅 환경 + RL)과 GPT-6 Astra vs GPT-5.6 Sol 비교 결과
