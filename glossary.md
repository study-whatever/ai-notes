# vLLM / LLM Serving 공통 용어집

이 저장소의 여러 노트에서 반복 등장하는 핵심 개념을 한곳에 모은 용어집이다. 각 항목은 정의만 간단히 두고, 구현 세부사항이나 원리를 더 깊게 다루는 문서가 있으면 그쪽으로 링크한다.

## Latency 지표

| 용어 | 의미 |
|---|---|
| **TTFT (Time To First Token)** | 요청 제출 시점부터 첫 번째 출력 토큰을 받을 때까지의 시간. Prefill 단계 성능을 대변 |
| **TPOT (Time Per Output Token)** | 요청에서 생성된 모든 토큰에 대한 평균 ITL. Decode 단계 성능을 대변 |
| **ITL (Inter-Token Latency)** | 연속된 두 토큰 사이의 시간 간격 (예: 토큰 i-1 → 토큰 i) |
| **E2E Latency** | 요청 제출부터 최종 출력 토큰 수령까지의 총 시간 (TTFT + 모든 ITL의 합) |
| **P50 / P95 / P99** | 응답시간 분포에서 50/95/99번째 백분위수. 평균만 보면 "가끔 매우 느린 요청"을 놓치므로 P95/P99를 함께 본다 |
| **Throughput (처리량)** | 시스템이 초당 처리할 수 있는 토큰 수(입력/출력/합산) 또는 요청 수 |
| **Goodput** | 단순 최대 처리량이 아니라, SLO(TTFT/TPOT/E2E 등 조건)를 만족하는 요청만 카운트한 유효 처리량 |

> 상세: [inside-vllm.md의 "Benchmarks and auto-tuning"](inside-vllm.md#benchmarks-and-auto-tuning--latency-vs-throughput) — `vllm bench` CLI 사용법 포함

## Prefill / Decode

| 용어 | 의미 |
|---|---|
| **Prefill** | 프롬프트 전체를 한 번에 처리해 KV 캐시를 채우는 단계. 행렬-행렬 곱으로 병렬화 가능해 **compute-bound** 경향 |
| **Decode** | 토큰을 하나씩 순차 생성하는 단계. 매 스텝마다 모델 가중치 전체를 다시 읽어야 해서 **memory-bound** 경향 |
| **Compute-bound / Memory-bound** | 성능 병목이 "연산 속도"인지 "메모리 대역폭"인지 구분하는 개념 |

> 상세: [inside-vllm.md의 "Scheduler"](inside-vllm.md#scheduler), [paged-attention.md의 "LLM 서비스 및 자연어 생성"](paged-attention.md#llm-서비스-및-자연어-생성)

## KV Cache & PagedAttention

| 용어 | 의미 |
|---|---|
| **KV Cache** | Attention 연산 재사용을 위해 저장해두는 Key/Value 텐서. 컨텍스트가 길어지고 동시 요청이 많아질수록 커짐 |
| **PagedAttention** | KV 캐시를 OS의 가상메모리 페이징처럼 고정 크기 블록(Block)으로 쪼개 관리하는 vLLM의 핵심 알고리즘. 메모리 단편화를 없애 동시 처리량을 크게 높임 |
| **Block / Block Table / Block Manager** | PagedAttention의 구성요소 — 고정 크기 KV 저장 단위(Block), 요청별 논리-물리 블록 매핑(Block Table), 전체 물리 블록 풀 관리자(Block Manager) |
| **Copy-on-Write (CoW)** | 병렬 샘플링·빔 서치 등에서 여러 시퀀스가 블록을 공유하다가, 한쪽이 실제로 값을 바꿔 써야 하는 순간에만 블록을 복사하는 방식 |
| **Continuous Batching** | 요청이 끝나는 즉시 그 자리에 대기 중인 다음 요청을 채워 넣는 반복(iteration) 단위 배칭. GPU를 쉬지 않게 해 처리량을 높임 |
| **Chunked Prefill** | 긴 prefill을 작은 청크로 쪼개 decode 요청들과 같은 배치에 섞어 처리하는 기법. decode가 밀리지 않게 하면서 GPU 활용률을 높임 (`long_prefill_token_threshold`로 임계값 조정) |
| **Prefix Caching** | 여러 요청이 공유하는 프롬프트 앞부분(공통 system prompt 등)을 재계산하지 않고 KV 캐시에서 재사용하는 기능. 블록 해싱(`hash_request_tokens`, `find_longest_cache_hit`)으로 구현됨 |
| **Preemption / Recomputation** | KV 캐시 공간이 부족할 때 일부 요청을 일시 중단(preempt)했다가 나중에 다시 계산(recompute)하는 것. 자주 발생하면 지연시간이 나빠짐 |

> 상세: [paged-attention.md](paged-attention.md) 전체(원 논문 정리), [inside-vllm.md의 "KV cache manager", "Chunked prefill", "Prefix Caching"](inside-vllm.md)

## 분산 서빙 (병렬화)

| 용어 | 의미 |
|---|---|
| **TP (Tensor Parallelism)** | 레이어 내부의 행렬 연산을 여러 GPU로 쪼개 분산 처리. GPU 간 통신이 빈번함. 노드 내 대역폭이 노드 간보다 훨씬 높아 일반적으로 PP보다 선호됨 |
| **PP (Pipeline Parallelism)** | 모델의 레이어 자체를 구간별로 나눠 GPU마다 서로 다른 구간을 담당 |
| **DP (Data Parallelism)** | 모델 전체를 GPU마다 통째로 복제(replica)해두고 트래픽을 나눠 분산 |
| **EP (Expert Parallelism)** | MoE 모델 전용 병렬화. expert들을 여러 GPU에 나눠 배치 |
| **Replica** | 동일한 모델 서빙 인스턴스를 복제한 것. DP의 단위 |

> 상세(내부 구현): [inside-vllm.md의 "From UniprocExecutor to MultiProcExecutor", "Distributed system serving vLLM"](inside-vllm.md), [paged-attention.md의 "분산 실행"](paged-attention.md#분산-실행)
> 상세(사이징 관점 — 언제 어떤 병렬화를 택할지): [gpu-sizing.md의 "7. 분산 서빙: TP/DP/PP/EP"](gpu-sizing.md)

## 기타 고급 기능

| 용어 | 의미 |
|---|---|
| **Guided Decoding (FSM)** | 문법 기반 유한 상태 머신으로 매 스텝 logit을 제약해, 출력이 특정 형식(JSON/regex/CFG)을 반드시 따르도록 강제하는 기법 |
| **Speculative Decoding** | 작은 draft 모델로 k개 토큰을 먼저 예측하고, 큰 모델이 한 번에 검증·채택/거부하는 방식으로 Decode 속도를 높이는 기법. n-gram, Eagle, Medusa 등의 draft 방식이 있음 |
| **Disaggregated P/D** | Prefill과 Decode를 서로 다른 성능 특성(compute-bound vs memory-bound) 때문에 물리적으로 분리된 인스턴스 풀에서 실행하는 아키텍처 |

> 상세: [inside-vllm.md의 "Advanced Features"](inside-vllm.md)

---

각 문서는 이 용어집에 있는 개념을 다시 정의하지 않고 링크만 건다. 새로운 공통 개념이 추가되면 이 파일에 먼저 정리한 뒤 각 문서에서 참조한다.
