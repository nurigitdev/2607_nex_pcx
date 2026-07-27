# NeX_PCX

**Software Requirements Specification v1.50**

*pre-CX RAG / Embedding / VectorDB Experiment Bench*

작성일: 2026-07-02  
문서 상태: Draft v1.50
대상 시스템: FastAPI + Bootstrap + PostgreSQL/pgvector 기반 RAG 실험 플랫폼

본 문서는 NeX-CX 본 개발 이전의 선행 검증 프로젝트인 NeX_PCX의 요구사항을 정의한다.

# 문서 관리 정보

| 항목 | 내용 |
| --- | --- |
| 문서명 | NeX_PCX Software Requirements Specification v1.50 |
| 프로젝트명 | NeX_PCX (pre-CX) |
| 문서 목적 | NeX-CX 본 개발 전 RAG/Embedding/VectorDB 선행 검증 플랫폼의 기능, 데이터, 품질, 테스트 요구사항 정의 |
| 주요 기술 스택 | FastAPI, Bootstrap, PostgreSQL, pgvector, Python, pytest, Playwright |
| 핵심 평가 대상 | KURE-v1 1024, bge-m3 1024, Qwen3-Embedding-4B 1000, Qwen3-Embedding-4B 2560, Qwen3.6-27B-NVFP4 generation runtime |
| 문서 버전 | v1.50 |

| 버전 | 일자 | 작성/변경 내용 |
| --- | --- | --- |
| 0.1 | 2026-07-02 | NeX_PCX 프로젝트명 반영 전 초안 분석 |
| 1.0 | 2026-07-02 | Embedding 4개 profile, FastAPI+Bootstrap, 파일 metadata, 품질관리/테스트 요구사항 반영 |
| 1.1 | 2026-07-02 | pgvector 2560차원 저장 방식, 핵심 DB 스키마, embedding job lifecycle, chunk policy, 검색 재현성 metadata 보강 |
| 1.2 | 2026-07-05 | 계정, 조직 계층, 문서 접근 범위, permission-aware search, 권한 기반 평가 요구사항 보강 |
| 1.3 | 2026-07-05 | 비동기 문서 처리 파이프라인, PostgreSQL job queue, worker 동시성, 진행 상태, 실패/재시도, 관리자 모니터링 요구사항 보강 |
| 1.4 | 2026-07-06 | chunk size/overlap 실험 정책, 기본 운영 정책과 문서 구조 보존 원칙 보강 |
| 1.5 | 2026-07-10 | embedding model 사전 다운로드, models bundle, 오프라인/고객사 설치 배포 정책 보강 |
| 1.6 | 2026-07-10 | GPU embedding provider API 분리, local smoke adapter, provider runtime metadata 요구사항 보강 |
| 1.7 | 2026-07-15 | ingestion metadata, normalized markdown artifact, document block, table/image artifact, chunk source contract 요구사항 보강 |
| 1.8 | 2026-07-20 | BM25 keyword baseline, 검색 전략/profile 계약, embedding-only/BM25/hybrid 비교 요구사항 보강 |
| 1.9 | 2026-07-20 | 한국어 친화 BM25 tokenizer baseline, KoNLPy 보류, Mecab-ko optional 검토 기준 보강 |
| 1.10 | 2026-07-21 | Search Compare BM25 tokenizer 선택, 검색 로그 runtime metadata 재현성 요구사항 보강 |
| 1.11 | 2026-07-24 | Mecab-ko optional BM25 tokenizer foundation, 설치/사전 dependency 상태 표시와 실험 tokenizer 요구사항 보강 |
| 1.12 | 2026-07-25 | BM25 tokenizer별 index backfill 제어, coverage/readiness feedback, 운영 보정 요구사항 보강 |
| 1.13 | 2026-07-25 | Hybrid BM25+Vector RRF 검색 foundation, score component 재현성 요구사항 보강 |
| 1.14 | 2026-07-25 | Search Compare API의 Hybrid RRF profile 실행, vector/BM25 후보 병합 runtime metadata 요구사항 보강 |
| 1.15 | 2026-07-25 | Search Compare UI의 Hybrid RRF profile 선택, hybrid vector-side profile 제어 요구사항 보강 |
| 1.16 | 2026-07-25 | Reranker runtime contract foundation, Qwen3-Reranker-4B provider 계약과 mock baseline 요구사항 보강 |
| 1.17 | 2026-07-25 | Reranked search result core foundation, source score/rank 보존과 rerank score component 요구사항 보강 |
| 1.18 | 2026-07-25 | Search Compare API의 reranked_vector_cosine profile 실행, source vector profile 및 reranker metadata 재현성 요구사항 보강 |
| 1.19 | 2026-07-25 | Remote reranker provider HTTP client/runtime config, Search Compare reranked profile의 remote 호출 설정 요구사항 보강 |
| 1.20 | 2026-07-25 | Qwen3-Reranker-4B remote runtime provider service, CrossEncoder backend와 HTTP 계약 smoke 요구사항 보강 |
| 1.21 | 2026-07-25 | Remote reranker provider foreground launch script, DGX health smoke 절차와 기본 포트 9104 요구사항 보강 |
| 1.22 | 2026-07-25 | Remote reranker request smoke runner, `/v1/rerank` 응답 계약과 score/rank evidence 요구사항 보강 |
| 1.23 | 2026-07-25 | Search Compare remote reranker E2E smoke, query embedding부터 search log metadata까지 검증 요구사항 보강 |
| 1.24 | 2026-07-25 | Search Compare reranker runtime controls UI, reranked source vector profile 선택 및 runtime metadata 표시 요구사항 보강 |
| 1.25 | 2026-07-25 | DGX remote reranker foreground health/request smoke 결과와 운영 실행 증적 요구사항 보강 |
| 1.26 | 2026-07-25 | DGX remote reranker Search Compare live E2E 검증, dev DB migration prerequisite와 search log/runtime metadata 증적 요구사항 보강 |
| 1.27 | 2026-07-25 | Remote reranker background lifecycle runner, PID/log 관리, 중복 실행 방지, status/request smoke evidence 요구사항 보강 |
| 1.28 | 2026-07-25 | Remote reranker operations status API/UI, process/health/request smoke 운영 표시와 runtime 설정 점검 요구사항 보강 |
| 1.29 | 2026-07-25 | Retrieval Context Package API/UI, 생성 단계 입력용 검색 결과 정규화, citation/source anchor/context budget 요구사항 보강 |
| 1.30 | 2026-07-25 | Citation Coverage + Source Anchor Readiness Check, 생성 전 근거 품질 gate와 issue code 요구사항 보강 |
| 1.31 | 2026-07-26 | Generation provider strategy, vLLM OpenAI-compatible runtime contract, Qwen3.6-27B-NVFP4 기본 LLM 후보와 mock-first 검증 원칙 보강 |
| 1.32 | 2026-07-26 | Generation run schema, retrieval package linkage, provider config seed, citation trace table 요구사항 보강 |
| 1.33 | 2026-07-26 | Generation prompt package builder, OpenAI-compatible messages 변환, prompt/context hash와 low-confidence blocked prompt 요구사항 보강 |
| 1.34 | 2026-07-26 | Generation run repository와 deterministic mock executor, no-answer guardrail 실행 기록, citation persistence 요구사항 보강 |
| 1.35 | 2026-07-26 | Generation run API, search log 기반 mock generation 실행 endpoint, generation run detail 조회 요구사항 보강 |
| 1.36 | 2026-07-26 | Generation Run UI MVP, 검색 로그 기반 mock 생성 실행, 답변/status/citation trace 표시 요구사항 보강 |
| 1.37 | 2026-07-26 | Generation Run Detail UI, prompt/context hash, provider/runtime metadata, citation trace 재현성 표시 요구사항 보강 |
| 1.38 | 2026-07-26 | Generation Prompt Preview API/UI, 저장 전 OpenAI-compatible messages, prompt/context hash, block reason 표시 요구사항 보강 |
| 1.39 | 2026-07-26 | vLLM generation provider metrics contract, OpenAI-compatible response usage/finish/error parser 요구사항 보강 |
| 1.40 | 2026-07-26 | Generation provider metrics snapshot API, mock generation run provider_metrics persistence 요구사항 보강 |
| 1.41 | 2026-07-26 | Generation provider metrics snapshot UI, 최근 generation run token/latency/success 운영 패널 요구사항 보강 |
| 1.42 | 2026-07-27 | OpenAI-compatible vLLM client foundation, remote chat completion request/response/error metrics 계약 보강 |
| 1.43 | 2026-07-27 | DGX vLLM remote generation smoke runner, Qwen3.6 model serving 연결 증적과 secret redaction 원칙 보강 |
| 1.44 | 2026-07-27 | Generation provider runtime config API, DGX vLLM seed defaults, DB 기반 LLM runtime 설정 조회와 secret env reference 원칙 보강 |
| 1.45 | 2026-07-27 | Generation provider runtime config UI, DGX vLLM seed action, default/runtime/secret env 상태 표시 요구사항 보강 |
| 1.46 | 2026-07-27 | Remote vLLM generation executor foundation, OpenAI-compatible provider 호출 결과/실패/guardrail no-answer 저장 계약 보강 |
| 1.47 | 2026-07-27 | Remote generation run API, retrieval context에서 DB-backed remote vLLM executor 실행과 env-only API key resolution 요구사항 보강 |
| 1.48 | 2026-07-27 | Remote generation run UI controls, 생성 화면의 default provider readiness 표시와 remote vLLM 실행 feedback 요구사항 보강 |
| 1.49 | 2026-07-27 | Generation provider metrics mode breakdown, mock/remote generation 실행의 성공률, token, latency 분리 관측 요구사항 보강 |
| 1.50 | 2026-07-27 | DGX vLLM generation run live E2E verification, retrieval context package에서 remote generation run/citation/metrics persistence까지 검증하는 운영 증적 요구사항 보강 |

# 목차

1. 개요

2. 전체 시스템 설명

3. 시스템 아키텍처

4. 기능 요구사항

5. 데이터 및 저장소 요구사항

6. Web UI 요구사항

7. Embedding 및 검색 평가 요구사항

8. 비기능 요구사항

9. 소프트웨어 품질관리 및 테스트 요구사항

10. 개발 단계 및 인수 기준

11. 주요 리스크 및 대응 방안

부록 A. 요구사항 ID 목록

부록 B. 참고자료

# 1. 개요

## 1.1 목적

NeX_PCX는 NeX-CX 개발 이전에 사내 RAG 기반 AX 구축 역량을 검증하기 위한 선행 실험 플랫폼이다. 본 SRS는 NeX_PCX의 기능 요구사항, 데이터 저장 구조, 검색 비교 방식, 품질관리 및 테스트 기준을 정의한다.

> **핵심 목적**
>
> PDF/DOCX/HWPX/PPTX/XLSX/MD 문서를 업로드하고, 동일한 chunk를 4개 embedding profile로 변환하여 PostgreSQL + pgvector에 저장한 뒤, BM25 keyword baseline과 embedding-only/hybrid 검색 결과를 병렬 비교하고 정량 평가 데이터를 축적한다.

## 1.2 배경

- 사내 AX 구축을 위해 RAG(Retrieval-Augmented Generation) 기반의 문서 검색/생성 시스템이 필요하다.

- 현재는 문서를 chunk 단위로 분할하고 embedding model로 vector화하여 VectorDB에 저장한다는 개념은 알고 있으나, 실제 구현 경험과 실험 데이터가 부족하다.

- Embedding model, chunk size, overlap, 문서 포맷, 표/그림 처리 방식, keyword retrieval, vector index에 따라 검색 품질과 저장용량이 달라질 수 있다.

- NeX-CX 본 개발 전, 계속 등장하는 embedding model과 검색 기법을 동일 기준으로 반복 비교할 수 있는 실험 환경이 필요하다.

- Vector-only RAG의 환각 위험을 줄이기 위해 BM25 keyword baseline, hybrid retrieval, 목차/section graph 기반 보조 검색 구조도 실험할 예정이다.

## 1.3 범위

| 구분 | 포함 범위 |
| --- | --- |
| 문서 업로드 | PDF, DOCX, HWPX, PPTX, XLSX, MD 파일 업로드 및 원본 metadata 저장 |
| 문서 처리 | 텍스트 추출, heading/section 추정, chunk 생성, chunk metadata 저장 |
| Embedding | KURE-v1 1024, bge-m3 1024, Qwen3 1000, Qwen3 2560 profile 생성 및 저장 |
| 비동기 처리 | 업로드 이후 텍스트 추출, parsing, chunking, embedding, vector indexing을 job queue와 worker로 처리 |
| 검색 | 동일 query에 대한 4개 embedding profile, BM25 keyword baseline, hybrid, reranked 검색 결과 표시 |
| 평가 | 검색 결과 정답/부분정답/오답 피드백 저장, 검색 로그/latency 저장 |
| 계정/권한 | 테스트용 사용자, 조직 계층, 역할, 문서 접근 범위, query 검색 범위 metadata 저장 |
| 생성 실험 | Retrieval context package를 기반으로 mock 또는 vLLM OpenAI-compatible LLM runtime에 grounded answer 생성을 요청하고 실행 조건과 runtime metadata를 저장 |
| 품질관리 | pytest, Playwright, pytest-cov, 독립 test DB, quality gate 절차 적용 |

## 1.4 범위 제외

| 항목 | 제외 사유 / 향후 계획 |
| --- | --- |
| 운영용 챗봇/업무 자동화 | NeX_PCX의 생성 기능은 검색 결과와 prompt/provider 계약 검증용 실험 기능으로 제한한다. 최종 사용자용 대화 UX, 장기 메모리, workflow automation은 NeX-CX 본 개발에서 확정한다. |
| 완전한 GraphRAG | 초기에는 heading/section/chunk metadata를 축적하고, 이후 graph-assisted retrieval로 확장한다. |
| 고도화된 이미지 embedding | 1차에서는 그림 원본/페이지/주변 텍스트/캡션 metadata 저장 수준으로 제한한다. |
| 운영용 SSO/HR 연동 | NeX_PCX에서는 테스트용 계정/조직 seed와 권한 시뮬레이션으로 제한한다. 운영 SSO/HR 연동은 NeX-CX 본 개발에서 확정한다. |
| 대규모 분산 검색 | 초기에는 단일 PostgreSQL + worker 구조로 검증하고, 성능 병목 확인 후 확장한다. |

## 1.5 용어 및 약어

| 용어 | 정의 |
| --- | --- |
| NeX_PCX | NeX-CX 본 개발 이전의 pre-CX 실험 플랫폼 |
| NeX-CX | Context Intelligence Transformation을 위한 사내 문서 ingestion, embedding, 검색 기반 제품/플랫폼 |
| RAG | Retrieval-Augmented Generation. 검색 결과를 LLM 답변 생성의 근거로 사용하는 방식 |
| Generation Provider | Retrieval context package와 user query를 입력으로 받아 grounded answer를 생성하는 LLM runtime. NeX_PCX는 mock과 vLLM OpenAI-compatible remote runtime을 우선 지원한다. |
| vLLM Runtime | DGX-Spark 등 GPU 서버에서 LLM을 OpenAI-compatible HTTP API로 제공하는 serving runtime |
| OpenAI-compatible Chat API | `/v1/chat/completions` 형식의 messages 기반 HTTP 생성 API. NeX_PCX의 remote LLM provider 기본 호출 계약으로 사용한다. |
| Embedding Profile | 특정 모델명, dimension, normalization, pooling 설정을 포함한 embedding 실행 단위 |
| Embedding Job | 특정 chunk와 embedding profile 조합에 대해 embedding 생성, 저장, 실패/재시도를 추적하는 background 작업 단위 |
| Search Profile | 검색 비교 화면과 검색 로그에서 하나의 결과 column/lane으로 표시되는 검색 실행 단위. embedding profile, BM25 keyword baseline, hybrid, reranked strategy를 포괄한다. |
| Retrieval Strategy | query를 chunk 후보군과 점수로 변환하는 검색 알고리즘 선택값. vector_cosine, bm25_keyword, hybrid_keyword_vector, reranked_vector_cosine 등을 포함한다. |
| BM25 | Best Match 25. query term 빈도, 문서 빈도, chunk 길이를 반영하는 keyword retrieval baseline으로 사용한다. |
| Hybrid Retrieval | BM25 keyword 결과와 embedding vector 결과를 fusion하여 단독 검색보다 안정적인 결과를 실험하는 검색 방식 |
| Chunk | 문서 본문을 검색 가능한 단위로 분할한 텍스트 조각 |
| Chunk Policy | chunk size, overlap, heading/table/code block 처리 방식 등을 포함하는 chunk 생성 정책 |
| pgvector | PostgreSQL에서 vector, halfvec type 및 similarity search를 지원하는 확장 |
| Groundedness | 답변 또는 검색 결과가 실제 문서 근거에 의해 뒷받침되는 정도 |
| Quality Gate | commit/sync 전에 반드시 통과해야 하는 테스트 및 품질 기준 |
| Actor | 문서 업로드, 검색, 피드백을 수행하는 사용자 또는 테스트 계정 |
| Organization Unit | 팀, 그룹, 본부 등 조직 계층을 표현하는 단위 |
| Document Access Scope | 문서가 검색 대상에 포함될 수 있는 범위. personal, team, org_tree, company 등 |
| Query Search Scope | 사용자가 query 실행 시 선택하는 검색 범위. mine, team, managed_org, company 등 |
| Permission Pre-filter | vector search 후보군 생성 전에 actor 권한 조건으로 검색 대상을 제한하는 방식 |
| Pipeline Job | 업로드 후 텍스트 추출, parsing, chunking, embedding, vector indexing 단계를 추적하는 비동기 작업 단위 |
| Pipeline Job Event | pipeline job의 상태 변경, 진행률 변경, 오류, 재시도 이력을 append-only로 기록한 이벤트 |
| Worker Lease | 여러 worker가 같은 job을 중복 처리하지 않도록 일정 시간 동안 job 점유권을 부여하는 방식 |
| Extraction Provider | PDF/DOCX/HWPX/PPTX/XLSX/MD 등 원본 파일에서 텍스트, 구조, 표, 그림 metadata를 추출하는 실행 단위. 초기에는 local runtime으로 시작하되 remote provider로 분리 가능한 contract를 가진다. |
| Extraction Artifact | 원본 파일에서 추출된 normalized markdown, plain text, parser metadata, warning/error report 등 재처리와 재현성에 필요한 산출물 |
| Normalized Markdown Artifact | 파일 형식과 무관하게 검색/검토/재처리에 사용할 수 있도록 정규화한 Markdown 또는 Markdown-like text 산출물 |
| Document Block | normalized artifact 안의 heading, paragraph, table, image, figure, list, code, page, slide, sheet 등 chunk 생성 이전의 구조 단위 |
| Source Anchor | block 또는 chunk가 원본 파일의 어느 위치에서 왔는지 나타내는 page/slide/sheet/cell/char range/geometry metadata |
| Chunk Source Contract | chunk가 어떤 extraction artifact와 document block에서 생성되었고, 앞/뒤 chunk와 어떻게 연결되는지 정의하는 공통 저장 계약 |

# 2. 전체 시스템 설명

## 2.1 제품 포지셔닝

NeX_PCX는 운영 서비스가 아니라 NeX-CX 본 개발을 위한 실험/검증 플랫폼이다. 따라서 최종 사용자에게 답변을 제공하는 챗봇보다, 개발팀과 기획팀이 모델·chunk·검색 정책을 비교하고 증거 데이터를 축적하는 도구에 가깝다.

| 관점 | NeX_PCX | NeX-CX |
| --- | --- | --- |
| 목적 | 선행 검증, 모델/정책 비교, benchmark 축적 | 사내 업무 문서 검색/생성 서비스 |
| 사용자 | 개발자, 기획자, 검증 담당자 | 전사 사용자, 관리자, 업무 담당자 |
| 중심 기능 | 검색 결과 비교와 평가 | 업무 답변 생성, 문서 자동화, 운영관리 |
| 품질 기준 | 재현 가능한 실험과 테스트 자동화 | 운영 안정성, 보안, 확장성, 사용자 경험 |

## 2.2 사용자 역할

| 역할 | 주요 책임 | 주요 사용 기능 |
| --- | --- | --- |
| Project Owner | 실험 목적과 우선순위 결정 | 대시보드, 평가 결과 조회 |
| Developer | 파서, chunker, embedding, 검색 API 구현 | 업로드, 검색, 로그, 테스트 실행 |
| Evaluator | 검색 결과 품질 판정 및 피드백 입력 | 검색 비교 화면, 정답/부분정답/오답 체크 |
| System Admin | DB, worker, 저장소, 모델 실행 환경 관리 | 대시보드, job 상태, 오류 로그 확인 |
| Test User | 권한 실험용 일반 사용자 | 개인/팀/company 문서 업로드, 제한된 검색 |
| Team Lead | 팀 단위 권한 실험용 사용자 | 본인 및 팀원 문서 검색, 팀 scope 평가 |
| Group Lead | 하위 조직 포함 권한 실험용 사용자 | managed_org scope 검색, 그룹 단위 평가 |
| Permission Admin | 테스트 계정, 조직, 역할 seed 관리 | 권한 fixture 관리, 접근 범위 검증 |

## 2.3 운영 가정

- 초기 PoC는 Docker 없이 내부 개발/검증용 단일 서버와 로컬 worker process 기준으로 운영한다.

- PostgreSQL에는 pgvector extension이 활성화되어야 한다.

- Qwen3-Embedding-4B 2560 profile은 GPU 메모리와 처리 시간이 가장 많이 필요하므로 별도 worker로 분리한다.

- Qwen3-Embedding-4B 2560 profile은 pgvector의 일반 vector type 차원 한계를 초과하므로 MVP에서는 halfvec(2560) 저장 방식을 사용한다.

- 개발 환경에서는 명시적인 download script로 embedding model을 `models/` 하위 디렉토리에 내려받을 수 있다.

- 운영 환경 또는 고객사 설치 환경에서는 앱 시작 시 public internet에서 model을 자동 다운로드하지 않고, 사전에 검증된 `models/` bundle을 배포한다.

- 개발 PC는 GPU가 없을 수 있으므로 local model 실행은 smoke/debug 용도로 제한하고, 대량 ingestion 및 benchmark는 GPU 서버의 embedding provider API를 호출하는 구조를 지원한다.

- NeX_PCX 본체는 upload, parsing, chunking, queue, permission, pgvector 저장, 검색 로그/평가 기록을 담당하고, embedding 계산은 local adapter 또는 remote provider 중 하나로 위임한다.

- 파일 업로드 후 embedding은 background job으로 수행하며, 대시보드에서 profile별 진행률을 확인할 수 있어야 한다.

- MVP의 background queue는 Redis/Celery 등 외부 broker가 아니라 PostgreSQL 기반 pipeline job queue로 구현한다.

- worker는 초기에는 단일 process/단일 concurrency로 시작하고, 이후 설정값으로 worker 수와 stage별 동시성을 늘릴 수 있어야 한다.

- 검색 비교의 공정성을 위해 동일 문서, 동일 parser, 동일 chunk, 동일 query, 동일 top-k 기준으로 embedding profile, BM25 keyword baseline, hybrid strategy를 비교한다.

- 권한 실험의 공정성을 위해 동일 query라도 actor, requested_search_scope, effective_permission_filter를 search log에 기록한다.

- 권한 조건은 vector search 이후 결과 제거 방식이 아니라 후보군 생성 단계의 pre-filter로 적용한다.

- NeX_PCX의 권한 기능은 운영 인증 체계가 아니라 NeX-CX 적용성 검증을 위한 permission simulation layer로 구현한다.

- 문서 추출 기능은 embedding provider와 유사하게 provider contract를 먼저 정의하되, MVP에서는 local/in-process extraction runtime으로 시작한다.

- OCR, layout-aware PDF, table structure recognition, image captioning처럼 무거운 추출 기능은 향후 remote extraction provider로 분리할 수 있어야 한다.

- 모든 파일 형식의 추출 결과는 normalized markdown artifact를 생성하고, chunk는 이 artifact와 document block/source anchor를 기준으로 생성한다.

- 파일 metadata는 확장자와 무관한 공통 metadata와 PDF/DOCX/HWPX/PPTX/XLSX/MD별 format-specific metadata를 분리하여 저장한다.

# 3. 시스템 아키텍처

## 3.1 논리 아키텍처

```text
[Bootstrap Web UI]
├─ Dashboard
├─ File Upload
├─ Permission Simulation
└─ Search / Model Compare
│
▼
[FastAPI Backend]
├─ Upload API
├─ Identity / Permission API
├─ Pipeline Job API
├─ Ingestion Artifact API
├─ Document Parsing Service
├─ Extraction Provider Client
├─ Chunking Service
├─ Embedding Job API
├─ Search API
├─ Statistics API
└─ Feedback API
│
▼
[PostgreSQL Pipeline Queue]
├─ pipeline_jobs
└─ pipeline_job_events
│
▼
[Background Worker Runners]
├─ text extraction / artifact worker
├─ parsing / document block worker
├─ chunking worker
└─ embedding worker / provider client
│
▼
[Extraction Provider Layer]
├─ local extraction runtime: md/plain text/pdf/docx/pptx/xlsx/hwpx
└─ future remote extraction provider: OCR / layout / table / image captioning
│
▼
[Embedding Provider Layer]
├─ local smoke adapter: sentence-transformers / qwen adapter
├─ remote GPU provider: kure_v1_1024
├─ remote GPU provider: bge_m3_1024
└─ remote GPU provider: qwen3_4b_1000 / qwen3_4b_2560
│
▼
[PostgreSQL + pgvector]
├─ app_users / org_units / memberships
├─ files / documents / extraction_artifacts / document_blocks / chunks
├─ table_artifacts / image_artifacts
├─ pipeline_jobs / pipeline_job_events
├─ embedding profile tables
├─ search_logs / feedback
└─ experiment/evaluation tables
```

## 3.2 주요 컴포넌트

| 컴포넌트 | 책임 |
| --- | --- |
| Web UI | Bootstrap 기반 대시보드, 파일 업로드, 검색 비교, 피드백 입력 화면 제공 |
| FastAPI Backend | REST API 제공, validation, 서비스 호출, DB transaction 처리 |
| File Storage | 업로드 원본 파일과 추출 산출물 저장 |
| Identity/Permission Service | 테스트 계정, 조직 계층, 역할, 문서 접근 범위, 검색 scope 계산 |
| Pipeline Job Service | 업로드 이후 텍스트 추출, parsing, chunking, embedding, vector indexing 작업 생성과 상태 전이 관리 |
| PostgreSQL Job Queue | pipeline_jobs를 기반으로 queued job, lease, retry, progress, worker heartbeat를 관리 |
| Extraction Provider Contract | 원본 파일 입력, normalized markdown artifact 출력, source anchor, warning/error metadata를 local/remote 동일 contract로 정의 |
| Parser Service | 파일 타입별 텍스트/구조/메타데이터 추출 및 document block 생성 |
| Ingestion Artifact Repository | normalized markdown, parser metadata, warning/error report, table/image artifact 위치를 저장하고 재처리 기준을 제공 |
| Chunking Service | document block과 chunk policy에 따라 chunk 생성, source anchor, prev/next 연결을 저장 |
| Worker Runners | stage별 job 점유, 처리, heartbeat, 실패/재시도, 처리 시간/오류 기록 |
| Embedding Worker | embedding job을 점유하고 provider를 호출한 뒤 vector 저장, job 상태, runtime metadata를 기록 |
| Embedding Provider Client | local adapter 또는 remote GPU provider API를 동일한 request/response contract로 호출 |
| Remote Embedding Provider | GPU 서버에서 model preload, batch embedding, provider health, latency metadata를 제공 |
| Search Service | query embedding 생성, BM25 keyword search, pgvector 검색, hybrid fusion, profile/strategy별 결과 병렬 반환 |
| Retrieval Context Service | 검색 로그 결과를 생성 입력으로 정규화하고 citation, source anchor, permission scope, confidence metadata를 포함한 context package를 제공 |
| Generation Provider Client | mock 또는 remote OpenAI-compatible vLLM runtime을 호출하고 model/provider/prompt/runtime metadata를 generation run에 기록 |
| Remote vLLM Runtime | DGX-Spark 등 GPU 서버에서 Qwen3.6-27B-NVFP4 등 LLM을 OpenAI-compatible `/v1/chat/completions` 계약으로 제공 |
| Statistics Service | 문서/chunk/vector/job/검색 통계 산출 |
| Model Distribution Tooling | KURE, bge-m3, Qwen3 embedding model을 `models/` bundle로 사전 다운로드하고 설치 환경에서 검증 |
| Quality/Test Framework | pytest, Playwright, coverage, 독립 test DB 기반 품질 검증 |

## 3.3 배포 구조

```text
Docker 미사용 로컬 서버 기준 초기 배포 구조

processes:
web-app: FastAPI + Bootstrap templates
postgres: PostgreSQL + pgvector
worker-default: text extraction, parsing, chunking, light embedding job 처리
worker-qwen: Qwen3-Embedding-4B 1000/2560 profile 전용 worker
generation-mock: 개발/test 환경의 deterministic mock generation runtime
vllm-qwen: DGX-Spark 외부 프로세스, Qwen3.6-27B-NVFP4 OpenAI-compatible chat completions runtime
playwright-test: E2E test runner (CI 또는 local gate)
```

MVP에서는 별도 broker process를 두지 않고 PostgreSQL의 row lock, lease, SKIP LOCKED 기반 claim query로 queue를 구현한다. 이후 NeX-CX 본 개발에서 처리량 요구가 커질 경우 Redis/Celery, RabbitMQ, cloud queue 등 외부 broker 전환 여부를 재검토한다.

# 4. 기능 요구사항

## 4.1 요구사항 우선순위 기준

| 우선순위 | 정의 |
| --- | --- |
| MUST | MVP에 반드시 포함되어야 하며, 누락 시 인수 불가 |
| SHOULD | 가능하면 포함하되 일정상 Phase 2로 이월 가능 |
| COULD | 후속 실험 또는 확장 기능으로 고려 |

## 4.2 기능 요구사항 목록

| ID | 기능 | 설명 | 우선순위 |
| --- | --- | --- | --- |
| FR-001 | 대시보드 통계 조회 | 문서, 파일, chunk, embedding profile, 검색 latency, 오류 현황을 조회한다. | MUST |
| FR-002 | 파일 업로드 | PDF/DOCX/HWPX/PPTX/XLSX/MD 파일을 업로드한다. | MUST |
| FR-003 | 원본 metadata 저장 | 원본 파일명, 파일크기, MIME type, checksum, 저장 경로, 업로드 시간 등을 저장한다. | MUST |
| FR-004 | 중복 파일 검출 | sha256 checksum 기준으로 중복 업로드를 탐지한다. | MUST |
| FR-005 | 문서 parsing | 파일 타입별 parser를 통해 text와 구조 metadata를 추출한다. | MUST |
| FR-006 | Chunk 생성 | heading-aware chunking을 기본 정책으로 chunk를 생성한다. | MUST |
| FR-007 | Chunk metadata 저장 | page/slide/sheet, heading_path, token_count, prev/next chunk를 저장한다. | MUST |
| FR-008 | Embedding profile 관리 | 4개 embedding profile을 metadata로 관리한다. | MUST |
| FR-009 | Profile별 embedding job | 동일 chunk를 4개 profile에 대해 background job으로 처리한다. | MUST |
| FR-010 | pgvector 저장 | profile별 embedding table에 vector를 저장한다. | MUST |
| FR-011 | 검색 API | query를 선택된 search profile 또는 retrieval strategy로 검색하고 결과를 병렬 반환한다. | MUST |
| FR-012 | 검색 결과 비교 UI | embedding profile, BM25 baseline, hybrid 결과를 column/lane UI로 표시한다. | MUST |
| FR-013 | 검색 피드백 저장 | 정답/부분정답/오답/중복/문맥부족 피드백을 저장한다. | MUST |
| FR-014 | 검색 로그 저장 | query, top-k, profile별 latency, 결과 chunk를 저장한다. | MUST |
| FR-015 | Embedding 실패 재처리 | 실패한 chunk/profile job을 재시도할 수 있어야 한다. | SHOULD |
| FR-016 | 평가 자동화 | golden question set 기반 Recall@k, MRR, nDCG를 산출한다. | SHOULD |
| FR-017 | Graph-assisted 확장 준비 | heading/section/chunk graph 저장을 위한 metadata를 보존한다. | SHOULD |
| FR-018 | Embedding job 상태 관리 | chunk/profile별 pending/running/succeeded/failed 상태, 시도 횟수, 오류 메시지를 저장한다. | MUST |
| FR-019 | Chunk policy 관리 | chunk size, overlap, split strategy, table/code block 보존 정책을 metadata로 관리한다. | MUST |
| FR-020 | 검색 재현성 metadata 저장 | 검색 시점의 profile runtime 설정, chunk policy, similarity metric, top-k, query instruction을 로그로 저장한다. | MUST |
| FR-021 | 테스트 계정 관리 | 권한 실험을 위한 사용자 계정, 표시명, 활성 상태를 관리한다. | MUST |
| FR-022 | 조직 계층 관리 | 팀, 그룹, 본부 등 계층형 조직 단위와 사용자 소속/역할을 관리한다. | MUST |
| FR-023 | 문서 소유자 및 접근 범위 저장 | 업로드 문서마다 업로드 사용자, 소유 조직, 접근 범위(personal/team/org_tree/company)를 저장한다. | MUST |
| FR-024 | 권한 기반 검색 범위 선택 | query 실행 시 actor와 requested_search_scope를 선택하고 effective scope를 산출한다. | MUST |
| FR-025 | Permission pre-filter 검색 | 권한 조건을 vector search 이후가 아니라 후보군 생성 단계에서 pre-filter로 적용한다. | MUST |
| FR-026 | 권한별 검색 결과 비교 | 동일 query에 대해 사용자/검색 범위별 결과 차이를 비교하고 평가할 수 있어야 한다. | SHOULD |
| FR-027 | 권한 metadata 감사 로그 | 업로드, 검색, 피드백 이벤트에 actor와 permission metadata를 남긴다. | MUST |
| FR-028 | 공통 문서 관리 | 회사 규칙, 업무 규칙, 사규 등 company scope 문서를 별도 구분하여 전사 검색 대상에 포함한다. | MUST |
| FR-029 | Pipeline job 생성 | 파일 업로드 완료 후 텍스트 추출, parsing, chunking, embedding, vector indexing을 위한 비동기 pipeline job을 생성한다. | MUST |
| FR-030 | Pipeline 진행 상태 조회 | 파일/문서별 현재 stage, status, 진행률, 시작/종료 시간, 오류 요약을 조회한다. | MUST |
| FR-031 | PostgreSQL job queue | pipeline_jobs를 PostgreSQL 기반 queue로 사용하고 worker가 lease 방식으로 job을 점유한다. | MUST |
| FR-032 | Worker 동시성 제어 | worker 수, stage별 동시성, profile별 worker 분리를 설정값으로 관리한다. | SHOULD |
| FR-033 | Pipeline 실패/재시도 | 실패한 pipeline job은 오류 원인, attempt 수, 재시도 가능 여부를 저장하고 수동 재시도할 수 있어야 한다. | MUST |
| FR-034 | Pipeline event 감사 | job 생성, 점유, stage 전환, 진행률, 실패, 재시도 이벤트를 append-only event로 저장한다. | SHOULD |
| FR-035 | Embedding provider 선택 | profile별 embedding 실행을 local adapter 또는 remote GPU provider로 선택할 수 있어야 한다. | MUST |
| FR-036 | Remote provider health check | provider URL, model key, dimension, device, ready 상태를 embedding 실행 전 점검할 수 있어야 한다. | MUST |
| FR-037 | Provider runtime metadata 저장 | embedding job과 search log에 provider type, endpoint, model source, dimension, device, elapsed_ms를 저장한다. | MUST |
| FR-038 | Extraction provider contract | 파일 추출 기능은 local runtime과 future remote provider가 공유하는 request/response contract를 가져야 한다. | MUST |
| FR-039 | Normalized markdown artifact 저장 | 원본 파일별 추출 결과를 Markdown 또는 Markdown-like text artifact로 저장한다. | MUST |
| FR-040 | 공통/형식별 metadata 분리 | 파일 확장자와 무관한 공통 metadata와 파일 형식별 metadata를 분리하여 저장한다. | MUST |
| FR-041 | Document block 저장 | heading, paragraph, table, image, figure, list, code, page, slide, sheet 등 chunk 생성 이전의 구조 block을 저장한다. | MUST |
| FR-042 | Chunk source contract 저장 | chunk는 artifact_id, block_id, chunk_type, source_anchor, prev/next chunk 연결을 통해 원본 근거를 추적할 수 있어야 한다. | MUST |
| FR-043 | Table artifact 보존 | 표는 검색용 텍스트 표현과 별도로 markdown/json/csv 중 하나 이상의 구조 artifact를 보존할 수 있어야 한다. | SHOULD |
| FR-044 | Image artifact 보존 | 그림은 파일 위치, page/slide 위치, OCR text, caption, 주변 텍스트 metadata를 저장할 수 있어야 한다. | SHOULD |
| FR-045 | 추출 결과 재처리 | extractor version, profile, artifact hash를 기준으로 원본 파일 재추출과 chunk 재생성이 가능해야 한다. | SHOULD |
| FR-046 | BM25 keyword baseline | embedding 없이 chunk text와 keyword index만으로 BM25 검색 결과를 반환한다. | SHOULD |
| FR-047 | Search profile/strategy 관리 | embedding profile, BM25, hybrid를 동일 비교 화면과 검색 로그에서 구분 가능한 search profile/strategy로 관리한다. | MUST |
| FR-048 | Hybrid 검색 비교 | BM25 결과와 vector 결과를 fusion하여 embedding-only, BM25-only, hybrid 방식을 같은 query/golden set으로 비교한다. | SHOULD |
| FR-049 | Retrieval context package | 검색 로그의 top-k 결과를 생성 단계에서 사용할 수 있는 context package로 정규화하고 citation, source anchor, 권한 scope, runtime metadata, context budget을 함께 제공한다. | MUST |
| FR-050 | Citation readiness check | 생성 단계로 넘기기 전 retrieval context의 citation key, 문서/파일 식별자, chunk 식별자, source anchor/location/lineage coverage를 점검하고 issue code를 표시한다. | MUST |
| FR-051 | Generation provider strategy | 생성 기능은 provider mode를 `mock`과 `remote_openai_compatible`로 분리하고, 개발/test 환경에서는 deterministic mock provider로 schema/API/UI 계약을 먼저 검증한다. | MUST |
| FR-052 | vLLM runtime contract | remote LLM provider는 vLLM OpenAI-compatible `/v1/chat/completions` 계약을 기본으로 사용하며 base URL, model id, API key 사용 여부, timeout, max tokens, temperature, top_p를 runtime 설정으로 관리한다. | MUST |
| FR-053 | 기본 LLM 후보 | NeX_PCX의 기본 remote LLM 후보는 `nvidia/Qwen3.6-27B-NVFP4`로 정의하되, 실제 DGX-Spark 연결 smoke가 완료되기 전까지 운영 test는 mock provider를 기본값으로 사용한다. | SHOULD |
| FR-054 | Grounded generation gate | 생성 실행은 retrieval confidence가 `answerable`이고 citation readiness가 `failed`가 아닌 context package를 우선 대상으로 하며, `low_confidence` 또는 `no_relevant_context` 상태에서는 no-answer 응답 또는 실행 차단을 기록한다. | MUST |
| FR-055 | Generation run execution persistence | 생성 실행기는 prompt package, provider snapshot, retrieval confidence, citation readiness, answer/no-answer 결과, token/latency 추정치, citation 사용 여부를 `generation_runs`와 `generation_run_citations`에 재현 가능하게 저장한다. | MUST |
| FR-056 | Generation run API | 검색 로그 ID를 기준으로 retrieval context를 구성해 mock generation run을 생성하는 API와, 저장된 generation run/citation detail을 조회하는 API를 제공한다. | MUST |
| FR-057 | Generation Run UI MVP | Web UI는 검색 로그 ID 기반으로 retrieval context를 불러오고 mock generation run을 실행한 뒤 답변, status, guardrail, token/latency, citation trace를 표시한다. | MUST |
| FR-058 | Generation Run Detail UI | Web UI는 저장된 generation run을 독립 URL로 열람하며 answer, provider/model, prompt/context hash, package key, token/latency, request/response/guardrail metadata, citation trace를 표시한다. | MUST |
| FR-059 | Generation Prompt Preview API/UI | 생성 실행 전 검색 로그 기반 retrieval context를 prompt package로 변환해 OpenAI-compatible messages, response language, citation keys, prompt/context hash, blocked 여부와 block reason을 저장 없이 조회할 수 있어야 한다. | MUST |
| FR-060 | vLLM generation provider metrics contract | remote OpenAI-compatible vLLM 응답에서 model/id, finish_reason, usage token count, HTTP status, latency, retry count, error code/message를 표준 metrics payload로 파싱해 generation run metadata와 운영 모니터링에서 재사용할 수 있어야 한다. | MUST |
| FR-061 | Generation provider metrics snapshot API | mock 및 future vLLM generation run은 `response_metadata.provider_metrics`에 표준 metrics payload를 저장하고, 운영 API는 최근 generation run의 성공/실패, no-answer, token, latency, provider elapsed summary를 조회할 수 있어야 한다. | MUST |
| FR-062 | Generation provider metrics snapshot UI | Web UI는 최근 generation run의 provider metrics snapshot을 요약 카드, 실행별 token/latency table, raw JSON evidence로 표시하고 각 run detail로 이동할 수 있어야 한다. | SHOULD |
| FR-063 | OpenAI-compatible vLLM client foundation | remote generation provider client는 `/v1/chat/completions` request payload를 구성하고 answer text, finish reason, token usage, latency, HTTP error payload를 표준 metrics와 함께 반환하거나 실패 객체로 보존할 수 있어야 한다. | MUST |
| FR-064 | DGX vLLM generation smoke evidence | DGX-Spark vLLM runtime에 대해 `/v1/chat/completions` smoke runner를 제공하고 endpoint, model id, HTTP status, latency, finish reason, token usage, answer preview를 secret 없이 증적으로 저장할 수 있어야 한다. | MUST |
| FR-065 | Generation provider runtime config API | 운영자는 DB에 저장된 generation provider 설정 목록과 default provider runtime contract를 조회하고, DGX vLLM Qwen3.6 기본값을 secret 값 없이 environment variable reference만으로 seed/upsert할 수 있어야 한다. | MUST |
| FR-066 | Generation provider runtime config UI | Web UI는 generation provider 설정 목록, default provider, runtime validation 상태, API key env reference 구성 여부를 표시하고 DGX vLLM 기본값을 secret 입력 없이 seed할 수 있어야 한다. | SHOULD |
| FR-067 | Remote vLLM generation executor foundation | remote generation executor는 default provider가 `remote_openai_compatible`일 때 prompt package를 `/v1/chat/completions` 요청으로 실행하고 성공, provider failure, guardrail no-answer 결과를 동일한 generation run/citation schema에 저장할 수 있어야 한다. | MUST |
| FR-068 | Remote generation run API | API 사용자는 검색 이력의 retrieval context를 remote vLLM generation run으로 승격할 수 있어야 하며, API key는 DB에 저장하지 않고 provider 설정의 env reference를 해석해 runtime에만 전달해야 한다. | MUST |
| FR-069 | Remote generation run UI controls | 생성 화면은 default generation provider runtime readiness, API key env 준비 상태, remote 실행 가능 여부를 표시하고, 준비된 경우 remote vLLM generation run을 시작해 결과 feedback으로 연결할 수 있어야 한다. | SHOULD |
| FR-070 | Generation provider metrics mode breakdown | 운영자는 generation provider metrics snapshot에서 mock과 remote_openai_compatible 실행을 provider mode별로 분리해 실행 수, 성공/실패, no-answer, token, elapsed/provider elapsed 평균을 비교할 수 있어야 한다. | SHOULD |
| FR-071 | DGX vLLM generation run live E2E verification | 운영자는 DGX-Spark vLLM runtime을 대상으로 임시 검색 로그 fixture에서 retrieval context package를 생성하고, remote generation executor 실행 결과가 `generation_runs`, `generation_run_citations`, provider metrics에 secret 없이 저장되는지 검증하는 CLI 증적을 남길 수 있어야 한다. | MUST |

## 4.3 대시보드 요구사항

| ID | 요구사항 |
| --- | --- |
| FR-001-01 | 전체 문서 수, 파일 타입별 문서 수, 문서 그룹별 문서 수를 표시한다. |
| FR-001-02 | 원본 파일 총 용량, 평균 파일 크기, 중복 파일 수를 표시한다. |
| FR-001-03 | 총 chunk 수, 평균 token 수, chunk 정책별 chunk 수를 표시한다. |
| FR-001-04 | kure_v1_1024, bge_m3_1024, qwen3_4b_1000, qwen3_4b_2560 profile별 완료/대기/실패 건수를 표시한다. |
| FR-001-05 | profile별 평균 embedding 시간과 평균 검색 latency를 표시한다. |
| FR-001-06 | parsing 실패, embedding 실패, DB 저장 실패 목록을 최근순으로 표시한다. |
| FR-001-07 | pipeline queue depth, running job 수, 오래 대기 중인 job, lease 만료 job, 실패 job 수를 표시한다. |
| FR-001-08 | stage별 평균 처리 시간과 최근 실패 원인을 표시한다. |

## 4.4 파일 업로드 및 parsing 요구사항

| 파일 형식 | 처리 요구사항 |
| --- | --- |
| PDF | page text 추출, page_no 저장, text 없는 page 감지, 원본 PDF 보관 |
| DOCX | heading, paragraph, table text 추출, 문서 구조 보존 |
| HWPX | XML 기반 paragraph/table text 추출, section 구조 보존 |
| PPTX | slide_no, title, body, table text 추출, slide 단위 chunk 지원 |
| XLSX | sheet_name, used range, row/table text 추출, sheet/table 단위 chunk 지원 |
| MD | heading 구조, code block, table 구조 보존 |

- 업로드 API는 원본 파일 저장과 metadata 저장을 완료한 뒤 heavy processing을 기다리지 않고 file_id, document_id, pipeline_job_id, queued status를 반환한다.

- 파일 업로드 직후 사용자는 Job Monitor 또는 Document Detail 화면에서 처리 진행 상태를 확인할 수 있어야 한다.

- 파일 metadata는 original_file_name, file_ext, mime_type, file_size_bytes, sha256_checksum, storage_path, uploaded_by, uploaded_at 같은 공통 metadata와 page_count, slide_count, sheet_names, document_properties, detected_language, encrypted/password-protected 여부 같은 format-specific metadata를 분리하여 저장한다.

- parser/extractor는 확장자만 신뢰하지 않고 MIME type, 파일 signature, 내부 manifest를 함께 확인하여 detected_file_type과 file_type_confidence를 기록한다.

- 모든 추출 작업은 extraction profile 또는 extractor runtime metadata를 남겨야 하며, extractor_name, extractor_version, provider_mode(local/remote), options, elapsed_ms, warning_count, error_count를 기록한다.

- 모든 파일 형식은 1차 추출 결과로 normalized markdown artifact를 생성한다. 원본 파일이 Markdown인 경우에도 normalized markdown artifact를 별도로 저장하여 parser version과 normalization rule 변경을 추적한다.

- normalized markdown artifact는 검색용 chunk 생성의 기준 입력이며, 원본 파일 재처리 없이 chunk policy만 바꾸는 실험에 재사용할 수 있어야 한다.

- PDF/DOCX/HWPX/PPTX/XLSX에서 추출된 table과 image는 당장 독립 검색 대상으로 사용하지 않더라도 document block과 source anchor를 통해 추적 가능해야 한다.

- text가 없는 PDF page, password-protected file, 깨진 archive, 추출 누락 table/image 등은 parsing 실패 또는 warning artifact로 보존한다.

## 4.5 검색 요구사항

- 검색 화면은 기본적으로 4개 embedding profile 전체 비교 모드로 동작하되, BM25 keyword baseline과 hybrid retrieval을 추가 선택할 수 있어야 한다.

- 검색 조건에는 query, 문서 그룹, 파일 타입, top-k, chunk policy, retrieval strategy, similarity/scoring metric을 포함한다.

- 초기 embedding-only 검색의 similarity metric은 cosine distance를 기본값으로 한다.

- BM25 baseline은 embedding provider, embedding job, vector table을 사용하지 않고 chunks.chunk_text 기반 keyword index와 corpus statistics만 사용한다.

- BM25 scoring은 Okapi BM25 기본 파라미터 k1=1.2, b=0.75를 시작점으로 하며, 후속 실험에서 tokenizer, stopword, k1, b 값을 search runtime metadata로 기록한다.

- 한국어 문서는 공백/기호 기반 tokenization만으로 형태소 품질이 제한될 수 있으므로, BM25 tokenizer는 여러 strategy를 side-by-side로 비교할 수 있어야 한다.

- 기본 BM25 tokenizer는 `unicode_word_v1`로 유지한다. 이는 외부 의존성이 없고 재현성이 높지만, 조사/어미/복합명사/띄어쓰기 변형에 취약할 수 있다.

- 한국어 친화 내장 baseline으로 `unicode_word_ko_2_3gram_v1`를 제공한다. 이 tokenizer는 Unicode word token에 더해 한글 연속 음절에서 2-gram/3-gram term을 생성하여 `업무보고서는`과 `업무 보고서` 같은 변형의 recall을 실험한다.

- `unicode_word_ko_2_3gram_v1`는 형태소 분석기가 아니며, term 수 증가와 noise 증가 가능성이 있다. 따라서 운영 기본값으로 즉시 전환하지 않고 BM25/golden question 비교 실험에서 Recall@k, MRR, nDCG, index size, query latency를 함께 평가한다.

- KoNLPy 계열 tokenizer는 라이선스 및 JVM/JPype/runtime 의존성 부담 때문에 기본 배포 대상에서 보류한다.

- Mecab-ko는 KoNLPy 대안의 optional 형태소 analyzer로 제공한다. `mecab_ko_morph_v1` tokenizer는 `mecab-ko-python`과 `mecab-ko-dic`가 설치된 환경에서만 활성화되며, 미설치 또는 dictionary 초기화 실패 시 UI와 API metadata에 unavailable 상태를 표시한다.

- `mecab_ko_morph_v1`는 운영 기본값이 아니라 한국어 BM25 품질 실험 tokenizer로 관리한다. 형태소 tokenizer 기반 keyword index는 tokenizer별로 별도 backfill해야 하며, Search Compare와 BM25 coverage 화면에서 tokenizer별 readiness를 확인한 뒤 비교 실험에 사용한다.

- Search Compare 화면은 BM25 Keyword profile 선택 시 BM25 tokenizer를 선택할 수 있어야 한다. 선택한 tokenizer는 BM25-only 검색, chunk policy compare, permission matrix 실험에 동일하게 적용한다.

- 선택한 BM25 tokenizer와 tokenizer별 index 준비 여부는 search runtime metadata와 BM25 coverage/freshness 화면을 통해 재현 가능해야 한다.

- BM25 coverage 화면은 선택한 tokenizer와 chunk policy 기준으로 term index와 corpus statistics를 backfill할 수 있는 운영 제어를 제공한다. Backfill 결과는 성공/실패 정책 수, tokenizer name, 실행 범위를 feedback으로 표시하고 API payload로도 제공한다.

- Hybrid retrieval은 1차 MVP에서 BM25 top-N과 vector top-N을 Reciprocal Rank Fusion(RRF)으로 결합한다.
- Hybrid retrieval의 RRF 결과는 vector rank, keyword rank, source별 원 score, RRF 기여도, rrf_k 값을 score_components에 저장해 Search Log와 실험 결과에서 재현 가능해야 한다.
- Search Compare API는 `hybrid_keyword_vector` profile을 실행할 때 hybrid_vector_profile_name을 명시하거나 선택된 embedding profile 중 첫 번째 active profile을 vector side로 사용하며, 후보 top-N과 RRF 병합 결과를 검색 로그에 기록해야 한다.
- Search Compare UI는 Hybrid RRF profile을 명시적으로 선택할 수 있어야 하며, 선택 시 BM25 tokenizer와 결합 대상 embedding profile을 함께 지정할 수 있어야 한다.
- Reranking은 1차로 검색 후보 chunk text를 query와 함께 reranker provider에 전달해 rerank score/rank를 받는 독립 계약으로 정의한다. Qwen3-Reranker-4B는 remote provider 후보로 관리하되, 개발/테스트 환경에서는 deterministic mock lexical-overlap reranker로 API와 로그 계약을 먼저 검증할 수 있어야 한다.
- Reranked search result는 원본 vector/BM25/hybrid 후보의 source profile, retrieval strategy, source rank, source score, source score_components를 보존하고, reranker profile/model/provider와 rerank score를 별도 score_components로 기록해야 한다.
- Search Compare API는 `reranked_vector_cosine` profile을 실행할 때 reranked_vector_profile_name을 명시하거나 선택된 embedding profile 중 첫 번째 active profile을 source vector 후보 검색에 사용하며, source vector profile, candidate_top_k, reranker profile/model/provider를 search runtime metadata와 result score_components에 기록해야 한다.
- remote reranker provider는 최소 `GET /healthz`, `POST /v1/rerank` HTTP 계약을 제공해야 하며, NeX_PCX는 `NEX_PCX_RERANKER_PROVIDER_MODE`, `NEX_PCX_REMOTE_RERANKER_PROVIDER_URL`, `NEX_PCX_REMOTE_RERANKER_PROVIDER_TIMEOUT_SECONDS` 설정으로 mock/remote provider를 전환할 수 있어야 한다.
- remote reranker 호출 결과의 provider runtime mode, base URL, timeout, provider_type, model_id, reranker runtime metadata는 Search Compare, Search Experiment, Golden Question 기반 검색 실험 로그에 재현 가능하게 기록되어야 한다.
- Search Compare UI는 `reranked_vector_cosine` profile을 명시적으로 선택할 수 있어야 하며, 선택 시 1차 후보 검색에 사용할 source vector profile과 현재 reranker runtime mode/base URL/timeout/profile/model 상태를 함께 표시해야 한다.
- Search Compare UI는 reranked profile 결과에서 source vector profile, candidate_top_k, candidate_count, reranker provider/profile/backend/device/latency metadata를 profile별 runtime panel에 표시해 remote/mock reranker 실행 여부를 사용자가 확인할 수 있어야 한다.
- Qwen3-Reranker-4B remote runtime provider는 `sentence-transformers` CrossEncoder backend를 통해 query/document pair score를 계산하고, score 내림차순과 source rank tie-breaker 기준으로 top-k rerank 결과를 반환해야 한다.
- reranker runtime service는 `NEX_PCX_RERANKER_PROVIDER_BACKEND`, `NEX_PCX_RERANKER_PROVIDER_MODELS_DIR`, `NEX_PCX_RERANKER_PROVIDER_MODEL_DIR_NAME`, `NEX_PCX_RERANKER_PROVIDER_DEVICE`, `NEX_PCX_RERANKER_PROVIDER_READY` 설정으로 DGX 환경의 모델 경로, device, readiness를 제어할 수 있어야 한다.
- Qwen3-Reranker-4B remote provider는 embedding provider 기본 포트 9101~9103과 충돌하지 않도록 기본 포트 `9104`를 사용하며, foreground launch script와 health smoke runner를 통해 DGX에서 `/healthz` 계약을 검증할 수 있어야 한다.
- remote reranker foreground smoke는 실행 전 기존 health endpoint 활성 여부를 확인하고, 실행 후 provider_type, provider_model_id, reranker_profile_name, device, backend, model_dir readiness가 기대값과 일치하는지 검증한 뒤 process stop과 health endpoint 종료까지 확인해야 한다.
- remote reranker request smoke는 `/v1/rerank`에 query와 후보 chunk text를 전달하여 provider_type, reranker_model_id, reranker_profile_name, retrieval_strategy, candidate_count, returned_count, top_k, rank sequence, finite score, runtime_metadata(service/backend/device)를 검증하고, score/rank preview를 markdown evidence로 남길 수 있어야 한다.
- Search Compare remote reranker E2E smoke는 `reranked_vector_cosine` profile 실행 시 query embedding, vector 후보 검색, remote reranker 호출, reranked result 생성, search log 저장, profile runtime metadata 기록이 하나의 흐름으로 동작하는지 검증하고, search_log_id, result score preview, provider_runtime_mode/base_url/timeout, reranker runtime metadata를 evidence로 남길 수 있어야 한다.
- 운영 전 DGX remote reranker 검증 증적은 service source 배치 상태, `/healthz` startup attempts, model shard load, `/v1/rerank` request/provider latency, finite score preview, runtime metadata, foreground shutdown 및 post-stop port close 상태를 함께 기록해야 한다.
- DGX remote reranker Search Compare live E2E 검증은 target DB가 `reranked_vector_cosine` search profile seed를 포함하는 migration revision 이상인지 확인하고, query embedding provider route, reranker provider URL, search_log_id, candidate_count, reranked result score, source vector score/rank, source_query_runtime_metadata를 운영 증적으로 남겨야 한다.
- remote reranker background lifecycle runner는 `start`, `status`, `stop`, `smoke` 동작을 제공하고, 이미 실행 중인 provider 중복 실행을 방지하며, PID file, log file, `/healthz`, `/v1/rerank` request smoke 결과를 JSON/Markdown evidence로 남길 수 있어야 한다.
- Remote reranker operations status API/UI는 운영자가 web admin 화면에서 process running 여부, PID/log 경로, `/healthz` readiness, provider model/profile/backend/device 계약 일치 여부, 선택적 `/v1/rerank` request smoke 결과를 읽기 전용으로 확인할 수 있어야 한다. 또한 앱 runtime 설정의 `NEX_PCX_RERANKER_PROVIDER_MODE`, remote URL, timeout 값을 함께 표시하여 provider가 살아 있어도 Search Compare가 mock/runtime misconfigured 상태로 동작하는 상황을 구분할 수 있어야 한다.

- 검색 결과에는 rank, score/distance, 문서명, page/slide/sheet, heading_path, chunk preview, 피드백 버튼을 포함한다.

- 서로 다른 search profile의 score는 직접 비교하지 않고, 정답 chunk 순위 및 top-k 포함 여부를 중심으로 비교한다.

- 생성 단계로 전달되는 retrieval context package는 search_log_id를 기준으로 query, actor, requested/effective permission scope, selected profiles, chunk policy, runtime metadata를 포함해야 한다.

- Retrieval context package는 reranked/hybrid/BM25/vector profile 결과를 우선순위에 따라 정렬하고, 동일 chunk가 여러 profile에서 발견되면 하나의 citation으로 deduplicate하되 supporting result metadata를 보존해야 한다.

- Retrieval context package는 포함된 chunk text, 선택적 prev/next neighbor context, document/file metadata, page/slide/sheet/cell 위치, heading_path, artifact_id/block_id/source_anchor, context character budget, truncation/exclusion reason을 API와 UI에서 확인할 수 있어야 한다.

- Citation readiness check는 retrieval context package 후보별로 citation key, document/file identity, chunk id/policy, generation text, source_anchor, page/slide/sheet/cell 또는 heading_path 위치 힌트, artifact_id/block_id lineage reference를 점검해야 한다.

- Citation readiness 결과는 ready/warning/failed 상태, source anchor coverage percent, citation ready percent, 후보별 issue code를 API와 UI에서 제공하여 생성 단계 진입 전 근거 품질 gate로 사용할 수 있어야 한다.

- Generation provider runtime은 embedding/reranker remote provider처럼 NeX_PCX 내부에 별도 heavy model wrapper를 만들기보다, 1차로 vLLM이 제공하는 OpenAI-compatible HTTP runtime을 직접 호출한다. NeX_PCX backend는 provider 설정, prompt packaging, retrieval/citation/confidence gate, 실행 로그, 응답 metadata 저장에 집중한다.

- Remote vLLM 호출의 기본 endpoint는 `/v1/chat/completions`로 하며, request에는 model, messages, max_tokens, temperature, top_p, optional stop sequence, trace metadata를 포함한다. response에서는 answer text, finish_reason, token usage, model id, provider latency, raw response summary를 generation run metadata로 저장할 수 있어야 한다.

- OpenAI-compatible vLLM client는 transport error, invalid JSON, non-object JSON, HTTP error response를 구분해 표준 provider metrics와 error payload를 보존해야 하며, DGX-Spark 연결 전에도 mock HTTP transport 기반으로 request/response 계약을 검증할 수 있어야 한다.

- DGX vLLM smoke runner는 API key 값을 코드, CLI dry-run output, markdown/json 증적에 기록하지 않고 환경변수 이름과 configured 여부만 남겨야 한다. 성공 증적은 HTTP 2xx, non-empty answer, finish reason, token usage, provider metrics succeeded를 기준으로 판단한다. Qwen smoke request는 기본적으로 `chat_template_kwargs.enable_thinking=false`를 사용해 짧은 smoke token budget에서도 최종 assistant content가 반환되도록 한다.

- 기본 remote LLM 후보는 `nvidia/Qwen3.6-27B-NVFP4`로 두되, 로컬 개발 환경과 remote DGX-Spark 접속이 불가능한 환경에서는 deterministic mock provider가 동일한 request/response shape로 동작해야 한다.

- Prompt package는 system instruction, user query, retrieval context blocks, citation rules, no-answer rule, response language preference를 분리해 구성하고, prompt version과 rendered prompt hash를 generation run에 기록해야 한다.

- `low_confidence`, `no_relevant_context`, citation readiness `failed` 상태의 retrieval package에 대해서는 LLM에 임의 context를 전달하지 않아야 한다. UI/API는 no-answer 또는 blocked status와 reason code를 사용자에게 표시해야 한다.

## 4.6 Chunk policy 요구사항

- MVP 기본 정책은 heading-aware chunking으로 한다.

- chunk size와 overlap은 검색 정확도, 저장용량, embedding 비용, hallucination 위험에 영향을 주므로 고정 상수가 아니라 chunk_policies table에 저장되는 실험 정책으로 관리한다.

- 운영 기본 정책은 하나로 유지하여 관리 복잡도를 낮추되, NeX_PCX 실험 단계에서는 2~3개 정책을 같은 fixture/golden question set으로 비교할 수 있어야 한다.

- 기본 chunk target size는 512 tokens, overlap은 64 tokens로 시작한다. 추가 실험 후보로 1000/200, 1500/200 계열 정책을 seed하여 짧은 사실 검색과 긴 문맥 검색의 trade-off를 비교한다.

- Heading 경계가 있는 경우 heading hierarchy를 우선 보존하고, heading 하나의 본문이 target size를 초과하면 paragraph/table/code block 경계를 우선하여 분할한다.

- Markdown code block과 table, DOCX/HWPX/PPTX/XLSX table은 가능한 한 하나의 chunk 안에서 보존한다. token size는 구조 경계를 무시하는 절대 길이가 아니라 구조 보존 후 분할 여부를 결정하는 상한선으로 사용한다. target size를 크게 초과하는 table은 row group 단위로 분할한다.

- 각 chunk는 chunk_policy_name, parser_name, parser_version, token_count, char_count, content_hash를 저장하여 regression test와 검색 재현성에 사용할 수 있어야 한다.

- chunk는 text/table/image/figure/code/list/heading 등 chunk_type을 가져야 한다. MVP 검색은 text embedding 중심으로 수행하되, table/image 기반 chunk도 검색용 텍스트 표현을 가진 공통 chunk contract를 따른다.

- chunk는 artifact_id와 block_id를 저장하여 어떤 normalized markdown artifact와 document block에서 생성되었는지 역추적할 수 있어야 한다.

- chunk는 source_anchor JSON을 저장하여 page_no, slide_no, sheet_name, cell_range, char_start/end, row_range, geometry 등 원본 위치를 표현할 수 있어야 한다.

- prev_chunk_id와 next_chunk_id는 같은 document_id와 chunk_policy_name 범위 안에서 연결한다. 검색 결과에서 주변 문맥을 확장할 때 권한 필터를 통과한 같은 문서의 앞/뒤 chunk만 사용할 수 있다.

- table block에서 생성된 chunk는 table_artifact_id 또는 metadata reference를 통해 원본 table 구조를 찾을 수 있어야 하며, target size 초과 시 row group 단위로 분할한다.

- image block에서 생성된 chunk는 OCR text, caption text, surrounding text 중 하나 이상의 검색용 텍스트를 사용할 수 있어야 한다. 이미지 embedding은 MVP 범위 밖이지만 source artifact는 보존한다.

## 4.7 Embedding job lifecycle 요구사항

- 파일 parsing과 chunk 생성이 완료되면 활성화된 embedding profile별로 chunk/profile 조합의 job을 생성한다.

- embedding model은 worker 실행 중 자동 다운로드하지 않고, `NEX_PCX_MODELS_DIR` 또는 동등한 설정이 가리키는 local model bundle에서 로딩한다.

- 개발자는 별도 download script를 실행하여 `models/kure_v1`, `models/bge_m3`, `models/qwen3_embedding_4b` 디렉토리를 구성할 수 있어야 한다.

- 고객사 또는 폐쇄망 설치 시에는 사전 다운로드된 model bundle을 전달하고, 설치 후 manifest/revision/local path 검증을 수행해야 한다.

- `qwen3_4b_1000`과 `qwen3_4b_2560` profile은 동일한 `Qwen/Qwen3-Embedding-4B` model directory를 공유하되 output dimension과 storage type 정책을 profile metadata로 분리한다.

- embedding worker는 profile별 provider 설정을 해석하여 mock, local sentence-transformers, remote HTTP provider 중 하나를 호출할 수 있어야 한다.

- remote provider request에는 profile_name, model_key, input_type(query/document), texts, normalize_embeddings, output_dimension, trace_id를 포함한다.

- remote provider response에는 embeddings, dimension, provider_model_id, provider_type, elapsed_ms, token_count 또는 input_count, runtime_metadata를 포함한다.

- provider 호출 실패, dimension mismatch, non-finite vector, timeout은 embedding job 실패로 기록하고 error_code와 error_message를 보존한다.

- job status는 pending, running, succeeded, failed, skipped 중 하나로 관리한다.

- worker는 job을 lease 방식으로 점유하고, lease 만료 시 다른 worker가 재시도할 수 있어야 한다.

- 실패한 job은 attempts, max_attempts, error_message, last_error_at을 기록한다.

- 동일 chunk/profile 조합에 대해 중복 job이 생성되지 않도록 unique constraint를 둔다.

- succeeded job은 해당 profile의 embedding table에 vector 또는 halfvec 저장이 완료된 상태를 의미한다.

## 4.8 계정, 권한 및 검색 범위 요구사항

- NeX_PCX는 운영 인증 시스템을 직접 구현하지 않고, 권한 실험을 위한 테스트 계정과 조직 seed를 제공한다.

- 사용자 계정은 active/inactive 상태를 가지며, 비활성 사용자는 업로드와 검색 actor로 선택할 수 없다.

- 조직은 parent-child 구조를 가져야 하며, 팀장/그룹장 역할은 하위 조직 범위를 계산할 수 있어야 한다.

- 사용자 소속은 하나 이상의 조직 membership으로 표현하며, membership role은 member, team_lead, group_lead, admin 중 하나로 시작한다.

- 문서 업로드 시 uploaded_by_user_id, owner_user_id, owner_org_unit_id, access_scope를 저장한다.

- access_scope 기본값은 personal이며, 지원 값은 personal, team, org_tree, company로 한다.

- personal 문서는 소유자와 managerial chain의 상위 역할자가 검색할 수 있다.

- team 문서는 소유 조직 구성원과 해당 조직의 상위 역할자가 검색할 수 있다.

- org_tree 문서는 소유 조직과 하위 조직 구성원, 그리고 상위 역할자가 검색할 수 있다.

- company 문서는 활성 사용자 전체가 검색할 수 있다.

- 검색 화면/API는 actor_user_id와 requested_search_scope를 받아 effective permission filter를 산출한다.

- requested_search_scope는 mine, team, managed_org, company 중 하나로 시작한다.

- 권한 필터는 chunk 후보군 생성 단계에서 document/file metadata와 join하여 적용한다.

- 검색 로그는 actor_user_id, requested_search_scope, effective_scope, permission_filter_metadata를 저장한다.

- 권한이 다른 actor가 같은 query를 실행하면 별도 검색 실험으로 기록하고, 평가 지표도 actor/scope별로 분리한다.

- 권한 필터로 제외된 문서의 내용, 제목, chunk preview, score는 UI와 API 응답에 노출하지 않는다.

## 4.9 비동기 문서 처리 파이프라인 요구사항

- 문서 처리 파이프라인은 upload_saved, text_extraction, parsing, chunking, embedding, vector_indexing, completed 단계로 추적한다.

- pipeline job status는 queued, running, succeeded, failed, canceled, skipped 중 하나로 관리한다.

- 업로드 요청 transaction은 원본 파일, files/documents metadata, root pipeline job 생성을 원자적으로 처리한다.

- worker는 `FOR UPDATE SKIP LOCKED` 방식의 claim query로 queued job을 점유하고 lease_owner, lease_expires_at, heartbeat_at을 갱신한다.

- lease가 만료된 running job은 재점유 가능해야 하며, 재점유 시 attempts와 event log를 남긴다.

- 초기 운영은 단일 worker와 순차 처리로 시작하되, NEX_PCX_WORKER_CONCURRENCY 또는 동등한 설정값으로 worker 동시성을 늘릴 수 있어야 한다.

- Qwen3-Embedding-4B 1000/2560 profile은 처리 시간이 길 수 있으므로 별도 worker pool로 분리 가능해야 한다.

- GPU provider를 사용하는 경우 worker pool은 model을 직접 preload하지 않고 provider health check와 request timeout, retry/backoff 정책을 관리한다.

- pipeline 진행률은 total_units, processed_units, progress_percent, current_stage, current_message로 표현한다.

- chunk/profile 단위 embedding 작업은 embedding_jobs로 세부 추적하고, 상위 pipeline job은 문서 단위 전체 진행률을 집계한다.

- 중복 checksum 파일은 기존 files/documents metadata를 재사용하거나 별도 정책에 따라 새 document record만 생성하되, 동일 chunk/profile embedding job 중복 생성은 방지한다.

- 실패한 job은 error_code, error_message, failed_at, attempts, max_attempts를 저장하고, 상세 stack trace 또는 내부 오류 상세는 app_logs와 연결한다.

- 관리자 또는 권한 있는 actor는 failed/canceled job을 수동 retry할 수 있어야 한다.

- 사용자는 본인이 업로드했거나 권한 범위에 포함되는 문서의 pipeline 상태만 조회할 수 있고, System Admin은 전체 queue를 조회할 수 있다.

# 5. 데이터 및 저장소 요구사항

## 5.1 저장소 원칙

- 원본 파일 metadata, 문서 metadata, chunk metadata, embedding profile, 검색 로그, 피드백은 모두 PostgreSQL에 저장한다.

- 사용자, 조직, 역할, 문서 접근 범위, 검색 actor/scope metadata도 PostgreSQL에 저장한다.

- 비동기 파이프라인 queue, worker lease, 진행률, 재시도, 이벤트 이력도 PostgreSQL에 저장한다.

- 원본 파일은 파일 시스템 또는 object storage에 저장하고, DB에는 storage_path와 checksum을 저장한다.

- 추출된 normalized markdown, parser metadata, warning/error report, table/image 산출물은 파일 시스템 또는 object storage에 저장하고, DB에는 artifact metadata와 storage_path 또는 content hash를 저장한다.

- chunk는 원본 파일을 직접 참조하기보다 documents, extraction_artifacts, document_blocks를 통해 원본 위치와 추출 경로를 추적한다.

- Embedding은 pgvector의 vector 또는 halfvec type으로 저장한다.

- KURE-v1 1024, bge-m3 1024, Qwen3 1000 profile은 vector type을 사용한다.

- Qwen3 2560 profile은 일반 vector type 차원 한계를 초과하므로 halfvec(2560)을 사용한다.

- Dimension이 다른 profile을 단순하게 관리하기 위해 MVP에서는 profile별 embedding table 분리 방식을 채택한다.

- MVP의 기본 검색은 cosine distance를 사용한다. 데이터가 5만 chunk 이하인 초기 실험에서는 exact search를 우선하고, 이후 HNSW 또는 IVFFlat index를 profile별로 추가한다.

- BM25 keyword baseline은 chunk text token index와 corpus statistics를 PostgreSQL에 저장하며, embedding vector와 독립적으로 재생성할 수 있어야 한다.

- BM25 keyword index는 tokenizer별로 독립적으로 재생성 가능해야 한다. 운영자는 `unicode_word_v1`, `unicode_word_ko_2_3gram_v1`, `mecab_ko_morph_v1` 등 선택 가능한 tokenizer마다 chunk policy 단위 backfill을 실행하고 coverage/readiness를 확인할 수 있어야 한다.

- Search log 결과 row는 embedding profile뿐 아니라 BM25 keyword baseline과 hybrid strategy도 저장할 수 있도록 search_profile 또는 retrieval_strategy 식별자를 가져야 한다.

- 권한 조건은 documents/files metadata와 join 가능한 정규화된 컬럼으로 저장하고, 검색 후보군 생성 단계에서 pre-filter로 적용한다.

- DB schema 변경 시 migration과 migration regression test를 추가한다.

## 5.2 핵심 테이블

| 테이블 | 설명 |
| --- | --- |
| app_users | 권한 실험용 사용자 계정 |
| org_units | 팀, 그룹, 본부 등 계층형 조직 단위 |
| user_org_memberships | 사용자-조직 소속과 역할 |
| files | 업로드 원본 파일 metadata 및 parsing 상태 |
| documents | 논리 문서 단위 정보 |
| extraction_profiles | extractor/provider 이름, 버전, option, local/remote 실행 정책 |
| extraction_runs | 파일별 추출 실행 이력, 상태, runtime metadata, warning/error 요약 |
| extraction_artifacts | normalized markdown, plain text, parser metadata, warning/error report 등 추출 산출물 |
| document_blocks | chunk 생성 이전의 heading/paragraph/table/image/list/code/page/slide/sheet 구조 block |
| table_artifacts | table block의 markdown/json/csv 구조 산출물 |
| image_artifacts | image/figure block의 파일 위치, OCR/caption, source 위치 metadata |
| chunks | 검색 단위 chunk와 구조 metadata |
| chunk_policies | chunk size, overlap, split strategy 등 chunk 생성 정책 |
| pipeline_jobs | 문서 처리 파이프라인의 상위 job, stage, status, lease, 진행률 |
| pipeline_job_events | pipeline job 상태 변경과 진행률 변경 이력 |
| embedding_profiles | 모델명, dimension, storage type, runtime 설정 등 profile metadata |
| search_profiles | 검색 비교용 profile/lane metadata. embedding profile, BM25 baseline, hybrid, reranked strategy를 포괄 |
| embedding_jobs | chunk/profile별 embedding 작업 상태와 재시도 정보 |
| chunk_embeddings_kure_v1_1024 | KURE-v1 1024차원 embedding 저장 |
| chunk_embeddings_bge_m3_1024 | bge-m3 1024차원 embedding 저장 |
| chunk_embeddings_qwen3_4b_1000 | Qwen3 1000차원 embedding 저장 |
| chunk_embeddings_qwen3_4b_2560 | Qwen3 2560차원 halfvec embedding 저장 |
| chunk_keyword_terms | BM25 keyword baseline용 chunk별 term frequency index |
| chunk_keyword_statistics | BM25 keyword baseline용 corpus statistics 또는 materialized summary |
| search_logs | 검색 query, 조건, profile별 latency 로그 |
| search_log_results | 검색 로그별 profile/rank/chunk 결과 |
| search_result_feedback | 검색 결과별 정답/부분정답/오답 피드백 |
| generation_provider_configs | mock/vLLM 등 생성 provider 설정, model id, endpoint, timeout, decoding option |
| generation_runs | search_log/retrieval package/prompt/provider 조합별 생성 실행 이력, status, answer, token usage, latency, guardrail metadata |
| generation_run_citations | 생성 답변에서 참조한 retrieval context citation key와 chunk/source anchor lineage |

## 5.3 계정/조직/역할 테이블 요구사항

```sql
CREATE TABLE app_users (
user_id BIGSERIAL PRIMARY KEY,
login_id TEXT NOT NULL UNIQUE,
display_name TEXT NOT NULL,
email TEXT,
is_active BOOLEAN DEFAULT true,
metadata JSONB DEFAULT '{}'::jsonb,
created_at TIMESTAMP DEFAULT now(),
updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE org_units (
org_unit_id BIGSERIAL PRIMARY KEY,
parent_org_unit_id BIGINT REFERENCES org_units(org_unit_id),
org_unit_name TEXT NOT NULL,
org_unit_type TEXT NOT NULL
    CHECK (org_unit_type IN ('company', 'division', 'group', 'team')),
is_active BOOLEAN DEFAULT true,
metadata JSONB DEFAULT '{}'::jsonb,
created_at TIMESTAMP DEFAULT now(),
updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE user_org_memberships (
membership_id BIGSERIAL PRIMARY KEY,
user_id BIGINT NOT NULL REFERENCES app_users(user_id) ON DELETE CASCADE,
org_unit_id BIGINT NOT NULL REFERENCES org_units(org_unit_id) ON DELETE CASCADE,
role_name TEXT NOT NULL
    CHECK (role_name IN ('member', 'team_lead', 'group_lead', 'admin')),
is_primary BOOLEAN DEFAULT false,
created_at TIMESTAMP DEFAULT now(),
updated_at TIMESTAMP DEFAULT now(),
UNIQUE (user_id, org_unit_id, role_name)
);
```

초기 seed는 일반 사용자 2명 이상, 팀장 1명 이상, 그룹장 1명 이상, company 공통 문서 소유 조직을 포함해야 한다.

## 5.4 files 테이블 요구사항

```sql
CREATE TABLE files (
file_id BIGSERIAL PRIMARY KEY,
original_file_name TEXT NOT NULL,
stored_file_name TEXT NOT NULL,
file_ext TEXT,
mime_type TEXT,
detected_file_type TEXT,
file_type_confidence NUMERIC(5, 2),
file_size_bytes BIGINT,
sha256_checksum TEXT UNIQUE,
storage_path TEXT NOT NULL,
document_group TEXT DEFAULT 'default',
security_level TEXT DEFAULT 'internal',
uploaded_by TEXT,
uploaded_by_user_id BIGINT REFERENCES app_users(user_id),
uploaded_at TIMESTAMP DEFAULT now(),
parser_name TEXT,
parser_version TEXT,
parse_status TEXT DEFAULT 'pending'
    CHECK (parse_status IN ('pending', 'running', 'succeeded', 'failed')),
parse_error_message TEXT,
page_count INT,
slide_count INT,
sheet_count INT,
extracted_text_size BIGINT,
common_metadata JSONB DEFAULT '{}'::jsonb,
format_metadata JSONB DEFAULT '{}'::jsonb,
created_at TIMESTAMP DEFAULT now(),
updated_at TIMESTAMP DEFAULT now()
);
```

files.common_metadata는 확장자와 무관한 업로드/보안/소유/저장소 metadata를 보조 저장하는 영역이며, files.format_metadata는 PDF page labels, DOCX core properties, PPTX slide layout, XLSX sheet names/used ranges, HWPX section metadata처럼 파일 형식에 밀접한 metadata를 저장한다. MVP에서는 JSONB로 시작하되, dashboard/filter/search pre-filter에 자주 쓰이는 값은 후속 migration에서 정규화된 컬럼으로 승격한다.

## 5.5 ingestion artifact 및 chunk source contract 테이블 요구사항

```sql
CREATE TABLE extraction_profiles (
extraction_profile_name TEXT PRIMARY KEY,
extractor_name TEXT NOT NULL,
extractor_version TEXT NOT NULL,
provider_mode TEXT NOT NULL DEFAULT 'local'
    CHECK (provider_mode IN ('local', 'remote')),
supported_file_types TEXT[] NOT NULL,
default_options JSONB DEFAULT '{}'::jsonb,
is_active BOOLEAN DEFAULT true,
created_at TIMESTAMP DEFAULT now(),
updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE extraction_runs (
extraction_run_id BIGSERIAL PRIMARY KEY,
file_id BIGINT NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
document_id BIGINT REFERENCES documents(document_id) ON DELETE CASCADE,
extraction_profile_name TEXT REFERENCES extraction_profiles(extraction_profile_name),
status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'skipped')),
provider_mode TEXT NOT NULL DEFAULT 'local'
    CHECK (provider_mode IN ('local', 'remote')),
extractor_name TEXT,
extractor_version TEXT,
started_at TIMESTAMP,
finished_at TIMESTAMP,
elapsed_ms INT,
warning_count INT DEFAULT 0,
error_count INT DEFAULT 0,
error_code TEXT,
error_message TEXT,
runtime_metadata JSONB DEFAULT '{}'::jsonb,
created_at TIMESTAMP DEFAULT now(),
updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE extraction_artifacts (
artifact_id BIGSERIAL PRIMARY KEY,
extraction_run_id BIGINT REFERENCES extraction_runs(extraction_run_id) ON DELETE CASCADE,
file_id BIGINT NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
document_id BIGINT REFERENCES documents(document_id) ON DELETE CASCADE,
artifact_type TEXT NOT NULL
    CHECK (artifact_type IN ('normalized_markdown', 'plain_text', 'parser_metadata', 'warning_report', 'source_snapshot')),
content_text TEXT,
storage_path TEXT,
content_hash TEXT,
size_bytes BIGINT,
language TEXT,
metadata JSONB DEFAULT '{}'::jsonb,
created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE document_blocks (
block_id BIGSERIAL PRIMARY KEY,
artifact_id BIGINT NOT NULL REFERENCES extraction_artifacts(artifact_id) ON DELETE CASCADE,
document_id BIGINT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
parent_block_id BIGINT REFERENCES document_blocks(block_id) ON DELETE CASCADE,
block_seq INT NOT NULL,
block_type TEXT NOT NULL
    CHECK (block_type IN ('document', 'heading', 'paragraph', 'table', 'image', 'figure', 'list', 'code', 'page', 'slide', 'sheet')),
content_text TEXT,
content_markdown TEXT,
heading_path TEXT[],
source_anchor JSONB DEFAULT '{}'::jsonb,
page_no INT,
slide_no INT,
sheet_name TEXT,
cell_range TEXT,
char_start INT,
char_end INT,
token_count INT,
metadata JSONB DEFAULT '{}'::jsonb,
created_at TIMESTAMP DEFAULT now(),
UNIQUE (artifact_id, block_seq)
);

CREATE TABLE table_artifacts (
table_artifact_id BIGSERIAL PRIMARY KEY,
block_id BIGINT NOT NULL REFERENCES document_blocks(block_id) ON DELETE CASCADE,
content_markdown TEXT,
content_json JSONB,
storage_path TEXT,
row_count INT,
column_count INT,
source_anchor JSONB DEFAULT '{}'::jsonb,
metadata JSONB DEFAULT '{}'::jsonb,
created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE image_artifacts (
image_artifact_id BIGSERIAL PRIMARY KEY,
block_id BIGINT NOT NULL REFERENCES document_blocks(block_id) ON DELETE CASCADE,
storage_path TEXT NOT NULL,
mime_type TEXT,
width_px INT,
height_px INT,
ocr_text TEXT,
caption_text TEXT,
surrounding_text TEXT,
source_anchor JSONB DEFAULT '{}'::jsonb,
metadata JSONB DEFAULT '{}'::jsonb,
created_at TIMESTAMP DEFAULT now()
);
```

Extraction provider request는 file_id, storage_path, mime_type, detected_file_type, extraction_profile_name, options, trace_id를 포함한다. response는 normalized markdown, document block list, artifact manifest, warning/error list, source anchor metadata, elapsed_ms, extractor runtime metadata를 포함한다.

실제 migration 생성 순서는 files → documents → extraction_profiles/extraction_runs/extraction_artifacts/document_blocks → chunk_policies/chunks 순서를 따른다. 위 SQL은 ingestion artifact contract를 설명하기 위한 논리 정의이다.

Normalized markdown artifact는 사람이 검토할 수 있는 형태를 우선하며, chunk 재생성, parser regression, 검색 결과 근거 확인의 기준 산출물로 사용한다. table/image는 검색용 text representation을 chunk로 제공하되, 원본 구조와 파일 위치는 table_artifacts/image_artifacts에 별도로 보존한다.

## 5.6 documents/chunks/chunk_policies 테이블 요구사항

```sql
CREATE TABLE documents (
document_id BIGSERIAL PRIMARY KEY,
file_id BIGINT NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
document_title TEXT,
document_group TEXT DEFAULT 'default',
security_level TEXT DEFAULT 'internal',
owner_user_id BIGINT REFERENCES app_users(user_id),
owner_org_unit_id BIGINT REFERENCES org_units(org_unit_id),
access_scope TEXT DEFAULT 'personal'
    CHECK (access_scope IN ('personal', 'team', 'org_tree', 'company')),
document_status TEXT DEFAULT 'active'
    CHECK (document_status IN ('active', 'archived', 'deleted')),
permission_metadata JSONB DEFAULT '{}'::jsonb,
metadata JSONB DEFAULT '{}'::jsonb,
created_at TIMESTAMP DEFAULT now(),
updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE chunk_policies (
chunk_policy_name TEXT PRIMARY KEY,
target_token_size INT NOT NULL,
overlap_token_size INT NOT NULL,
split_strategy TEXT NOT NULL,
preserve_table BOOLEAN DEFAULT true,
preserve_code_block BOOLEAN DEFAULT true,
description TEXT,
created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE chunks (
chunk_id BIGSERIAL PRIMARY KEY,
document_id BIGINT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
artifact_id BIGINT REFERENCES extraction_artifacts(artifact_id) ON DELETE SET NULL,
block_id BIGINT REFERENCES document_blocks(block_id) ON DELETE SET NULL,
chunk_seq INT NOT NULL,
chunk_type TEXT NOT NULL DEFAULT 'text'
    CHECK (chunk_type IN ('text', 'table', 'image', 'figure', 'code', 'list', 'heading')),
chunk_text TEXT NOT NULL,
content_markdown TEXT,
content_hash TEXT NOT NULL,
chunk_policy_name TEXT NOT NULL REFERENCES chunk_policies(chunk_policy_name),
parser_name TEXT,
parser_version TEXT,
heading_path TEXT[],
source_anchor JSONB DEFAULT '{}'::jsonb,
page_no INT,
slide_no INT,
sheet_name TEXT,
cell_range TEXT,
source_char_start INT,
source_char_end INT,
token_count INT,
char_count INT,
prev_chunk_id BIGINT REFERENCES chunks(chunk_id),
next_chunk_id BIGINT REFERENCES chunks(chunk_id),
metadata JSONB DEFAULT '{}'::jsonb,
created_at TIMESTAMP DEFAULT now(),
UNIQUE (document_id, chunk_policy_name, chunk_seq),
UNIQUE (document_id, content_hash, chunk_policy_name)
);
```

chunks는 별도 owner/access 컬럼을 중복 저장하지 않고 documents의 owner_user_id, owner_org_unit_id, access_scope를 상속한다. 검색 SQL은 chunks → documents → files 순서로 join하여 permission pre-filter를 적용하고, 근거 추적이 필요할 때 chunks → document_blocks → extraction_artifacts → files 순서로 source artifact를 조회한다.

초기 chunk policy는 다음 값을 기본값으로 한다.

| chunk_policy_name | target_token_size | overlap_token_size | split_strategy | 비고 |
| --- | --- | --- | --- | --- |
| heading_512_64 | 512 | 64 | heading-aware | MVP 기본 정책 |
| heading_1000_200 | 1000 | 200 | heading-aware | 실무 기본 후보, 문맥/비용 균형 비교 |
| heading_1500_200 | 1500 | 200 | heading-aware | 긴 문맥 문서 후보, 저장량 대비 recall 비교 |

## 5.7 pipeline job queue 테이블 요구사항

```sql
CREATE TABLE pipeline_jobs (
job_id BIGSERIAL PRIMARY KEY,
job_type TEXT NOT NULL
    CHECK (job_type IN ('document_ingestion', 'text_extraction', 'parsing', 'chunking', 'embedding', 'vector_indexing')),
file_id BIGINT REFERENCES files(file_id) ON DELETE CASCADE,
document_id BIGINT REFERENCES documents(document_id) ON DELETE CASCADE,
parent_job_id BIGINT REFERENCES pipeline_jobs(job_id) ON DELETE CASCADE,
requested_by_user_id BIGINT REFERENCES app_users(user_id),
status TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'canceled', 'skipped')),
stage TEXT NOT NULL DEFAULT 'upload_saved'
    CHECK (stage IN ('upload_saved', 'text_extraction', 'parsing', 'chunking', 'embedding', 'vector_indexing', 'completed')),
priority INT NOT NULL DEFAULT 100,
total_units INT DEFAULT 0,
processed_units INT DEFAULT 0,
progress_percent NUMERIC(5, 2) DEFAULT 0,
current_message TEXT,
attempts INT NOT NULL DEFAULT 0,
max_attempts INT NOT NULL DEFAULT 3,
lease_owner TEXT,
lease_expires_at TIMESTAMP,
heartbeat_at TIMESTAMP,
error_code TEXT,
error_message TEXT,
metadata JSONB DEFAULT '{}'::jsonb,
queued_at TIMESTAMP DEFAULT now(),
started_at TIMESTAMP,
finished_at TIMESTAMP,
updated_at TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_pipeline_jobs_claim
ON pipeline_jobs (status, priority, queued_at)
WHERE status = 'queued';

CREATE INDEX idx_pipeline_jobs_file ON pipeline_jobs (file_id);
CREATE INDEX idx_pipeline_jobs_document ON pipeline_jobs (document_id);
CREATE INDEX idx_pipeline_jobs_lease ON pipeline_jobs (lease_expires_at)
WHERE status = 'running';

CREATE TABLE pipeline_job_events (
event_id BIGSERIAL PRIMARY KEY,
job_id BIGINT NOT NULL REFERENCES pipeline_jobs(job_id) ON DELETE CASCADE,
event_type TEXT NOT NULL
    CHECK (event_type IN ('created', 'claimed', 'heartbeat', 'stage_started', 'progress', 'stage_succeeded', 'failed', 'retried', 'canceled')),
stage TEXT,
status TEXT,
message TEXT,
event_metadata JSONB DEFAULT '{}'::jsonb,
created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_pipeline_job_events_job ON pipeline_job_events (job_id, created_at);
```

pipeline_jobs는 문서 단위 ingestion 상태와 UI 진행률을 제공한다. embedding_jobs는 chunk/profile 단위 작업을 세밀하게 추적하며, 상위 pipeline job의 embedding stage는 embedding_jobs의 완료/실패 건수를 집계한다.

MVP의 worker claim query는 다음 원칙을 따른다.

```sql
SELECT job_id
FROM pipeline_jobs
WHERE status = 'queued'
ORDER BY priority ASC, queued_at ASC
FOR UPDATE SKIP LOCKED
LIMIT 1;
```

claim 성공 후 같은 transaction에서 status, lease_owner, lease_expires_at, heartbeat_at을 갱신하고 pipeline_job_events에 claimed event를 기록한다.

## 5.8 embedding profile 및 job 테이블 요구사항

```sql
CREATE TABLE embedding_profiles (
profile_name TEXT PRIMARY KEY,
model_name TEXT NOT NULL,
dimension INT NOT NULL,
storage_type TEXT NOT NULL CHECK (storage_type IN ('vector', 'halfvec')),
max_sequence_length INT,
mvp_max_input_tokens INT,
normalize_embeddings BOOLEAN DEFAULT true,
pooling_strategy TEXT,
query_instruction TEXT,
document_instruction TEXT,
dtype TEXT,
adapter_name TEXT,
is_active BOOLEAN DEFAULT true,
created_at TIMESTAMP DEFAULT now(),
updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE embedding_jobs (
job_id BIGSERIAL PRIMARY KEY,
chunk_id BIGINT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
profile_name TEXT NOT NULL REFERENCES embedding_profiles(profile_name),
status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'skipped')),
attempts INT DEFAULT 0,
max_attempts INT DEFAULT 3,
lease_owner TEXT,
lease_expires_at TIMESTAMP,
error_message TEXT,
last_error_at TIMESTAMP,
created_at TIMESTAMP DEFAULT now(),
started_at TIMESTAMP,
finished_at TIMESTAMP,
updated_at TIMESTAMP DEFAULT now(),
UNIQUE (chunk_id, profile_name)
);
```

embedding profile 초기 데이터는 다음과 같다.

| profile_name | model_name | dimension | storage_type | max_sequence_length | mvp_max_input_tokens | normalize | pooling/query instruction 정책 | 비고 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kure_v1_1024 | nlpai-lab/KURE-v1 | 1024 | vector | 8192 | 8192 | true | sentence-transformers 기본값 | 한국어 특화 retrieval 기준선 |
| bge_m3_1024 | BAAI/bge-m3 | 1024 | vector | 8192 | 8192 | true | dense embedding, query instruction 없음 | multilingual / dense 기준선 |
| qwen3_4b_1000 | Qwen/Qwen3-Embedding-4B | 1000 | vector | 32768 | 8192 | true | query-side instruction 사용 여부를 profile metadata에 기록 | Qwen 저용량 profile |
| qwen3_4b_2560 | Qwen/Qwen3-Embedding-4B | 2560 | halfvec | 32768 | 8192 | true | query-side instruction 사용 여부를 profile metadata에 기록 | Qwen 최대 차원 profile |

## 5.9 profile별 embedding table

```sql
CREATE TABLE chunk_embeddings_kure_v1_1024 (
chunk_id BIGINT PRIMARY KEY REFERENCES chunks(chunk_id),
embedding vector(1024) NOT NULL,
elapsed_ms INT,
created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE chunk_embeddings_bge_m3_1024 (
chunk_id BIGINT PRIMARY KEY REFERENCES chunks(chunk_id),
embedding vector(1024) NOT NULL,
elapsed_ms INT,
created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE chunk_embeddings_qwen3_4b_1000 (
chunk_id BIGINT PRIMARY KEY REFERENCES chunks(chunk_id),
embedding vector(1000) NOT NULL,
elapsed_ms INT,
created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE chunk_embeddings_qwen3_4b_2560 (
chunk_id BIGINT PRIMARY KEY REFERENCES chunks(chunk_id),
embedding halfvec(2560) NOT NULL,
elapsed_ms INT,
created_at TIMESTAMP DEFAULT now()
);
```

## 5.10 search log 및 feedback 테이블 요구사항

```sql
CREATE TABLE search_logs (
search_log_id BIGSERIAL PRIMARY KEY,
query_text TEXT NOT NULL,
normalized_query_text TEXT,
actor_user_id BIGINT REFERENCES app_users(user_id),
requested_search_scope TEXT,
effective_search_scope TEXT,
permission_filter_metadata JSONB DEFAULT '{}'::jsonb,
document_group TEXT,
file_type TEXT,
chunk_policy_name TEXT,
strategy_name TEXT DEFAULT 'vector_cosine',
top_k INT NOT NULL,
similarity_metric TEXT NOT NULL DEFAULT 'cosine',
profiles JSONB NOT NULL,
query_runtime_metadata JSONB DEFAULT '{}'::jsonb,
total_elapsed_ms INT,
created_by TEXT,
created_by_user_id BIGINT REFERENCES app_users(user_id),
created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE search_log_results (
search_log_result_id BIGSERIAL PRIMARY KEY,
search_log_id BIGINT NOT NULL REFERENCES search_logs(search_log_id) ON DELETE CASCADE,
profile_name TEXT NOT NULL,
search_profile_name TEXT,
retrieval_strategy TEXT,
rank INT NOT NULL,
chunk_id BIGINT NOT NULL REFERENCES chunks(chunk_id),
distance DOUBLE PRECISION,
score DOUBLE PRECISION,
score_components JSONB DEFAULT '{}'::jsonb,
profile_elapsed_ms INT,
created_at TIMESTAMP DEFAULT now(),
UNIQUE (search_log_id, profile_name, rank)
);

CREATE TABLE search_result_feedback (
feedback_id BIGSERIAL PRIMARY KEY,
search_log_result_id BIGINT NOT NULL REFERENCES search_log_results(search_log_result_id) ON DELETE CASCADE,
relevance_label TEXT NOT NULL
    CHECK (relevance_label IN ('correct', 'partial', 'wrong', 'duplicate', 'insufficient_context')),
comment TEXT,
created_by TEXT,
created_by_user_id BIGINT REFERENCES app_users(user_id),
created_at TIMESTAMP DEFAULT now()
);
```

permission_filter_metadata에는 actor의 primary org, managed org subtree, 포함된 access_scope 목록, company 문서 포함 여부, filter SQL/parameter fingerprint를 저장한다.

BM25와 hybrid 검색을 지원하기 위해 search_log_results.profile_name은 물리적으로 embedding_profiles에만 종속되지 않아야 한다. BM25-only 결과는 profile_name/search_profile_name을 `bm25_keyword`로 기록하고, hybrid 결과는 `hybrid_keyword_vector` 또는 fusion strategy 이름을 기록한다. score_components에는 BM25 score, vector score, RRF rank contribution, tokenizer name, k1, b 같은 세부 점수 근거를 JSONB로 저장할 수 있어야 한다.

BM25 keyword index는 다음 논리 구조를 따른다.

```sql
CREATE TABLE search_profiles (
search_profile_name TEXT PRIMARY KEY,
profile_kind TEXT NOT NULL
    CHECK (profile_kind IN ('embedding', 'keyword', 'hybrid')),
embedding_profile_name TEXT REFERENCES embedding_profiles(profile_name),
strategy_name TEXT NOT NULL,
display_name TEXT NOT NULL,
is_active BOOLEAN DEFAULT true,
runtime_parameters JSONB DEFAULT '{}'::jsonb,
created_at TIMESTAMP DEFAULT now(),
updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE chunk_keyword_terms (
chunk_id BIGINT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
chunk_policy_name TEXT NOT NULL,
term TEXT NOT NULL,
term_frequency INT NOT NULL CHECK (term_frequency > 0),
created_at TIMESTAMP DEFAULT now(),
PRIMARY KEY (chunk_id, term)
);

CREATE TABLE chunk_keyword_statistics (
chunk_policy_name TEXT NOT NULL,
term TEXT NOT NULL,
document_frequency INT NOT NULL CHECK (document_frequency >= 0),
corpus_chunk_count INT NOT NULL CHECK (corpus_chunk_count >= 0),
average_document_length NUMERIC(12, 4) NOT NULL CHECK (average_document_length >= 0),
updated_at TIMESTAMP DEFAULT now(),
PRIMARY KEY (chunk_policy_name, term)
);
```

초기 BM25 indexer는 chunks.chunk_text에서 토큰을 추출해 chunk_keyword_terms를 재생성할 수 있어야 하며, statistics table은 backfill runner 또는 materialized refresh 절차로 갱신한다.

## 5.11 저장용량 산정

저장 타입 기준 embedding 원시 저장용량은 vector profile은 chunk 수 × dimension × 4 bytes, halfvec profile은 chunk 수 × dimension × 2 bytes로 산정한다. 실제 운영 용량은 PostgreSQL row overhead, pgvector index, chunk text, metadata, 원본 파일, 추출 산출물 용량을 추가로 고려해야 한다.

| Chunk 수 | KURE 1024 | bge-m3 1024 | Qwen 1000 | Qwen 2560 | 합계 |
| --- | --- | --- | --- | --- | --- |
| 100,000 | 391 MB | 391 MB | 381 MB | 488 MB | 약 1.65 GB |
| 1,000,000 | 3.9 GB | 3.9 GB | 3.8 GB | 4.9 GB | 약 16.5 GB |
| 10,000,000 | 39 GB | 39 GB | 38 GB | 49 GB | 약 165 GB |

## 5.12 index 요구사항

- MVP 초기에는 exact search를 기준 결과로 남긴다.

- HNSW index를 추가하는 경우 cosine distance 기준으로 vector profile은 vector_cosine_ops, halfvec profile은 halfvec_cosine_ops를 사용한다.

- approximate index 적용 전후의 Recall@k 차이를 regression fixture로 비교한다.

# 6. Web UI 요구사항

## 6.1 공통 UI 원칙

- Bootstrap 기반의 단순하고 명확한 화면을 제공한다.

- MVP에서는 고급 SPA 구조보다 FastAPI template + Bootstrap 중심의 서버 렌더링 구성을 우선한다.

- 검색 결과 비교 화면에서는 선택된 search profile column의 시각적 정렬을 유지한다. 기본 embedding-only 비교는 4개 embedding profile column을 사용하고, BM25와 hybrid 선택 시 keyword/hybrid column을 같은 결과 카드 패턴으로 표시한다.

- 대시보드와 검색 결과는 개발자/평가자가 실험 결과를 빠르게 판단할 수 있도록 숫자와 상태 중심으로 표현한다.

## 6.2 화면 목록

| 화면 | 주요 구성요소 |
| --- | --- |
| Dashboard | 문서 수, 파일 용량, chunk 수, pipeline/embedding 진행률, queue backlog, 검색 latency, 오류 목록 |
| File Upload | 파일 선택, 업로드 사용자, 소유 조직, 접근 범위, 문서 그룹, 보안 등급, 업로드 결과, queued job link |
| Permission Simulation | 테스트 사용자, 조직 계층, membership role, 문서 접근 범위 fixture 관리 |
| Search Compare | 검색어, actor, 검색 scope, 필터, top-k, retrieval strategy, embedding/BM25/hybrid 결과 비교, 피드백 버튼 |
| Document Detail | 원본 metadata, owner, access scope, pipeline timeline, parsing 결과, chunk 목록, embedding 상태 |
| Job Monitor | pipeline job queue, stage/status/progress, 완료/실패/재처리 상태, worker lease 상태 |
| Ingestion Artifact Preview | normalized markdown, extraction warning/error, document block, chunk source anchor, table/image artifact metadata |

## 6.3 검색 비교 화면 레이아웃

```text
┌────────────────┬────────────────┬──────────────────┬──────────────────┐
│ KURE-v1 1024 │ bge-m3 1024 │ Qwen3 1000 │ Qwen3 2560 │
├────────────────┼────────────────┼──────────────────┼──────────────────┤
│ rank 1 │ rank 1 │ rank 1 │ rank 1 │
│ distance │ distance │ distance │ distance │
│ 문서명/페이지 │ 문서명/페이지 │ 문서명/페이지 │ 문서명/페이지 │
│ heading path │ heading path │ heading path │ heading path │
│ chunk preview │ chunk preview │ chunk preview │ chunk preview │
│ 정답/부분/오답 │ 정답/부분/오답 │ 정답/부분/오답 │ 정답/부분/오답 │
└────────────────┴────────────────┴──────────────────┴──────────────────┘
```

검색 비교 화면 상단에는 actor_user, requested_search_scope, effective permission summary를 표시한다. 권한 필터로 제외된 문서 수는 aggregate 숫자로만 표시할 수 있으며, 제외된 문서의 제목이나 본문 preview는 노출하지 않는다.

## 6.4 Pipeline / Job Monitor 화면 요구사항

- File Upload 완료 화면은 업로드 성공/중복 여부와 함께 root pipeline_job_id, 현재 status, Job Monitor 링크를 표시한다.

- Job Monitor 목록은 파일명, 업로드 actor, access scope, status, current stage, progress_percent, attempts, lease_owner, queued_at, started_at, finished_at, duration, error summary를 표시한다.

- Job Monitor는 queued, running, failed, succeeded, canceled 상태별 필터를 제공한다.

- failed/canceled job에는 권한 있는 actor에게 retry action을 제공한다.

- Document Detail 화면은 upload_saved부터 vector_indexing까지 stage timeline을 표시한다.

- 진행 상태 갱신은 MVP에서 server-rendered page refresh 또는 lightweight polling으로 구현하며, 고급 SPA/WebSocket은 후속 phase에서 검토한다.

- 권한 없는 사용자는 본인이 볼 수 없는 문서의 job 상세, 오류 메시지, 파일명, document title을 볼 수 없다.

# 7. Embedding 및 검색 평가 요구사항

## 7.1 평가 원칙

- 동일 문서, 동일 parser, 동일 chunk, 동일 query, 동일 top-k 조건에서 embedding profile, BM25 keyword baseline, hybrid strategy를 바꾸어 비교한다.

- 권한 실험에서는 동일 query와 동일 embedding profile이라도 actor와 requested_search_scope가 다르면 별도 실험으로 기록한다.

- 서로 다른 search profile의 score 절대값은 직접 비교하지 않는다.

- 주요 비교 기준은 정답 chunk의 rank, top-k 포함 여부, 관련 section 검색 여부, 피드백 분포이다.

- permission-aware Recall@k, permission-aware MRR은 actor가 접근 가능한 정답 chunk 집합을 기준으로 계산한다.

- Qwen3 1000 vs Qwen3 2560 비교를 통해 저장용량/검색품질 trade-off를 측정한다.

- Qwen3 2560 profile은 halfvec 저장에 따른 저장용량 절감과 검색 품질 변화를 함께 측정한다.

- BM25 keyword baseline은 embedding-only 결과가 keyword baseline 대비 실제로 유의미한 개선을 제공하는지 판단하는 기준선으로 사용한다.

- Hybrid retrieval은 BM25-only와 embedding-only 결과를 모두 기록한 뒤 RRF 또는 weighted fusion 결과가 Recall@k/MRR/nDCG를 개선하는지 검증한다.

- bge-m3의 sparse/hybrid 기능은 BM25 baseline과 일반 hybrid MVP 이후 별도 phase에서 실험한다.

- approximate index를 적용하는 경우 exact search 결과를 baseline으로 저장하고 Recall@k 변화량을 함께 기록한다.

## 7.2 검색 품질 지표

| 지표 | 정의 |
| --- | --- |
| Recall@k | 정답 chunk가 top-k 결과에 포함되었는지 여부 |
| MRR | 정답 chunk가 처음 등장한 rank의 reciprocal 평균 |
| nDCG | 정답/부분정답/오답 등 graded relevance를 반영한 순위 품질 |
| Latency | query embedding 생성부터 검색 결과 반환까지 걸린 시간 |
| Storage per chunk | profile별 chunk당 vector 저장용량 |
| Feedback distribution | 정답/부분정답/오답/문맥부족/중복 비율 |
| No-answer robustness | 문서에 없는 질문에 대해 관련 없는 chunk를 과도하게 반환하는 정도 |
| Exact vs ANN delta | exact search 대비 HNSW/IVFFlat 적용 후 Recall@k, latency 변화 |
| Permission-filtered Recall@k | actor의 effective scope 내 정답 chunk가 top-k에 포함되었는지 여부 |
| Scope variance | 동일 query에 대해 mine/team/managed_org/company scope별 결과 차이 |
| BM25 baseline delta | BM25-only 대비 embedding-only 또는 hybrid의 Recall@k/MRR/nDCG 변화 |
| Hybrid uplift | BM25-only와 embedding-only 중 더 좋은 단독 결과 대비 hybrid fusion의 개선폭 |

## 7.3 Golden Question Set 요구사항

- 초기에는 문서 50~100개와 질문 100개 이상의 내부 평가셋을 구성한다.

- 각 질문에는 expected_chunk_id 또는 expected_heading_path를 지정한다.

- 권한 평가용 질문에는 actor_user_id, requested_search_scope, expected_visible_chunk_ids, expected_hidden_chunk_ids를 지정할 수 있다.

- 질문 유형은 단일근거, section 기반, 비교, 문서에 없는 질문, 표/그림 관련 질문으로 분류한다.

- 피드백이 누적되면 golden question set으로 승격하는 절차를 둔다.

## 7.4 검색 재현성 metadata 요구사항

- 모든 검색 로그는 query 원문과 normalized query를 함께 저장한다.

- 검색 로그는 search_profile_name, retrieval_strategy, profile_kind를 저장해야 한다. embedding profile인 경우 profile_name, model_name, dimension, storage_type, normalize_embeddings, pooling_strategy, query_instruction, mvp_max_input_tokens를 함께 저장해야 한다.

- Qwen 계열 profile은 query-side instruction 사용 여부와 instruction text를 반드시 profile metadata 또는 search log에 남긴다.

- 검색 로그는 chunk_policy_name, parser_name, parser_version, similarity_metric/scoring_metric, top_k, index_type, index_parameters를 저장해야 한다.

- BM25 검색 로그는 tokenizer, tokenizer_version, term_filtering rule, k1, b, corpus_chunk_count, average_document_length, statistics_refreshed_at을 기록해야 한다. Search Compare에서 선택한 BM25 tokenizer는 profile runtime metadata와 search log top-level metadata 양쪽에 기록한다.

- 검색 결과는 chunk_id만 저장하지 않고, 결과 chunk의 artifact_id, block_id, chunk_type, source_anchor snapshot을 조회 가능해야 한다.

- normalized markdown artifact hash, extraction profile, extractor version, chunk source contract가 달라지면 같은 query라도 별도 검색 실험으로 간주한다.

- 검색 로그는 actor_user_id, requested_search_scope, effective_search_scope, permission_filter_metadata를 저장해야 한다.

- 검색 결과는 search profile별 rank, chunk_id, distance, score, score_components, profile_elapsed_ms를 개별 row로 저장한다.

- 사용자가 입력한 피드백은 검색 결과 row와 연결하여 저장하고, 같은 chunk라도 검색 실험이 다르면 별도 feedback으로 보존한다.

# 8. 비기능 요구사항

| ID | 구분 | 요구사항 |
| --- | --- | --- |
| NFR-001 | 성능 | 문서 50~100개, chunk 1만~5만 개 수준에서 검색 결과를 실험적으로 비교할 수 있어야 한다. |
| NFR-002 | 성능 | 대시보드 조회는 3초 이내를 목표로 한다. |
| NFR-003 | 확장성 | 새 embedding profile 추가 시 최소한의 adapter/table/index 추가로 실험 가능해야 한다. |
| NFR-004 | 신뢰성 | embedding job 실패 시 실패 원인을 기록하고 재처리할 수 있어야 한다. |
| NFR-005 | 재현성 | 검색 실험 시점의 profile runtime 설정, chunk_policy, top-k, query, query instruction, index 설정, 결과 chunk를 로그로 남긴다. |
| NFR-006 | 보안 | 업로드 파일은 내부 PoC 저장소에 보관하며 보안 등급 metadata를 저장한다. |
| NFR-007 | 유지보수성 | parser, chunker, embedding adapter, repository, API layer를 분리한다. |
| NFR-008 | 관측성 | parsing, embedding, search, feedback 이벤트에 대해 로그를 남긴다. |
| NFR-009 | 데이터 무결성 | chunk/profile job, search result, feedback은 FK와 unique constraint로 중복 및 고아 데이터를 방지한다. |
| NFR-010 | 접근제어 | actor가 접근 권한이 없는 문서, chunk, score, preview는 API/UI/search log result에 노출하지 않는다. |
| NFR-011 | 재현성 | 권한 기반 검색 실험은 actor, 조직 membership snapshot, requested/effective scope, permission filter metadata를 함께 저장한다. |
| NFR-012 | 운영성 | 오래 걸리는 ingestion pipeline은 API request thread를 점유하지 않고 queue와 worker에서 처리한다. |
| NFR-013 | 신뢰성 | worker 중단, lease 만료, 일시적 DB 오류가 발생해도 job은 재시도 가능하고 중복 저장을 방지해야 한다. |
| NFR-014 | 관측성 | pipeline job의 stage 전환, 진행률, worker heartbeat, 실패 원인, 재시도 이력을 UI와 로그로 추적할 수 있어야 한다. |
| NFR-015 | 재현성 | 원본 파일, extraction artifact, document block, chunk, embedding, search result 사이의 lineage를 역추적할 수 있어야 한다. |
| NFR-016 | 확장성 | 추출 기능은 local runtime으로 시작하되 OCR/layout/table/image captioning을 remote extraction provider로 분리할 수 있어야 한다. |
| NFR-017 | 데이터 보존 | table/image는 MVP 검색 대상이 아니더라도 source artifact와 검색용 text representation을 보존하여 후속 실험에 사용할 수 있어야 한다. |

# 9. 소프트웨어 품질관리 및 테스트 요구사항

## 9.1 품질관리 원칙

> **개발 원칙**
>
> 기능을 바로 추가하기보다 기존 코드 구조를 먼저 검토한다. 코드 구조 변경(refactoring)은 기능 추가보다 우선한다. 단위 기능 구현 후 unit/regression/smoke test를 통과해야 GitHub commit/sync를 수행한다.

| ID | 품질 요구사항 |
| --- | --- |
| QA-001 | 기능 추가 전 코드 구조를 검토하고 design note를 작성한다. |
| QA-002 | 중복 코드, 비대한 handler, 산재한 SQL/fixture가 발견되면 refactoring을 우선한다. |
| QA-003 | pytest 기반 unit, regression, smoke testing을 수행한다. |
| QA-004 | PostgreSQL + pgvector 연동 기능은 독립 test database를 초기화한 후 테스트한다. |
| QA-005 | Web UI는 Playwright 기반 자동 테스트를 수행한다. |
| QA-006 | pytest-cov로 statement coverage와 branch coverage를 측정한다. |
| QA-007 | statement coverage 90%, branch coverage 85%를 목표로 한다. |
| QA-008 | boundary value analysis, robustness test, worst case 기반 test case를 설계한다. |
| QA-009 | regression test 통과 후 GitHub commit/sync를 수행한다. |
| QA-010 | pgvector vector/halfvec schema, embedding job status transition, search log 재현성 metadata를 integration/regression test에 포함한다. |
| QA-011 | 권한 fixture는 일반 사용자, 팀장, 그룹장, company 문서를 포함하고 permission pre-filter 결과를 regression test로 검증한다. |
| QA-012 | 같은 query에 대해 actor/scope별 결과 차이를 비교하는 permission-aware search test를 포함한다. |
| QA-013 | pipeline job queue schema, claim query, lease 만료, retry, progress update를 integration test에 포함한다. |
| QA-014 | 동시 worker claim 상황에서 같은 job이 중복 점유되지 않는지 concurrency regression test를 포함한다. |
| QA-015 | 업로드 API는 heavy processing을 기다리지 않고 queued job id를 반환하는 smoke test를 포함한다. |
| QA-016 | extraction artifact, document block, chunk source anchor schema는 migration/integration test에 포함한다. |
| QA-017 | 동일 normalized markdown artifact에서 chunk policy만 바꿔 chunk를 재생성하는 regression test를 포함한다. |
| QA-018 | PDF/DOCX/PPTX/XLSX/HWPX/MD fixture는 최소 하나 이상의 source anchor와 warning/error case를 포함한다. |

## 9.2 테스트 계층

| 테스트 유형 | 목적 | 예시 |
| --- | --- | --- |
| Unit Test | 개별 함수/클래스 검증 | parser, chunker, metadata extractor, embedding adapter |
| Integration Test | DB/pgvector/API/worker 연동 검증 | pgvector extension, pipeline queue claim, vector search SQL, repository transaction |
| Regression Test | 기존 검색/업로드/embedding 동작 유지 확인 | 고정 fixture 기반 결과 순위 검증 |
| Smoke Test | 핵심 실행 흐름의 최소 동작 확인 | 서버 기동, DB 연결, MD 업로드, 검색 |
| E2E/UI Test | 브라우저 기반 사용자 흐름 검증 | 대시보드, 업로드, 검색 비교, 피드백 |

## 9.3 PostgreSQL + pgvector 테스트 DB 초기화 원칙

```text
테스트 실행 절차
1. nex_pcx_test database 생성
2. CREATE EXTENSION vector 실행
3. schema migration 적용
4. fixture data 삽입
5. unit/integration/regression/smoke test 실행
6. 테스트 종료 후 DB drop 또는 truncate

금지사항
- 운영 DB 직접 테스트 금지
- 개발 DB에 fixture 강제 삽입 금지
- 테스트 실패 후 불완전한 DB 상태 방치 금지
```

## 9.4 Test Case 설계 기준

| 기법 | 적용 대상 | 예시 |
| --- | --- | --- |
| Boundary Value Analysis | 파일 크기, chunk size, overlap, top-k, query 길이 | 0 byte, max+1, overlap 100%, top-k 0 |
| Robustness Test | 깨진 파일, 확장자 위조, DB 장애, 모델 로딩 실패 | password PDF, corrupted DOCX, duplicate checksum |
| Worst Case Test | 대용량 문서, 대량 chunk, 동시 업로드, Qwen 2560 검색 | 수백 page PDF, 수천 chunk, index rebuild |
| Permission Matrix Test | actor role, org hierarchy, access_scope, requested_scope 조합 | 개인 문서 숨김, 팀장 scope 포함, company 문서 전체 노출 |
| Concurrency Test | worker claim, lease expiry, retry, duplicate job 방지 | worker 2개 동시 claim, lease 만료 job 재점유 |

## 9.5 테스트 디렉토리 구조

```text
tests/
├─ unit/
│ ├─ test_parsers_pdf.py
│ ├─ test_parsers_docx.py
│ ├─ test_parsers_hwpx.py
│ ├─ test_parsers_pptx.py
│ ├─ test_parsers_xlsx.py
│ ├─ test_extraction_artifacts.py
│ ├─ test_document_blocks.py
│ ├─ test_chunker.py
│ ├─ test_chunk_policies.py
│ ├─ test_embedding_profiles.py
│ ├─ test_embedding_jobs.py
│ ├─ test_permission_scope.py
│ └─ test_file_metadata.py
├─ integration/
│ ├─ test_pgvector_extension.py
│ ├─ test_pgvector_halfvec.py
│ ├─ test_document_repository.py
│ ├─ test_permission_repository.py
│ ├─ test_pipeline_job_repository.py
│ ├─ test_extraction_artifact_repository.py
│ ├─ test_chunk_source_contract_schema.py
│ ├─ test_embedding_repository.py
│ ├─ test_embedding_job_repository.py
│ └─ test_vector_search_profiles.py
├─ api/
│ ├─ test_upload_api.py
│ ├─ test_pipeline_jobs_api.py
│ ├─ test_dashboard_api.py
│ ├─ test_search_api.py
│ └─ test_feedback_api.py
├─ regression/
│ ├─ test_upload_regression.py
│ ├─ test_pipeline_job_regression.py
│ ├─ test_permission_scope_regression.py
│ ├─ test_chunking_regression.py
│ ├─ test_extraction_artifact_regression.py
│ ├─ test_search_ranking_regression.py
│ └─ test_search_reproducibility_regression.py
├─ smoke/
│ ├─ test_app_startup.py
│ ├─ test_db_ready.py
│ ├─ test_upload_to_pipeline_flow.py
│ └─ test_upload_to_search_flow.py
├─ e2e/
│ ├─ test_dashboard_ui.py
│ ├─ test_file_upload_ui.py
│ ├─ test_job_monitor_ui.py
│ └─ test_search_compare_ui.py
├─ fixtures/
│ ├─ files/
│ ├─ permissions/
│ ├─ expected_chunks/
│ ├─ expected_artifacts/
│ ├─ expected_blocks/
│ └─ expected_search_results/
└─ conftest.py
```

## 9.6 Quality Gate 절차

| Gate | 필수 확인 항목 |
| --- | --- |
| Design Gate | 기능 추가 전 구조 검토, design note 작성, refactoring 필요성 판단 |
| Unit Gate | unit test 통과, boundary/robustness/worst case 포함 |
| Integration Gate | 독립 test DB 초기화, pgvector extension, repository/API 연동 검증 |
| Regression Gate | 기존 fixture 기반 검색 순위와 업로드 흐름 유지 확인 |
| Smoke Gate | 앱 기동, DB 연결, 업로드-embedding-검색 핵심 흐름 확인 |
| Coverage Gate | statement 90%, branch 85% 목표 확인 |
| UI Gate | Playwright 기반 대시보드/업로드/검색 비교 화면 자동 테스트 |
| GitHub Sync Gate | 모든 gate 통과 후 commit/push/sync |

## 9.7 권장 품질 명령

```bash
ruff check app tests
black --check app tests
pytest --cov=app --cov-branch --cov-report=term-missing --cov-report=json:/tmp/nex_pcx_coverage.json tests/unit tests/smoke tests/integration tests/regression
pytest tests/e2e
```

# 10. 개발 단계 및 인수 기준

| Phase | 목표 | 주요 산출물 | 인수 기준 |
| --- | --- | --- | --- |
| Phase 1 | Skeleton + 품질 기반 구축 | FastAPI/Bootstrap/PostgreSQL/pgvector/pytest/Playwright 기본 구조, migration 골격 | 앱 기동, DB 연결, vector/halfvec extension 확인, test DB 초기화, 기본 smoke test 통과 |
| Phase 2 | 파일 업로드 + metadata 저장 | 업로드 API/UI, files/documents 테이블, checksum 중복 검출, 문서 그룹/보안 등급 metadata | 지원 확장자 업로드와 metadata 저장 test 통과 |
| Phase 2.5 | Identity + Permission Metadata | 테스트 계정, 조직 계층, document access scope, permission-aware search metadata | actor/scope별 접근 가능 문서 fixture와 permission pre-filter test 통과 |
| Phase 2.6 | Pipeline Queue + Worker Foundation | pipeline_jobs/pipeline_job_events, PostgreSQL queue claim, worker lease, 진행 상태 API/UI | 업로드 후 queued job 생성, worker claim/retry/progress integration test 통과 |
| Phase 2.7 | Ingestion Source Contract | extraction_profiles/runs/artifacts, document_blocks, table/image artifact placeholder, chunk source contract | normalized markdown artifact와 block/source anchor를 기준으로 chunk lineage test 통과 |
| Phase 3 | Parsing + Chunking | 파일별 extractor/parser, heading-aware chunker, chunk_policies/chunks 저장 | fixture 문서별 artifact/block/chunk 생성 결과와 chunk policy regression test 통과 |
| Phase 4 | 4개 Embedding Profile 처리 | profile별 worker/table, embedding_jobs 상태 관리, Qwen 2560 halfvec 저장 | 동일 chunk에 대해 4개 profile embedding 저장 및 job status transition 검증 |
| Phase 5 | 검색 비교 UI | embedding-only 검색 화면, search_logs/search_log_results/feedback 저장 | 동일 query에 대한 4개 embedding profile top-k 결과 표시, 재현성 metadata 기록, 피드백 저장 |
| Phase 5.5 | BM25/Hybrid 검색 비교 | BM25 keyword index, BM25 baseline, hybrid fusion, strategy 비교 UI | 동일 query/golden set에서 BM25-only, embedding-only, hybrid top-k 결과와 품질 지표 비교 |
| Phase 6 | 평가 자동화 | golden question set, Recall@k/MRR/nDCG 리포트 | 질문셋 기반 profile별 평가 리포트 생성 |
| Phase 6.5 | Retrieval-to-Generation Handoff | retrieval context package, citation readiness, low-confidence/no-answer guardrail | 검색 결과가 생성 입력으로 전달 가능한지 package/citation/confidence gate 검증 |
| Phase 7 | Grounded Generation MVP | generation run schema, prompt package builder, mock generation provider, vLLM OpenAI-compatible runtime config/client/smoke evidence | mock provider로 생성 실행 로그와 citation trace가 저장되고, DGX vLLM smoke 전환 조건, HTTP client 계약, live connection evidence가 정의됨 |

## 10.1 MVP 완료 기준

- PDF/DOCX/HWPX/PPTX/XLSX/MD 파일 업로드와 원본 metadata 저장이 가능하다.

- 업로드 문서는 actor, owner, owner org, access scope metadata를 가져야 한다.

- 업로드 API는 heavy processing을 기다리지 않고 pipeline job을 queued 상태로 생성해야 한다.

- 사용자는 업로드한 문서의 pipeline stage, status, progress, 실패 원인을 화면에서 확인할 수 있어야 한다.

- 파일별 parser가 최소 fixture 문서에 대해 정상적으로 text를 추출한다.

- 모든 업로드 문서는 normalized markdown artifact를 생성하고, chunk는 artifact_id/block_id/source_anchor를 통해 원본 근거를 역추적할 수 있어야 한다.

- table/image는 MVP에서 직접 vector embedding하지 않더라도 검색용 text representation과 source artifact metadata를 보존해야 한다.

- 동일 chunk를 4개 embedding profile로 변환하여 pgvector에 저장한다.

- Qwen3 2560 profile은 halfvec(2560)으로 저장하고 integration test로 검증한다.

- 검색 화면에서 4개 embedding profile 결과를 병렬 비교할 수 있다.

- BM25 baseline 도입 후 검색 화면에서 BM25-only, embedding-only, hybrid 결과를 같은 query와 필터 조건으로 비교할 수 있다.

- 검색 화면/API에서 actor와 requested scope를 선택할 수 있고, 권한 범위 밖 문서는 결과에 노출되지 않는다.

- 검색 로그는 search profile/strategy runtime 설정, chunk policy, top-k, similarity/scoring metric, query instruction 또는 BM25 tokenizer/parameter, 결과 chunk를 재현 가능하게 저장한다.

- 검색 로그는 actor_user_id, requested_search_scope, effective_search_scope, permission_filter_metadata를 재현 가능하게 저장한다.

- 검색 결과에 대해 정답/부분정답/오답 피드백을 저장할 수 있다.

- 검색 로그 기반 retrieval context package, citation readiness, retrieval confidence gate를 통해 생성 실행 가능 여부를 판단할 수 있다.

- 생성 단계의 초기 실행은 deterministic mock provider로 가능해야 하며, remote DGX-Spark 접속 가능 시 vLLM OpenAI-compatible runtime smoke를 통해 Qwen3.6-27B-NVFP4 전환 가능성을 검증한다.

- pytest 기반 unit/regression/smoke test와 Playwright UI test가 통과한다.

- coverage 목표를 측정할 수 있고, 미달 시 품질 리포트를 생성한다.

# 11. 주요 리스크 및 대응 방안

| 리스크 | 영향 | 대응 방안 |
| --- | --- | --- |
| Qwen3-Embedding-4B 실행 자원 부족 | embedding 속도 저하, GPU OOM | Qwen worker 분리, batch size 조정, 1000/2560 profile 분리, smoke test 선행 |
| CPU-only 개발 PC의 embedding 처리 지연 | 대량 ingestion benchmark 시간이 과도하게 증가 | local adapter는 smoke/debug로 제한하고 GPU embedding provider API를 기본 benchmark 경로로 지원 |
| Remote embedding provider 장애 | embedding job 대기/실패 증가 | provider health check, timeout, retry/backoff, provider runtime metadata, failed job retry 제공 |
| Qwen 2560 vector 저장 타입 오류 | migration 실패 또는 검색 index 생성 실패 | qwen3_4b_2560은 halfvec(2560)으로 고정하고 pgvector halfvec integration test 추가 |
| HWPX parsing 품질 부족 | 국내 문서 검색 품질 저하 | 초기 XML text 추출 후 fixture 확대, parser regression test 강화 |
| 파일 확장자와 실제 내용 불일치 | 잘못된 parser 선택, 추출 실패 또는 보안 위험 | MIME type, 파일 signature, 내부 manifest를 함께 검사하고 detected_file_type/file_type_confidence를 저장 |
| 추출 산출물 미보존 | chunk 정책 변경, parser 개선, 검색 결과 재현이 어려움 | normalized markdown artifact와 artifact hash를 저장하고 chunk 재생성 기준으로 사용 |
| Table 구조 손실 | 표 기반 질문의 검색 정확도 저하 | table block은 검색용 text와 별도 table_artifacts markdown/json/csv 구조를 보존 |
| Image/OCR 정보 손실 | 그림/스캔 PDF 기반 질문을 후속 실험으로 확장하기 어려움 | image_artifacts에 source 위치, OCR text, caption, surrounding text metadata를 보존 |
| Chunk source anchor 누락 | 검색 결과 근거 화면과 환각 분석이 어려움 | chunk마다 artifact_id, block_id, chunk_type, source_anchor, prev/next link를 저장 |
| Chunk 정책 부적절 | 검색 recall 저하 또는 중복 검색 증가 | chunk_policy를 DB에 저장하고 정책별 평가 자동화 |
| Score 비교 오해 | 모델 성능 오판 | score 절대값이 아닌 rank/top-k/피드백 기준으로 비교 |
| Query instruction 미기록 | Qwen 계열 검색 결과 재현 불가 | profile metadata와 search log에 query instruction 사용 여부와 instruction text 저장 |
| BM25 tokenizer 품질 부족 | 한국어 keyword baseline이 실제 문서 의미를 충분히 반영하지 못함 | `unicode_word_v1`을 기본 재현 baseline으로 유지하고 `unicode_word_ko_2_3gram_v1`와 optional Mecab-ko 후보를 별도 실험 strategy로 비교 |
| 한국어 형태소 analyzer 라이선스/배포 리스크 | 고객사 설치 시 GPL/JVM/사전 배포 이슈가 발생할 수 있음 | KoNLPy는 보류하고 Mecab-ko는 `korean-tokenizers` optional dependency와 설치 evidence로 관리하며, 기본 tokenizer는 외부 의존성 없는 `unicode_word_v1`로 유지 |
| Hybrid score 왜곡 | 서로 다른 score scale을 단순 합산하여 품질을 오판 | 1차 hybrid는 RRF 기반 rank fusion으로 구현하고 score_components를 search log에 저장 |
| Retrieval confidence gate 미적용 | 관련 없는 검색 결과를 근거로 LLM이 그럴듯한 오답을 생성 | `answerable` 상태와 citation readiness를 generation run 전제 조건으로 기록하고, low/no-context는 no-answer 또는 blocked status로 처리 |
| vLLM remote runtime 장애 | 생성 실행 실패, timeout, 운영자 혼선 | mock provider를 기본 fallback으로 유지하고 provider base URL/model/timeout/health smoke evidence를 generation runtime metadata로 기록 |
| Qwen3.6-27B-NVFP4 GPU 자원 부족 | DGX memory pressure, 긴 latency, context length 제한 | 초기 context budget을 보수적으로 제한하고 max_tokens/temperature/top_p/context budget을 설정화하며, office network에서 별도 smoke 후 활성화 |
| 권한 필터 누락 | 접근 권한이 없는 문서/chunk가 검색 결과에 노출 | permission pre-filter를 repository/search SQL 단계에서 적용하고 permission matrix regression test 추가 |
| 사후 필터링으로 인한 top-k 왜곡 | 권한 없는 결과 제거 후 관련 chunk가 누락되어 검색 품질 오판 | vector search 후보군 생성 전 actor/scope 조건을 pre-filter로 적용 |
| 권한 metadata 미기록 | 같은 query 결과 차이를 재현하거나 설명하기 어려움 | search_logs에 actor, requested/effective scope, permission filter metadata 저장 |
| Queue backlog 누적 | 대용량/동시 업로드 시 처리 지연과 사용자 불안 증가 | queue depth와 stage별 latency를 dashboard에 표시하고 worker concurrency 설정값 제공 |
| Worker 중단 또는 lease 고착 | running job이 영구 대기 상태로 남음 | lease_expires_at과 heartbeat_at을 사용해 만료 job 재점유 및 retry 처리 |
| 중복 job 처리 | 같은 파일/chunk/profile이 중복 embedding되어 저장용량과 비용 증가 | checksum, chunk content_hash, chunk/profile unique constraint, idempotent worker 저장 로직 적용 |
| Pipeline 상태 UI 부재 | 사용자가 긴 처리 시간을 오류로 오해 | 업로드 직후 queued 상태와 stage timeline, progress_percent, error summary를 표시 |
| 테스트 DB 관리 실패 | 개발 DB 오염, regression 신뢰도 저하 | 독립 test DB 자동 생성/초기화/정리 fixture 적용 |
| UI 테스트 취약 | 브라우저 회귀 결함 누락 | Playwright smoke flow를 quality gate에 포함 |
| 실험 로그 미흡 | 결과 재현 불가 | profile runtime 설정, chunk_policy, query, top-k, index 설정, retrieved chunk 로그 필수 저장 |

# 부록 A. 요구사항 ID 요약

| Prefix | 범위 |
| --- | --- |
| FR | Functional Requirement. 기능 요구사항 |
| DR | Data Requirement. 데이터 및 저장소 요구사항 |
| UI | User Interface Requirement. 화면 요구사항 |
| NFR | Non-Functional Requirement. 비기능 요구사항 |
| IAM | Identity and Access Management. 계정, 조직, 역할, 접근 범위 요구사항 |
| ACL | Access Control List / Scope. 문서 접근 범위와 permission filter 요구사항 |
| PIPE | Pipeline Processing. 비동기 문서 처리, queue, worker, 진행 상태 요구사항 |
| GEN | Grounded Generation. retrieval context 기반 LLM 생성, provider runtime, prompt/citation/guardrail 요구사항 |
| QA | Quality Assurance Requirement. 품질관리 요구사항 |
| TST | Test Requirement. 테스트 요구사항 |
| ACC | Acceptance Criteria. 인수 기준 |

# 부록 B. 참고자료

본 SRS의 모델 사양 및 주요 기술 항목은 다음 공개 자료를 기준으로 작성하였다. 실제 개발 시에는 모델 카드와 라이선스 조건을 다시 확인해야 한다.

- nlpai-lab/KURE-v1 model card: https://huggingface.co/nlpai-lab/KURE-v1

- Qwen/Qwen3-Embedding-4B model card: https://huggingface.co/Qwen/Qwen3-Embedding-4B

- BAAI/bge-m3 model card: https://huggingface.co/BAAI/bge-m3

- BGE-M3 documentation: https://bge-model.com/bge/bge_m3.html

- vLLM OpenAI-compatible server documentation: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/

- NVIDIA Qwen3.6-27B-NVFP4 model card: https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4

- pgvector GitHub: https://github.com/pgvector/pgvector

- FastAPI testing documentation: https://fastapi.tiangolo.com/tutorial/testing/

- Playwright Python pytest documentation: https://playwright.dev/python/docs/test-runners

> **최종 정의**
>
> NeX_PCX는 FastAPI + Bootstrap 기반 Web 환경에서 사내 문서를 업로드하고, PostgreSQL 기반 pipeline queue와 worker로 텍스트 추출, chunking, embedding, vector indexing을 수행한 뒤, 동일 chunk를 4개 embedding profile과 BM25/hybrid retrieval strategy로 병렬 비교하는 NeX-CX 선행 검증 플랫폼이다. 본 프로젝트의 핵심 산출물은 프로그램 자체뿐 아니라, 한국어 사내 문서 검색을 위한 모델/검색전략/정책/운영/품질관리 기준 데이터이다.
