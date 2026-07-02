# NeX_PCX

**Software Requirements Specification v1.1**

*pre-CX RAG / Embedding / VectorDB Experiment Bench*

작성일: 2026-07-02  
문서 상태: Draft v1.1  
대상 시스템: FastAPI + Bootstrap + PostgreSQL/pgvector 기반 RAG 실험 플랫폼

본 문서는 NeX-CX 본 개발 이전의 선행 검증 프로젝트인 NeX_PCX의 요구사항을 정의한다.

# 문서 관리 정보

| 항목 | 내용 |
| --- | --- |
| 문서명 | NeX_PCX Software Requirements Specification v1.1 |
| 프로젝트명 | NeX_PCX (pre-CX) |
| 문서 목적 | NeX-CX 본 개발 전 RAG/Embedding/VectorDB 선행 검증 플랫폼의 기능, 데이터, 품질, 테스트 요구사항 정의 |
| 주요 기술 스택 | FastAPI, Bootstrap, PostgreSQL, pgvector, Python, pytest, Playwright |
| 핵심 평가 대상 | KURE-v1 1024, bge-m3 1024, Qwen3-Embedding-4B 1000, Qwen3-Embedding-4B 2560 |
| 문서 버전 | v1.1 |

| 버전 | 일자 | 작성/변경 내용 |
| --- | --- | --- |
| 0.1 | 2026-07-02 | NeX_PCX 프로젝트명 반영 전 초안 분석 |
| 1.0 | 2026-07-02 | Embedding 4개 profile, FastAPI+Bootstrap, 파일 metadata, 품질관리/테스트 요구사항 반영 |
| 1.1 | 2026-07-02 | pgvector 2560차원 저장 방식, 핵심 DB 스키마, embedding job lifecycle, chunk policy, 검색 재현성 metadata 보강 |

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
> PDF/DOCX/HWPX/PPTX/XLSX/MD 문서를 업로드하고, 동일한 chunk를 4개 embedding profile로 변환하여 PostgreSQL + pgvector에 저장한 뒤, 검색 화면에서 모델별 결과를 병렬 비교하고 정량 평가 데이터를 축적한다.

## 1.2 배경

- 사내 AX 구축을 위해 RAG(Retrieval-Augmented Generation) 기반의 문서 검색/생성 시스템이 필요하다.

- 현재는 문서를 chunk 단위로 분할하고 embedding model로 vector화하여 VectorDB에 저장한다는 개념은 알고 있으나, 실제 구현 경험과 실험 데이터가 부족하다.

- Embedding model, chunk size, overlap, 문서 포맷, 표/그림 처리 방식, vector index에 따라 검색 품질과 저장용량이 달라질 수 있다.

- NeX-CX 본 개발 전, 계속 등장하는 embedding model과 검색 기법을 동일 기준으로 반복 비교할 수 있는 실험 환경이 필요하다.

- Vector-only RAG의 환각 위험을 줄이기 위해 향후 목차/section graph 기반 보조 검색 구조도 실험할 예정이다.

## 1.3 범위

| 구분 | 포함 범위 |
| --- | --- |
| 문서 업로드 | PDF, DOCX, HWPX, PPTX, XLSX, MD 파일 업로드 및 원본 metadata 저장 |
| 문서 처리 | 텍스트 추출, heading/section 추정, chunk 생성, chunk metadata 저장 |
| Embedding | KURE-v1 1024, bge-m3 1024, Qwen3 1000, Qwen3 2560 profile 생성 및 저장 |
| 검색 | 동일 query에 대한 4개 profile 병렬 검색 결과 표시 |
| 평가 | 검색 결과 정답/부분정답/오답 피드백 저장, 검색 로그/latency 저장 |
| 품질관리 | pytest, Playwright, pytest-cov, 독립 test DB, quality gate 절차 적용 |

## 1.4 범위 제외

| 항목 | 제외 사유 / 향후 계획 |
| --- | --- |
| LLM 답변 생성 | MVP에서는 검색 품질 검증이 우선이다. LLM 답변 생성은 Phase 6 이후 확장한다. |
| 완전한 GraphRAG | 초기에는 heading/section/chunk metadata를 축적하고, 이후 graph-assisted retrieval로 확장한다. |
| 고도화된 이미지 embedding | 1차에서는 그림 원본/페이지/주변 텍스트/캡션 metadata 저장 수준으로 제한한다. |
| 조직 권한관리/SSO | 내부 PoC 단계에서는 최소 인증 또는 개발자 접근 환경으로 제한한다. |
| 대규모 분산 검색 | 초기에는 단일 PostgreSQL + worker 구조로 검증하고, 성능 병목 확인 후 확장한다. |

## 1.5 용어 및 약어

| 용어 | 정의 |
| --- | --- |
| NeX_PCX | NeX-CX 본 개발 이전의 pre-CX 실험 플랫폼 |
| NeX-CX | Context Intelligence Transformation을 위한 사내 문서 ingestion, embedding, 검색 기반 제품/플랫폼 |
| RAG | Retrieval-Augmented Generation. 검색 결과를 LLM 답변 생성의 근거로 사용하는 방식 |
| Embedding Profile | 특정 모델명, dimension, normalization, pooling 설정을 포함한 embedding 실행 단위 |
| Embedding Job | 특정 chunk와 embedding profile 조합에 대해 embedding 생성, 저장, 실패/재시도를 추적하는 background 작업 단위 |
| Chunk | 문서 본문을 검색 가능한 단위로 분할한 텍스트 조각 |
| Chunk Policy | chunk size, overlap, heading/table/code block 처리 방식 등을 포함하는 chunk 생성 정책 |
| pgvector | PostgreSQL에서 vector, halfvec type 및 similarity search를 지원하는 확장 |
| Groundedness | 답변 또는 검색 결과가 실제 문서 근거에 의해 뒷받침되는 정도 |
| Quality Gate | commit/sync 전에 반드시 통과해야 하는 테스트 및 품질 기준 |

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

## 2.3 운영 가정

- 초기 PoC는 내부 개발/검증용 단일 서버 또는 Docker Compose 환경에서 운영한다.

- PostgreSQL에는 pgvector extension이 활성화되어야 한다.

- Qwen3-Embedding-4B 2560 profile은 GPU 메모리와 처리 시간이 가장 많이 필요하므로 별도 worker로 분리한다.

- Qwen3-Embedding-4B 2560 profile은 pgvector의 일반 vector type 차원 한계를 초과하므로 MVP에서는 halfvec(2560) 저장 방식을 사용한다.

- 파일 업로드 후 embedding은 background job으로 수행하며, 대시보드에서 profile별 진행률을 확인할 수 있어야 한다.

- 검색 비교의 공정성을 위해 동일 문서, 동일 parser, 동일 chunk, 동일 query, 동일 top-k 기준으로 4개 profile을 비교한다.

# 3. 시스템 아키텍처

## 3.1 논리 아키텍처

```text
[Bootstrap Web UI]
├─ Dashboard
├─ File Upload
└─ Search / Model Compare
│
▼
[FastAPI Backend]
├─ Upload API
├─ Document Parsing Service
├─ Chunking Service
├─ Embedding Job API
├─ Search API
├─ Statistics API
└─ Feedback API
│
▼
[Background Workers]
├─ kure_v1_1024 worker
├─ bge_m3_1024 worker
├─ qwen3_4b_1000 worker
└─ qwen3_4b_2560 worker
│
▼
[PostgreSQL + pgvector]
├─ files / documents / chunks
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
| Parser Service | 파일 타입별 텍스트/메타데이터 추출 |
| Chunking Service | 문서 구조와 정책에 따라 chunk 생성 및 prev/next 연결 |
| Embedding Workers | profile별 모델 로딩, embedding 생성, 처리 시간/오류 기록 |
| Search Service | query embedding 생성, pgvector 검색, 4개 profile 결과 병렬 반환 |
| Statistics Service | 문서/chunk/vector/job/검색 통계 산출 |
| Quality/Test Framework | pytest, Playwright, coverage, 독립 test DB 기반 품질 검증 |

## 3.3 배포 구조

```text
Docker Compose 기준 초기 배포 구조

services:
web-app: FastAPI + Bootstrap templates
postgres: PostgreSQL + pgvector
redis: background job broker
worker-kure: KURE-v1 1024 embedding worker
worker-bge: bge-m3 1024 embedding worker
worker-qwen-1000: Qwen3-Embedding-4B 1000 worker
worker-qwen-2560: Qwen3-Embedding-4B 2560 worker
playwright-test: E2E test runner (CI 또는 local gate)
```

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
| FR-011 | 검색 API | query를 4개 profile로 검색하고 결과를 병렬 반환한다. | MUST |
| FR-012 | 검색 결과 비교 UI | 4개 profile 결과를 4-column UI로 표시한다. | MUST |
| FR-013 | 검색 피드백 저장 | 정답/부분정답/오답/중복/문맥부족 피드백을 저장한다. | MUST |
| FR-014 | 검색 로그 저장 | query, top-k, profile별 latency, 결과 chunk를 저장한다. | MUST |
| FR-015 | Embedding 실패 재처리 | 실패한 chunk/profile job을 재시도할 수 있어야 한다. | SHOULD |
| FR-016 | 평가 자동화 | golden question set 기반 Recall@k, MRR, nDCG를 산출한다. | SHOULD |
| FR-017 | Graph-assisted 확장 준비 | heading/section/chunk graph 저장을 위한 metadata를 보존한다. | SHOULD |
| FR-018 | Embedding job 상태 관리 | chunk/profile별 pending/running/succeeded/failed 상태, 시도 횟수, 오류 메시지를 저장한다. | MUST |
| FR-019 | Chunk policy 관리 | chunk size, overlap, split strategy, table/code block 보존 정책을 metadata로 관리한다. | MUST |
| FR-020 | 검색 재현성 metadata 저장 | 검색 시점의 profile runtime 설정, chunk policy, similarity metric, top-k, query instruction을 로그로 저장한다. | MUST |

## 4.3 대시보드 요구사항

| ID | 요구사항 |
| --- | --- |
| FR-001-01 | 전체 문서 수, 파일 타입별 문서 수, 문서 그룹별 문서 수를 표시한다. |
| FR-001-02 | 원본 파일 총 용량, 평균 파일 크기, 중복 파일 수를 표시한다. |
| FR-001-03 | 총 chunk 수, 평균 token 수, chunk 정책별 chunk 수를 표시한다. |
| FR-001-04 | kure_v1_1024, bge_m3_1024, qwen3_4b_1000, qwen3_4b_2560 profile별 완료/대기/실패 건수를 표시한다. |
| FR-001-05 | profile별 평균 embedding 시간과 평균 검색 latency를 표시한다. |
| FR-001-06 | parsing 실패, embedding 실패, DB 저장 실패 목록을 최근순으로 표시한다. |

## 4.4 파일 업로드 및 parsing 요구사항

| 파일 형식 | 처리 요구사항 |
| --- | --- |
| PDF | page text 추출, page_no 저장, text 없는 page 감지, 원본 PDF 보관 |
| DOCX | heading, paragraph, table text 추출, 문서 구조 보존 |
| HWPX | XML 기반 paragraph/table text 추출, section 구조 보존 |
| PPTX | slide_no, title, body, table text 추출, slide 단위 chunk 지원 |
| XLSX | sheet_name, used range, row/table text 추출, sheet/table 단위 chunk 지원 |
| MD | heading 구조, code block, table 구조 보존 |

## 4.5 검색 요구사항

- 검색 화면은 기본적으로 4개 embedding profile 전체 비교 모드로 동작한다.

- 검색 조건에는 query, 문서 그룹, 파일 타입, top-k, chunk policy, similarity metric을 포함한다.

- 초기 MVP의 similarity metric은 cosine distance를 기본값으로 한다.

- 검색 결과에는 rank, score/distance, 문서명, page/slide/sheet, heading_path, chunk preview, 피드백 버튼을 포함한다.

- 서로 다른 profile의 score는 직접 비교하지 않고, 정답 chunk 순위 및 top-k 포함 여부를 중심으로 비교한다.

## 4.6 Chunk policy 요구사항

- MVP 기본 정책은 heading-aware chunking으로 한다.

- 기본 chunk target size는 512 tokens, overlap은 64 tokens로 시작하되, chunk_policies table에 저장하여 실험별로 변경 가능해야 한다.

- Heading 경계가 있는 경우 heading hierarchy를 우선 보존하고, heading 하나의 본문이 target size를 초과하면 paragraph/table/code block 경계를 우선하여 분할한다.

- Markdown code block과 table, DOCX/HWPX/PPTX/XLSX table은 가능한 한 하나의 chunk 안에서 보존한다. target size를 크게 초과하는 table은 row group 단위로 분할한다.

- 각 chunk는 chunk_policy_name, parser_name, parser_version, token_count, char_count, content_hash를 저장하여 regression test와 검색 재현성에 사용할 수 있어야 한다.

## 4.7 Embedding job lifecycle 요구사항

- 파일 parsing과 chunk 생성이 완료되면 활성화된 embedding profile별로 chunk/profile 조합의 job을 생성한다.

- job status는 pending, running, succeeded, failed, skipped 중 하나로 관리한다.

- worker는 job을 lease 방식으로 점유하고, lease 만료 시 다른 worker가 재시도할 수 있어야 한다.

- 실패한 job은 attempts, max_attempts, error_message, last_error_at을 기록한다.

- 동일 chunk/profile 조합에 대해 중복 job이 생성되지 않도록 unique constraint를 둔다.

- succeeded job은 해당 profile의 embedding table에 vector 또는 halfvec 저장이 완료된 상태를 의미한다.

# 5. 데이터 및 저장소 요구사항

## 5.1 저장소 원칙

- 원본 파일 metadata, 문서 metadata, chunk metadata, embedding profile, 검색 로그, 피드백은 모두 PostgreSQL에 저장한다.

- 원본 파일은 파일 시스템 또는 object storage에 저장하고, DB에는 storage_path와 checksum을 저장한다.

- Embedding은 pgvector의 vector 또는 halfvec type으로 저장한다.

- KURE-v1 1024, bge-m3 1024, Qwen3 1000 profile은 vector type을 사용한다.

- Qwen3 2560 profile은 일반 vector type 차원 한계를 초과하므로 halfvec(2560)을 사용한다.

- Dimension이 다른 profile을 단순하게 관리하기 위해 MVP에서는 profile별 embedding table 분리 방식을 채택한다.

- MVP의 기본 검색은 cosine distance를 사용한다. 데이터가 5만 chunk 이하인 초기 실험에서는 exact search를 우선하고, 이후 HNSW 또는 IVFFlat index를 profile별로 추가한다.

- DB schema 변경 시 migration과 migration regression test를 추가한다.

## 5.2 핵심 테이블

| 테이블 | 설명 |
| --- | --- |
| files | 업로드 원본 파일 metadata 및 parsing 상태 |
| documents | 논리 문서 단위 정보 |
| chunks | 검색 단위 chunk와 구조 metadata |
| chunk_policies | chunk size, overlap, split strategy 등 chunk 생성 정책 |
| embedding_profiles | 모델명, dimension, storage type, runtime 설정 등 profile metadata |
| embedding_jobs | chunk/profile별 embedding 작업 상태와 재시도 정보 |
| chunk_embeddings_kure_v1_1024 | KURE-v1 1024차원 embedding 저장 |
| chunk_embeddings_bge_m3_1024 | bge-m3 1024차원 embedding 저장 |
| chunk_embeddings_qwen3_4b_1000 | Qwen3 1000차원 embedding 저장 |
| chunk_embeddings_qwen3_4b_2560 | Qwen3 2560차원 halfvec embedding 저장 |
| search_logs | 검색 query, 조건, profile별 latency 로그 |
| search_log_results | 검색 로그별 profile/rank/chunk 결과 |
| search_result_feedback | 검색 결과별 정답/부분정답/오답 피드백 |

## 5.3 files 테이블 요구사항

```sql
CREATE TABLE files (
file_id BIGSERIAL PRIMARY KEY,
original_file_name TEXT NOT NULL,
stored_file_name TEXT NOT NULL,
file_ext TEXT,
mime_type TEXT,
file_size_bytes BIGINT,
sha256_checksum TEXT UNIQUE,
storage_path TEXT NOT NULL,
document_group TEXT DEFAULT 'default',
security_level TEXT DEFAULT 'internal',
uploaded_by TEXT,
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
created_at TIMESTAMP DEFAULT now(),
updated_at TIMESTAMP DEFAULT now()
);
```

## 5.4 documents/chunks/chunk_policies 테이블 요구사항

```sql
CREATE TABLE documents (
document_id BIGSERIAL PRIMARY KEY,
file_id BIGINT NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
document_title TEXT,
document_group TEXT DEFAULT 'default',
security_level TEXT DEFAULT 'internal',
document_status TEXT DEFAULT 'active'
    CHECK (document_status IN ('active', 'archived', 'deleted')),
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
chunk_seq INT NOT NULL,
chunk_text TEXT NOT NULL,
content_hash TEXT NOT NULL,
chunk_policy_name TEXT NOT NULL REFERENCES chunk_policies(chunk_policy_name),
parser_name TEXT,
parser_version TEXT,
heading_path TEXT[],
page_no INT,
slide_no INT,
sheet_name TEXT,
cell_range TEXT,
token_count INT,
char_count INT,
prev_chunk_id BIGINT REFERENCES chunks(chunk_id),
next_chunk_id BIGINT REFERENCES chunks(chunk_id),
metadata JSONB DEFAULT '{}'::jsonb,
created_at TIMESTAMP DEFAULT now(),
UNIQUE (document_id, chunk_seq),
UNIQUE (document_id, content_hash, chunk_policy_name)
);
```

초기 chunk policy는 다음 값을 기본값으로 한다.

| chunk_policy_name | target_token_size | overlap_token_size | split_strategy | 비고 |
| --- | --- | --- | --- | --- |
| heading_512_64 | 512 | 64 | heading-aware | MVP 기본 정책 |

## 5.5 embedding profile 및 job 테이블 요구사항

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

## 5.6 profile별 embedding table

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

## 5.7 search log 및 feedback 테이블 요구사항

```sql
CREATE TABLE search_logs (
search_log_id BIGSERIAL PRIMARY KEY,
query_text TEXT NOT NULL,
normalized_query_text TEXT,
document_group TEXT,
file_type TEXT,
chunk_policy_name TEXT,
top_k INT NOT NULL,
similarity_metric TEXT NOT NULL DEFAULT 'cosine',
profiles JSONB NOT NULL,
query_runtime_metadata JSONB DEFAULT '{}'::jsonb,
total_elapsed_ms INT,
created_by TEXT,
created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE search_log_results (
search_log_result_id BIGSERIAL PRIMARY KEY,
search_log_id BIGINT NOT NULL REFERENCES search_logs(search_log_id) ON DELETE CASCADE,
profile_name TEXT NOT NULL REFERENCES embedding_profiles(profile_name),
rank INT NOT NULL,
chunk_id BIGINT NOT NULL REFERENCES chunks(chunk_id),
distance DOUBLE PRECISION,
score DOUBLE PRECISION,
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
created_at TIMESTAMP DEFAULT now()
);
```

## 5.8 저장용량 산정

저장 타입 기준 embedding 원시 저장용량은 vector profile은 chunk 수 × dimension × 4 bytes, halfvec profile은 chunk 수 × dimension × 2 bytes로 산정한다. 실제 운영 용량은 PostgreSQL row overhead, pgvector index, chunk text, metadata, 원본 파일, 추출 산출물 용량을 추가로 고려해야 한다.

| Chunk 수 | KURE 1024 | bge-m3 1024 | Qwen 1000 | Qwen 2560 | 합계 |
| --- | --- | --- | --- | --- | --- |
| 100,000 | 391 MB | 391 MB | 381 MB | 488 MB | 약 1.65 GB |
| 1,000,000 | 3.9 GB | 3.9 GB | 3.8 GB | 4.9 GB | 약 16.5 GB |
| 10,000,000 | 39 GB | 39 GB | 38 GB | 49 GB | 약 165 GB |

## 5.9 index 요구사항

- MVP 초기에는 exact search를 기준 결과로 남긴다.

- HNSW index를 추가하는 경우 cosine distance 기준으로 vector profile은 vector_cosine_ops, halfvec profile은 halfvec_cosine_ops를 사용한다.

- approximate index 적용 전후의 Recall@k 차이를 regression fixture로 비교한다.

# 6. Web UI 요구사항

## 6.1 공통 UI 원칙

- Bootstrap 기반의 단순하고 명확한 화면을 제공한다.

- MVP에서는 고급 SPA 구조보다 FastAPI template + Bootstrap 중심의 서버 렌더링 구성을 우선한다.

- 검색 결과 비교 화면에서는 4개 profile column의 시각적 정렬을 유지한다.

- 대시보드와 검색 결과는 개발자/평가자가 실험 결과를 빠르게 판단할 수 있도록 숫자와 상태 중심으로 표현한다.

## 6.2 화면 목록

| 화면 | 주요 구성요소 |
| --- | --- |
| Dashboard | 문서 수, 파일 용량, chunk 수, profile별 embedding 진행률, 검색 latency, 오류 목록 |
| File Upload | 파일 선택, 문서 그룹, 보안 등급, chunk policy, embedding profile 선택, 업로드 결과 |
| Search Compare | 검색어, 필터, top-k, 4-column 결과 비교, 피드백 버튼 |
| Document Detail | 원본 metadata, parsing 결과, chunk 목록, embedding 상태 |
| Job Monitor | embedding job queue, 완료/실패/재처리 상태 |

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

# 7. Embedding 및 검색 평가 요구사항

## 7.1 평가 원칙

- 동일 문서, 동일 parser, 동일 chunk, 동일 query, 동일 top-k 조건에서 embedding profile만 바꾸어 비교한다.

- 서로 다른 profile의 score 절대값은 직접 비교하지 않는다.

- 주요 비교 기준은 정답 chunk의 rank, top-k 포함 여부, 관련 section 검색 여부, 피드백 분포이다.

- Qwen3 1000 vs Qwen3 2560 비교를 통해 저장용량/검색품질 trade-off를 측정한다.

- Qwen3 2560 profile은 halfvec 저장에 따른 저장용량 절감과 검색 품질 변화를 함께 측정한다.

- bge-m3의 sparse/hybrid 기능은 MVP 이후 별도 phase에서 실험한다.

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

## 7.3 Golden Question Set 요구사항

- 초기에는 문서 50~100개와 질문 100개 이상의 내부 평가셋을 구성한다.

- 각 질문에는 expected_chunk_id 또는 expected_heading_path를 지정한다.

- 질문 유형은 단일근거, section 기반, 비교, 문서에 없는 질문, 표/그림 관련 질문으로 분류한다.

- 피드백이 누적되면 golden question set으로 승격하는 절차를 둔다.

## 7.4 검색 재현성 metadata 요구사항

- 모든 검색 로그는 query 원문과 normalized query를 함께 저장한다.

- 검색 로그는 profile_name, model_name, dimension, storage_type, normalize_embeddings, pooling_strategy, query_instruction, mvp_max_input_tokens를 저장해야 한다.

- Qwen 계열 profile은 query-side instruction 사용 여부와 instruction text를 반드시 profile metadata 또는 search log에 남긴다.

- 검색 로그는 chunk_policy_name, parser_name, parser_version, similarity_metric, top_k, index_type, index_parameters를 저장해야 한다.

- 검색 결과는 profile별 rank, chunk_id, distance, score, profile_elapsed_ms를 개별 row로 저장한다.

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

## 9.2 테스트 계층

| 테스트 유형 | 목적 | 예시 |
| --- | --- | --- |
| Unit Test | 개별 함수/클래스 검증 | parser, chunker, metadata extractor, embedding adapter |
| Integration Test | DB/pgvector/API/worker 연동 검증 | pgvector extension, vector search SQL, repository transaction |
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

## 9.5 테스트 디렉토리 구조

```text
tests/
├─ unit/
│ ├─ test_parsers_pdf.py
│ ├─ test_parsers_docx.py
│ ├─ test_parsers_hwpx.py
│ ├─ test_parsers_pptx.py
│ ├─ test_parsers_xlsx.py
│ ├─ test_chunker.py
│ ├─ test_chunk_policies.py
│ ├─ test_embedding_profiles.py
│ ├─ test_embedding_jobs.py
│ └─ test_file_metadata.py
├─ integration/
│ ├─ test_pgvector_extension.py
│ ├─ test_pgvector_halfvec.py
│ ├─ test_document_repository.py
│ ├─ test_embedding_repository.py
│ ├─ test_embedding_job_repository.py
│ └─ test_vector_search_profiles.py
├─ api/
│ ├─ test_upload_api.py
│ ├─ test_dashboard_api.py
│ ├─ test_search_api.py
│ └─ test_feedback_api.py
├─ regression/
│ ├─ test_upload_regression.py
│ ├─ test_chunking_regression.py
│ ├─ test_search_ranking_regression.py
│ └─ test_search_reproducibility_regression.py
├─ smoke/
│ ├─ test_app_startup.py
│ ├─ test_db_ready.py
│ └─ test_upload_to_search_flow.py
├─ e2e/
│ ├─ test_dashboard_ui.py
│ ├─ test_file_upload_ui.py
│ └─ test_search_compare_ui.py
├─ fixtures/
│ ├─ files/
│ ├─ expected_chunks/
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
pytest tests/unit
pytest tests/integration
pytest tests/regression
pytest tests/smoke
pytest --cov=app --cov-branch --cov-report=term-missing --cov-report=html
pytest tests/e2e
```

# 10. 개발 단계 및 인수 기준

| Phase | 목표 | 주요 산출물 | 인수 기준 |
| --- | --- | --- | --- |
| Phase 1 | Skeleton + 품질 기반 구축 | FastAPI/Bootstrap/PostgreSQL/pgvector/pytest/Playwright 기본 구조, migration 골격 | 앱 기동, DB 연결, vector/halfvec extension 확인, test DB 초기화, 기본 smoke test 통과 |
| Phase 2 | 파일 업로드 + metadata 저장 | 업로드 API/UI, files/documents 테이블, checksum 중복 검출, 문서 그룹/보안 등급 metadata | 지원 확장자 업로드와 metadata 저장 test 통과 |
| Phase 3 | Parsing + Chunking | 파일별 parser, heading-aware chunker, chunk_policies/chunks 저장 | fixture 문서별 chunk 생성 결과와 chunk policy regression test 통과 |
| Phase 4 | 4개 Embedding Profile 처리 | profile별 worker/table, embedding_jobs 상태 관리, Qwen 2560 halfvec 저장 | 동일 chunk에 대해 4개 profile embedding 저장 및 job status transition 검증 |
| Phase 5 | 검색 비교 UI | 4-column 검색 화면, search_logs/search_log_results/feedback 저장 | 동일 query에 대한 4개 profile top-k 결과 표시, 재현성 metadata 기록, 피드백 저장 |
| Phase 6 | 평가 자동화 | golden question set, Recall@k/MRR/nDCG 리포트 | 질문셋 기반 profile별 평가 리포트 생성 |

## 10.1 MVP 완료 기준

- PDF/DOCX/HWPX/PPTX/XLSX/MD 파일 업로드와 원본 metadata 저장이 가능하다.

- 파일별 parser가 최소 fixture 문서에 대해 정상적으로 text를 추출한다.

- 동일 chunk를 4개 embedding profile로 변환하여 pgvector에 저장한다.

- Qwen3 2560 profile은 halfvec(2560)으로 저장하고 integration test로 검증한다.

- 검색 화면에서 4개 profile 결과를 병렬 비교할 수 있다.

- 검색 로그는 profile runtime 설정, chunk policy, top-k, similarity metric, query instruction, 결과 chunk를 재현 가능하게 저장한다.

- 검색 결과에 대해 정답/부분정답/오답 피드백을 저장할 수 있다.

- pytest 기반 unit/regression/smoke test와 Playwright UI test가 통과한다.

- coverage 목표를 측정할 수 있고, 미달 시 품질 리포트를 생성한다.

# 11. 주요 리스크 및 대응 방안

| 리스크 | 영향 | 대응 방안 |
| --- | --- | --- |
| Qwen3-Embedding-4B 실행 자원 부족 | embedding 속도 저하, GPU OOM | Qwen worker 분리, batch size 조정, 1000/2560 profile 분리, smoke test 선행 |
| Qwen 2560 vector 저장 타입 오류 | migration 실패 또는 검색 index 생성 실패 | qwen3_4b_2560은 halfvec(2560)으로 고정하고 pgvector halfvec integration test 추가 |
| HWPX parsing 품질 부족 | 국내 문서 검색 품질 저하 | 초기 XML text 추출 후 fixture 확대, parser regression test 강화 |
| Chunk 정책 부적절 | 검색 recall 저하 또는 중복 검색 증가 | chunk_policy를 DB에 저장하고 정책별 평가 자동화 |
| Score 비교 오해 | 모델 성능 오판 | score 절대값이 아닌 rank/top-k/피드백 기준으로 비교 |
| Query instruction 미기록 | Qwen 계열 검색 결과 재현 불가 | profile metadata와 search log에 query instruction 사용 여부와 instruction text 저장 |
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
| QA | Quality Assurance Requirement. 품질관리 요구사항 |
| TST | Test Requirement. 테스트 요구사항 |
| ACC | Acceptance Criteria. 인수 기준 |

# 부록 B. 참고자료

본 SRS의 모델 사양 및 주요 기술 항목은 다음 공개 자료를 기준으로 작성하였다. 실제 개발 시에는 모델 카드와 라이선스 조건을 다시 확인해야 한다.

- nlpai-lab/KURE-v1 model card: https://huggingface.co/nlpai-lab/KURE-v1

- Qwen/Qwen3-Embedding-4B model card: https://huggingface.co/Qwen/Qwen3-Embedding-4B

- BAAI/bge-m3 model card: https://huggingface.co/BAAI/bge-m3

- BGE-M3 documentation: https://bge-model.com/bge/bge_m3.html

- pgvector GitHub: https://github.com/pgvector/pgvector

- FastAPI testing documentation: https://fastapi.tiangolo.com/tutorial/testing/

- Playwright Python pytest documentation: https://playwright.dev/python/docs/test-runners

> **최종 정의**
>
> NeX_PCX는 FastAPI + Bootstrap 기반 Web 환경에서 사내 문서를 업로드하고, 동일 chunk를 4개 embedding profile로 변환하여 PostgreSQL + pgvector에 저장한 뒤, 검색 결과를 병렬 비교하고 품질 피드백을 축적하는 NeX-CX 선행 검증 플랫폼이다. 본 프로젝트의 핵심 산출물은 프로그램 자체뿐 아니라, 한국어 사내 문서 검색을 위한 모델/정책/품질관리 기준 데이터이다.
