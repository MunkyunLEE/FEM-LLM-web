# FEM-LLM-web 개발 안내 (Claude Code 용)

개인용 FEM 웹 도구. 지오메트리(OBJ/STL) 업로드 → 문장으로 경계·하중 지정(규칙 해석기 우선, 선택적으로 OpenAI 호환 LLM) → 로컬 솔버 → 웹 뷰어로 결과 확인.

## 사용자와 작업 방식
- 사용자는 CAE 엔지니어(Abaqus 로 별도 검증). 대화와 화면 문구, 결과 요약은 한국어 존댓말
- 코드에 주석을 달지 않음. 큰 파일은 통째로 다시 쓰지 말고 필요한 곳만 수정
- 해석은 음해법만 씀(명시적 동적 해석은 쓰지 않음). 불안정 구간은 감쇠로 넘김
- 단위 mm · N · tonne · s · MPa (밀도 예: 알루미늄 2.7e-9)
- 확인하지 못한 것은 확인하지 못했다고 보고

## 실행 환경
- WSL2 Ubuntu: `scripts/setup_wsl.sh [--apt]`, `scripts/run_wsl.sh` (자세히는 `docs/WSL.md`)
- 가상환경 `~/.venvs/femweb`, 서버 `python -m femweb --no-browser` → `http://localhost:8765`
- 시험: `python -m pytest tests -q` (전체 약 8분, 자체 쉘만 `tests/test_shell.py tests/test_shell_quad.py` 약 1분). CalculiX 가 없으면 관련 시험은 건너뜀

## 구조
- `femweb/server.py`: FastAPI. 상태 `S`, 해석은 spawn 하위 프로세스(`/api/solve`, 강제 중단 `/api/cancel/solve`)
- `femweb/spec.py`: 스펙(JSON) 과 수정 지시(ops) 적용·검증(`apply_ops`, `warnings_for`)
- `femweb/rules.py`: 한국어·영어 문장 → ops 규칙 해석기. 새 문장 패턴은 여기에 추가하고 `tests/test_rules.py` 에 시험 추가
- `femweb/llm.py`: OpenAI 호환 `/chat/completions` 호출(프록시·사내 인증서 지원). 엔드포인트가 비어 있으면 LLM 미사용
- `femweb/solvers/`: `calculix.py`(ccx 입력 생성·실행), `barhinge.py`, `solid_tm.py`(TensorMesh), `shell.py`(자체 쉘: 삼각형 요소, Newton, IPC 접촉, 감쇠, 탄소성, 재개, `solve` 어댑터), `shell_quad.py`(4절점 쿼드 요소)
- `femweb/contact.py`, `control.py`(변위 제어), `loads.py`, `select.py`(절점 선택), `plots.py`, `result.py`
- `web/`: `index.html`, `app.js`(three.js 뷰어), `theory.html`(원리 문서, 기능을 바꾸면 같이 갱신)
- `examples/make_examples.py` → `examples/presets.json` 과 예제 OBJ/STL 생성. 예제를 바꾸면 `tests/test_api.py::test_examples_listed` 도 갱신

## 자체 쉘 솔버 요점
- 에너지 기반 Newton: 요소 내력·강성은 PyTorch 자동미분(`QuadShell._gh`), 수정 Newton(LU 재사용), 대칭 정렬 SuperLU
- 접촉: ipctk 장벽법, 최소 간격 = 두께, 장벽 범위 0.25 t, 처음 형상에서 가까운 절점 쌍 제외, 마찰 정규화 0.1 t
- 강체 누름판: 판에 닿은 절점의 판 이동량을 지정 변위로 둔 탄성 예측으로 증분 시작
- 감쇠: 레일리 등, 1·2차 고유진동수(강체 모드 제외)에서 계산. 누름판 압괴는 스텝 시간을 1차 주기의 약 1000배로 두면 감쇠 소산이 외부 일의 3% 안팎
- 에너지 이력 `energy.*` 와 `energy.external` 의 균형 오차를 결과 검증에 씀
- 재개: `solve(model, spec, checkpoint=path, every=초, resume=path)`
- 환경 변수: `FEMWEB_DEVICE`(cpu/cuda/mps/auto), `FEMWEB_HESS`(float32/float64), `FEMWEB_THREADS`

## 커밋
- 작업 브랜치 `wip/shell-solver`. 커밋 메시지는 한국어 한 줄 요약 + 항목
- 수정 후 관련 시험을 돌리고 결과를 함께 보고
