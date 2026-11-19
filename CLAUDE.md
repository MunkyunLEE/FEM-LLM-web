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
- 기준점 커플링: spec `couplings`(`add_coupling`, `update` 로 `fix`/`force`/`moment`), 자체 쉘은 `Augmented`(기준점 6자유도 추가) + `Coupling`(벌칙), CalculiX 는 `*RIGID BODY`. 시험 `tests/test_coupling.py`
- 솔리드 모멘트·회전: `loads.distributed_moment`(면 모멘트 → 합력 0 절점 힘), `spec.solid_rotations`(솔리드 면 회전 구속 → 자동 기준점 `cp_<bcid>`), TensorMesh `_coupled_linear`. 시험 `tests/test_solid_coupling.py`
- 대칭: `xsymm/ysymm/zsymm`, 반대칭 `xasymm/yasymm/zasymm`. CalculiX 쉘은 병진 성분만 구속
- 강체 부품: 그룹 전체를 기준점에 묶기(`add_coupling` `rigid`) 또는 강체 접촉면(`add_contact` `shape: "mesh"`, `body`). 후자는 `contact.split_rigid` 가 부품을 빼고 선택을 절점 목록으로 바꾼 뒤 IPC 장애물로 넘김(자체 쉘만). 시험 `tests/test_rigid_part.py`
- Abaqus 내보내기 `femweb/abaqus.py`(`export(model, spec, info)`, info 는 자체 쉘 결과 `res.abaqus`: 결함 좌표·α·β·스텝 시간), 비교 곡선 `femweb/compare.py`(`parse`, `metrics`), 서버 `S["refs"]`, `/api/refs`. 시험 `tests/test_abaqus.py`
- 초기 결함: `spec.imperfections(step)` → `[[모드, mm], ...]` (`imperfection_modes` 우선, 없으면 `imperfection` 을 1차로). 모드는 최대 성분 +1 정규화
- 진행 표시: `femweb/progress.py` (`context`, `stage`, `detail`), 서버 `GET /api/progress`
- 환경 변수: `FEMWEB_DEVICE`(cpu/cuda/mps/auto), `FEMWEB_HESS`(float32/float64), `FEMWEB_THREADS`

## BH-web (`bhweb/`, 솔버 `bh`)
- 웹 무관 계산 패키지: `geom`(이면각 닫힌 식 경사·헤시안), `model`(BHModel, 주름 판정 classify, 자름 split_cuts), `core`(바·힌지 탄소성, 조립), `solve`(Problem, 적응 감쇠 뉴턴 + 에너지 선 탐색, MGDCM), `analysis`(modal, dof_analysis, rigid_fold_path, solution_space), `contact`(GeometricContact), `shapes`, `benchmarks`
- 어댑터 `femweb/solvers/bh.py`, spec `creases`(엣지 "a-b" → fold/bend/cut), `bh`(설정), ops `set_crease`, `set_bh`, 서버 `/api/creases`, `/api/result/kin3d`
- 화면 기본 설정은 접힘 강성(자동 MERLIN2 `Lstar_ratio` = L*/L_F, 주름마다 L* = 비율×길이 / 수동 `fold_percent`)과 기하 접촉뿐. 나머지(`Lstar` 절대값, `fold_k`, `bar_area`, `tri_bars`, 소성, `cycles`·`cycle_back`, `gap`, `use_mass` 등)는 부가 설정: 규칙 `rules.parse_bh`·LLM 으로 넣고 화면 왼쪽 아래 "부가 설정" 에 표시
- bh 결과에는 기본 그래프 2개(`plots.bh_defaults`: 스텝-에너지, 하중-변위)가 항상 붙음. 컨투어는 요소 에너지를 표시 메쉬에 퍼뜨린 `energyDensity`
- 예제 `bh_*`, 시험 `tests/test_bh.py` (약 15초)
- 간이 `barhinge` 솔버는 그대로 둠

## 커밋
- 작업 브랜치 `wip/shell-solver`. 커밋 메시지는 한국어 한 줄 요약 + 항목
- 수정 후 관련 시험을 돌리고 결과를 함께 보고
