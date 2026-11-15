# FEM-LLM-web

지오메트리를 웹에 올리고, 문장으로 경계조건·하중을 지시한 뒤, 로컬 솔버로 해석하고 결과를 3D로 확인하는 개인용 FEM 도구입니다.

```
지오메트리 업로드 (OBJ / STL)
  → 문장 지시 → LLM이 스펙 수정 지시(JSON)를 생성 → 검증 → 화면에 구속·하중 표시
  → 해석 실행 (로컬 솔버)
  → 변형·컨투어·그래프 확인 (그래프도 문장으로 요청)
```

| 용도 | 솔버 | 입력 |
|---|---|---|
| 간이 검토 (오리가미 접힘) | 바-힌지 자체 솔버 | OBJ 면(`f`) + `hinge_*` 선 |
| 바·쉘 FEM | CalculiX | OBJ 선(`l`)·면(`f`) |
| 솔리드 FEM | TensorMesh (CalculiX로도 가능) | STL (닫힌 표면) |

단위는 mm · N · tonne · s · MPa 입니다.

## 설치 (macOS, Apple Silicon 기준)

```bash
git clone https://github.com/MunkyunLEE/FEM-LLM-web.git
cd FEM-LLM-web
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-solid.txt
```

`requirements-solid.txt`(PyTorch, TensorMesh)는 솔리드 해석에만 필요합니다. Python 3.10 이상이 필요합니다.

CalculiX:

```bash
brew install costerwi/homebrew-calculix/calculix-ccx
```

설치된 실행 파일 이름이 `ccx` 가 아니면 (`ccx_2.22` 등) 자동으로 찾습니다. 못 찾으면 `export FEMWEB_CCX=/경로/ccx_2.22` 로 지정합니다.

Ollama (로컬 LLM, 선택):

```bash
brew install ollama        # 또는 https://ollama.com/download 의 앱 설치
ollama serve               # 앱으로 설치했다면 생략
ollama pull qwen3:14b
```

LLM이 없어도 규칙 기반 해석기가 대신 동작하므로 화면 흐름은 그대로 쓸 수 있습니다.

## 실행

```bash
python -m femweb
```

브라우저에서 `http://127.0.0.1:8765` 가 열립니다. 서버가 화면까지 함께 제공하므로 별도 웹 호스팅은 필요 없습니다.

환경변수: `FEMWEB_LLM_URL` (기본 `http://localhost:11434/v1`), `FEMWEB_LLM_MODEL` (기본 `qwen3:14b`), `FEMWEB_LLM_KEY`, `FEMWEB_LLM_MODE` (`auto` · `llm` · `rule`), `FEMWEB_CCX`. LLM 설정은 화면 오른쪽 위에서도 바꿀 수 있고, OpenAI 호환 엔드포인트면 어떤 것이든 연결됩니다.

## 지오메트리 규칙

**OBJ (보·쉘)** — Rhino에서 메쉬와 선을 OBJ로 내보냅니다.

- `l` 선 = 보 요소, `f` 면(삼각형·사각형) = 쉘 요소
- `g 이름` 그룹은 선택 대상과 단면 지정에 쓰입니다 (`그룹 col_base 를 고정해줘`)
- 접힘선은 그룹 이름이 `hinge` 로 시작하는 `l` 선으로, 쉘 엣지 위에 그립니다
- 같은 위치의 정점은 자동으로 합쳐집니다
- `요소 분할수` k 를 주면 각 엣지를 k 등분합니다 (선은 k개, 사각형은 k×k개). 하중 합과 힌지 강성은 분할수와 무관하게 유지됩니다. 곡면은 직선 분할이므로 원래 메쉬를 충분히 잘게 내보내야 합니다

**STL (솔리드)** — 닫힌(watertight) 표면이어야 합니다. 서버가 Gmsh로 사면체 메쉬를 만들고, 평면·곡면 조각을 `face_1`, `face_2` … 그룹으로 등록합니다.

## 지시 예시

```
x 최소인 끝을 완전 고정해줘
x 최대 끝단에 z 방향으로 -1000 N 하중 줘
네 변 모두 z 방향만 구속해줘
상단 면에 0.5 MPa 압력
x 최대 변을 x 방향으로 -5 mm 밀어줘
좌굴 해석으로 바꾸고 모드 4개
비선형 해석으로 증분 20
시간-총에너지 그래프 그려줘
절점 (100, 0, 0) 의 z 방향 변위 그래프 그려줘
bc1 삭제해줘
```

화면에서 절점을 클릭하면 좌표가 표시되고 `지시문에 넣기` 로 문장에 넣을 수 있습니다. 구속·하중 목록에서 값을 직접 고치거나 지울 수도 있으며, 문장 지시와 같은 스펙을 수정합니다.

### 스펙 수정 지시 (LLM 출력 형식)

LLM은 `.inp` 를 쓰지 않고 아래 형태의 수정 지시만 냅니다. 서버가 검증한 뒤 솔버 입력으로 바꿉니다.

```json
{"ops": [
  {"op": "add_bc", "type": "fixed", "select": {"plane": {"axis": "x", "at": "min"}}},
  {"op": "add_load", "type": "force", "vector": [0, 0, -1000], "select": {"plane": {"axis": "x", "at": "max"}}},
  {"op": "add_plot", "title": "에너지", "x": {"quantity": "time"}, "series": [{"quantity": "energy.total"}]}
], "reply": "…"}
```

선택(`select`)은 `group`, `plane`, `box`, `nearest`, `nodes`, `all` 과 `and` / `or` 조합입니다. 힘은 선택 영역 전체에 대한 합이며 면·엣지·절점 순으로 분배합니다.

## 예제 (`examples/`)

`python examples/make_examples.py` 로 다시 만들 수 있습니다.

| 예제 | 내용 | 이 저장소에서 얻은 값 | 기준값 |
|---|---|---|---|
| `cantilever_beam.obj` | L=1000, 50×100, 끝단 1000 N (CalculiX B31) | 처짐 0.3834 mm | 보 이론 0.381 mm |
| `shell_panel.obj` | 200×200, t=2, 단순지지 엣지 압축 좌굴 (S8R) | 계수 30.09 | 30.37 |
| `cylinder_shell.obj` | R=50, L=200, t=2 축압축 좌굴 | 계수 28.72 | 고전해 31.9 |
| `miura.obj` | 4×4 미우라, 접힘 변위 30 mm (바-힌지) | 접힘 에너지 비율 97.8% | 강체 접힘에 가까움 |
| `cantilever_solid.stl` | 1000×50×100 솔리드 (TensorMesh tet10) | 처짐 0.3829 mm | 0.381 mm, CalculiX C3D10 0.3829 |
| `cylinder_solid.stl` | R50/R40, L=200 축압축 100 kN | 축변형 0.03365 mm | PL/EA 0.0337 mm |

## 구현 범위와 한계

- **CalculiX 어댑터**: B31 보, S8R/S6 쉘(중간 절점 자동 생성), C3D10 솔리드. 정적(선형·기하비선형·소성), 좌굴, 모달. 결과는 CalculiX가 펼친 3D 메쉬 그대로 표시합니다
- **바-힌지**: N5B8 위상(사각 패널 중심 절점 + 바 8개). 바 면적과 패널 굽힘 강성은 Filipov 등(2017)의 N4B5 식을 그대로 적용한 근사입니다. 하중·변위 제어 Newton, 원통형 호장법. 접촉은 없습니다
- **힌지 강성**: 길이당 κ (N·mm/rad/mm). 비워 두면 κ = 비율 × D / L (D 판 휨강성, L 패널 평균 변 길이)
- **CalculiX의 접힘선**: 회전 스프링이 없어 접힘선 양쪽 요소 한 줄의 두께를 줄인 띠로 근사합니다. 미우라 예제에서 바-힌지보다 뻣뻣하고 분할수에 따라 값이 달라집니다(분할 4, 8, 12에서 반력이 약 13배, 6배, 4배). Abaqus 커넥터 결과로 보정이 필요합니다
- **TensorMesh 어댑터**: 2차 사면체 선형 정적, Neo-Hookean 기하비선형(Newton-CG). 좌굴·모달·소성·열·유체는 아직 연결하지 않았습니다
- **LLM**: Ollama 실모델로는 아직 시험하지 못했습니다. 연결 흐름은 모의 서버로 시험했습니다
- 시험은 Linux에서 했습니다. macOS 실기 확인이 필요합니다

## 시험

```bash
pip install pytest
python -m pytest tests -q
```

## 라이선스

이 저장소는 MIT. three.js(MIT)를 `web/vendor` 에 포함합니다. CalculiX(GPL-2.0)와 Gmsh(GPL-2.0+)는 포함하지 않고 별도 설치본을 호출합니다.
