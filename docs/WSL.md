# WSL2 Ubuntu 설치·실행 안내 (회사 PC)

Windows 회사 PC 의 WSL2 Ubuntu 안에서 FEM-LLM-web 을 돌리는 절차입니다. 화면은 Windows 브라우저로 봅니다.

## 1. 준비 확인

PowerShell 에서:

```powershell
wsl -l -v
nvidia-smi
```

- WSL 버전이 2 이고 배포판이 Ubuntu 22.04 또는 24.04 이면 됩니다.
- NVIDIA GPU 를 쓰려면 Windows 쪽에 NVIDIA 드라이버만 설치되어 있으면 됩니다. WSL 안에 드라이버를 따로 설치하지 않습니다.

Ubuntu 안에서:

```bash
python3 --version
nvidia-smi
```

`nvidia-smi` 가 WSL 안에서도 GPU 를 보여 주면 CUDA 를 쓸 수 있습니다.

## 2. 프록시와 사내 인증서

이미 Claude Code 용으로 프록시와 인증서를 맞춰 두었다면 그대로 씁니다. `~/.bashrc` 에 다음이 있어야 합니다.

```bash
export HTTP_PROXY=http://<프록시주소>:<포트>
export HTTPS_PROXY=$HTTP_PROXY
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTP_PROXY
export NO_PROXY=localhost,127.0.0.1,<사내 도메인>
export no_proxy=$NO_PROXY
export FEMWEB_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

사내 루트 인증서를 시스템에 등록했다면(`/usr/local/share/ca-certificates/*.crt` 후 `sudo update-ca-certificates`) `FEMWEB_CA_BUNDLE` 은 위처럼 시스템 묶음 파일을 가리키면 됩니다. 설치 스크립트가 이 값을 pip 와 LLM 호출 양쪽에 넘깁니다.

apt 가 프록시를 쓰도록 `/etc/apt/apt.conf.d/95proxy` 에 다음을 둡니다.

```
Acquire::http::Proxy "http://<프록시주소>:<포트>";
Acquire::https::Proxy "http://<프록시주소>:<포트>";
```

## 3. 설치

작업 폴더는 WSL 쪽(홈 아래)에 둡니다. `/mnt/c/...` 에 두면 파일 입출력이 몇 배 느립니다.

```bash
git clone -b wip/shell-solver https://github.com/MunkyunLEE/FEM-LLM-web.git ~/FEM-LLM-web
cd ~/FEM-LLM-web
./scripts/setup_wsl.sh --apt
```

- `--apt` 는 처음 한 번만 붙입니다. CalculiX(`calculix-ccx`), gmsh 가 쓰는 그래픽 라이브러리, 빌드 도구를 apt 로 설치합니다.
- 가상환경은 `~/.venvs/femweb` 에 만듭니다(`FEMWEB_VENV` 로 바꿀 수 있음).
- Linux 용 PyTorch 는 PyPI 에서 받아도 CUDA 가 들어 있습니다. 회사 프록시에서 큰 파일(약 1 GB)이 끊기면 다시 실행하면 이어서 받습니다.
- 마지막에 torch(CUDA 여부), ipctk, gmsh, ccx 확인 결과가 나옵니다.

## 4. 실행

```bash
~/FEM-LLM-web/scripts/run_wsl.sh
```

- Windows 브라우저에서 `http://localhost:8765` 를 엽니다(가능하면 자동으로 열림).
- GPU 가 있으면 `FEMWEB_DEVICE=auto` 로 요소 강성 계산을 CUDA 배정밀도로 합니다. 결과 요약의 "계산 장치" 줄에서 확인합니다.
- 포트를 바꾸려면 `FEMWEB_PORT=8800 ./scripts/run_wsl.sh`.

## 5. LLM 연결 (선택)

LLM 은 설치하지 않습니다. 화면 왼쪽 LLM 칸에 넣으면 연결됩니다.

- 엔드포인트: OpenAI 호환 주소(`.../v1` 까지). 예: 사내 게이트웨이, `https://api.openai.com/v1`
- 모델: 그 엔드포인트의 모델 이름
- API 키: 필요한 경우만

엔드포인트를 비워 두면 규칙 기반 해석기만 씁니다. 사내 주소는 `NO_PROXY` 에 넣어 프록시를 거치지 않게 합니다. 시작할 때부터 연결하려면 환경 변수 `FEMWEB_LLM_URL`, `FEMWEB_LLM_MODEL`, `FEMWEB_LLM_KEY` 를 씁니다.

## 6. WSL 자원 설정

WSL2 는 기본으로 PC 메모리의 절반 정도만 씁니다. 큰 해석을 돌릴 때는 Windows 의 `C:\Users\<이름>\.wslconfig` 에:

```
[wsl2]
memory=24GB
processors=8
```

저장 후 PowerShell 에서 `wsl --shutdown` 으로 다시 시작합니다.

## 7. 긴 해석

- 자체 쉘 솔버는 `checkpoint` 로 상태를 저장하고 `resume` 으로 이어 풀 수 있습니다(README "해석 재개").
- 창을 닫아도 계속 돌게 하려면 `tmux` 나 `nohup` 안에서 실행합니다.

## 8. Rhino·Grasshopper 에서 부르기

서버가 HTTP 로 열려 있으므로 Windows 의 Rhino·Grasshopper 에서 `http://localhost:8765/api/...` 로 요청을 보낼 수 있습니다. 모델 업로드는 `/api/upload`, 문장 지시는 `/api/chat`, 해석은 `/api/solve` 입니다.

## 9. 알려진 제한

- 제 작업 환경(리눅스)에서 설치·실행 스크립트와 서버 기동은 확인했지만, 실제 WSL2·회사 프록시·NVIDIA GPU 에서는 확인하지 못했습니다.
- 선형 방정식 풀이는 CPU(SuperLU) 입니다. GPU 는 요소 강성 계산만 맡습니다.
