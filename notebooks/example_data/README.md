# example_data — 테스크마다 예제 10건, 원시 입력째로

`notebooks/tasks/*.ipynb` 의 4절이 이것을 읽습니다. **3.7 GB parquet 도 원본
아카이브도 필요 없습니다** — 저장소만 받으면 열립니다.

```
example_data/
  gen_data/<task>.jsonl        LLM 학습에 쓰이는 아이템 그대로 (10줄)
  raw/<task>/00..09/
    frames/f00.jpg ...         그 아이템에 실제로 들어가는 프레임 (384x216)
    radar.npz                  그 창의 레이더 반사점
    meta.json                  gen_data 의 같은 줄
```

`python -m notebooks.build_example_data` 로 다시 만듭니다.

## gen_data 한 줄의 필드

| 필드 | 내용 |
|---|---|
| `instruction` | 모델이 받는 질문. **출력 형식을 고르는 것이 이 문장입니다** |
| `ego` | 자차 운동, 32구간 양자화 (`t0:s3a4y16` = 속도 bin 3, 가속도 4, 요레이트 16) |
| `sensors` | 그 클립 리그가 실제로 단 레이더. 없으면 `no radar` |
| `target` | 정답. 평문 변형이 내야 하는 문자열 |
| `rationale` | CoT 변형이 정답 앞에 내야 하는 근거 |
| `cot_target` | CoT 변형의 정답 전체 — `{"rationale": ..., "answer": ...}` |
| `window` | 입력 창 사양 (`kind`, `seconds`, `radar_hz`, `video_frames`) |
| `radar_points` | 이 창에 있는 반사점 수. 0 이면 전방 레이더가 없는 클립 |

`cot_target` 의 `answer` 는 `target` 과 **글자 그대로 같습니다.** 그래서 한 건이
평문 변형과 CoT 변형을 모두 보여줍니다. 두 변형을 따로 뽑으면 서로 다른 클립이
잡혀 근거의 수치와 정답의 수치가 어긋나 보입니다.

## radar.npz

패딩을 뺀 반사점만 들어 있습니다. 로더가 넘기는 `[20, 1024, 8]` 텐서는 대부분이
패딩이라 그대로 두면 예제 하나가 327 kB 입니다.

| 배열 | 형태 | 내용 |
|---|---|---|
| `points` | `[M, 8]` float16 | `x, y, z, radial_velocity, doppler_residual, rcs, snr, range` |
| `scan` | `[M]` uint8 | 각 점이 몇 번째 스캔의 것인지 (0..19) |
| `channels` | `[8]` | 위 채널 이름 |

좌표는 자차 기준입니다 — x 전방, y 좌, z 상, 미터.

## 창은 테스크마다 다르고 스캔 수는 항상 20

| 태스크 | 창 | 레이더 | 비전 | 예제 크기 |
|---|---|---|---|---|
| `det_objects_azdeg` / `_3dbbox` | 순간 | 1초 · 20 Hz | 1장 | 0.9 / 1.0 MB |
| `track_step_azdeg` / `_bbox` | 5초 | 4 Hz | 5장 | 1.8 MB |
| `plan_ego_xy` / `_control` | 2초 | 10 Hz | 2장 | 1.1 / 1.3 MB |
| `agent_traj_azdeg` / `_bbox` | 2초 | 10 Hz | 2장 | 1.4 / 1.5 MB |
| `motion_seg_azdeg` / `_bbox` | 2초 | 10 Hz | 2장 | 1.3 / 1.6 MB |
| `qa` | 클립 전체 | 1 Hz | 20장 | 4.2 MB |

창 길이가 바뀌면 샘플링 속도가 따라 바뀝니다. 스캔 수가 20 으로 고정이라
포인트클라우드 인코더의 입력 모양은 어떤 태스크에서도 변하지 않습니다.

## 예제가 대표하는 것과 하지 않는 것

`train` 분할에서, 리그 프로필을 섞어서(`all_profiles=True`) 뽑았습니다. 그래서
전방 레이더가 없는 클립도 섞여 있습니다 — `plan_ego_control` 은 10건 중 4건이
그렇습니다. 그것은 결함이 아니라 데이터의 사실이고, 모델이 학습 중에 만나는
비율입니다. `radar_points` 로 걸러 보면 됩니다.

10건은 형식과 규모를 보기 위한 것이지 분포를 보기 위한 것이 아닙니다. 분포는
`notebooks/tasks/*.ipynb` 의 8절이 빌드 전체에서 셉니다.
