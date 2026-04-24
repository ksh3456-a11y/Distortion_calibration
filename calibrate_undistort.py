"""
드론 카메라 렌즈 왜곡 보정 파이프라인

체커보드 캘리브레이션(동영상 또는 이미지 폴더) → 파라미터 JSON 저장 → 왜곡 보정 이미지 일괄 출력.
캘리브레이션 해상도와 보정 대상 해상도가 다를 경우 fx/fy/cx/cy를 자동 스케일링한다.

사용 시나리오:
  1. 이미지 폴더 보정 (기존 JSON):
       python calibrate_undistort.py <input_dir> <output_dir> \\
           --calib_json calib.json

  2. 동영상 파일 보정 (→ 동영상으로 출력):
       python calibrate_undistort.py input.MP4 output.mp4 \\
           --calib_json calib.json

  3. 체커보드 비디오로 새 캘리브레이션:
       python calibrate_undistort.py <input_dir> <output_dir> \\
           --calib_video data/calib.MP4 \\
           --calib_json calib.json   # 결과 저장 경로 (선택)

  4. 체커보드 이미지 폴더로 새 캘리브레이션:
       python calibrate_undistort.py <input_dir> <output_dir> \\
           --calib_images data/dji_mini_pro_4_calibration/video_frames \\
           --calib_json calib.json   # 결과 저장 경로 (선택)

캘리브레이션 해상도 vs 보정 대상 해상도:
  - 해상도가 다르면 fx, fy, cx, cy를 비율에 맞게 자동 스케일링
  - 왜곡 계수(k1, k2, k3, p1, p2)는 무차원 → 스케일링 불필요
"""

import os
import glob
import json
import logging

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── 기본 상수 ──────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
VIDEO_EXTENSIONS = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV"}
JPEG_QUALITY = 95
DEFAULT_BOARD_SIZE = (10, 7)    # 체커보드 내부 코너 수 (cols, rows) — 실제 보드에 맞게 변경
DEFAULT_SQUARE_SIZE_MM = 25.0  # 체커보드 정사각형 한 변 (mm) — 실제 보드에 맞게 변경
DEFAULT_CALIB_FPS = 2.0        # 비디오에서 캘리브레이션 프레임 추출 FPS (n_calib_frames 미지정 시 사용)
DEFAULT_N_CALIB_FRAMES = 30    # 비디오를 N등분하여 추출할 프레임 수 (기본값)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 내부 유틸 함수
# ─────────────────────────────────────────────────────────────────────────────

def _load_images_from_dir(images_dir: str) -> list:
    """이미지 폴더에서 BGR numpy 배열 리스트로 로드."""
    paths = sorted([
        p for p in glob.glob(os.path.join(images_dir, "*"))
        if os.path.splitext(p)[1] in SUPPORTED_EXTENSIONS
    ])
    if not paths:
        raise RuntimeError(f"이미지 파일 없음: {images_dir}")
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)
        else:
            log.warning(f"읽기 실패, 건너뜀: {p}")
    log.info(f"이미지 폴더에서 {len(images)}장 로드: {images_dir}")
    return images


def _get_video_rotation(video_path: str) -> int:
    """
    ffprobe로 비디오 rotation 메타데이터를 읽어 반환한다.
    DJI 드론 비디오는 보통 rotate=270 메타데이터를 가진다.
    반환값: 0, 90, 180, 270 중 하나.
    """
    try:
        import subprocess, json as _json
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
            capture_output=True, text=True, timeout=10
        )
        data = _json.loads(result.stdout)
        v = next((s for s in data["streams"] if s["codec_type"] == "video"), {})
        rotate = int(v.get("tags", {}).get("rotate", 0))
        return rotate
    except Exception:
        return 0


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """rotation 각도(0/90/180/270)에 따라 프레임을 회전한다."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def _extract_frames_from_video(
    video_path: str,
    n_frames: int = DEFAULT_N_CALIB_FRAMES,
    fps: float = None,
) -> list:
    """
    비디오에서 프레임을 추출하여 BGR numpy 배열 리스트로 반환.

    OpenCV VideoCapture는 rotation 메타데이터를 무시하므로,
    ffprobe로 rotation을 별도 확인하여 직접 적용한다.
    (FFmpeg은 자동 적용하지만 OpenCV는 무시 → 캘리브레이션 방향 불일치 방지)

    n_frames가 지정되면 비디오 전체를 N등분하여 각 구간 중앙 프레임을 추출.
    n_frames가 None이고 fps가 지정되면 FPS 기반으로 추출.

    Args:
        video_path: 비디오 파일 경로
        n_frames:   추출할 총 프레임 수 (비디오 N등분 방식)
        fps:        FPS 기반 추출 (n_frames=None일 때만 사용)
    """
    rotation = _get_video_rotation(video_path)
    if rotation != 0:
        log.info(f"비디오 rotation 메타데이터 감지: {rotation}° → 프레임에 자동 적용")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if n_frames is not None:
        # N등분: 각 구간의 중앙 프레임 인덱스 계산
        n = min(n_frames, total_frames)
        target_indices = set(
            int((i + 0.5) * total_frames / n) for i in range(n)
        )
        mode_desc = f"N등분 {n}장"
    else:
        # FPS 기반: interval마다 추출
        interval = max(1, int(round(video_fps / fps)))
        target_indices = set(range(0, total_frames, interval))
        mode_desc = f"FPS={fps}"

    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in target_indices:
            frames.append(_apply_rotation(frame, rotation))
        idx += 1
    cap.release()

    log.info(
        f"비디오에서 {len(frames)}개 프레임 추출 "
        f"(전체 {total_frames}프레임, {mode_desc}, rotation={rotation}°): {video_path}"
    )
    return frames

def _is_aspect_flipped(src_w: int, src_h: int, dst_w: int, dst_h: int,
                       tol: float = 0.05) -> bool:
    """src와 dst의 종횡비가 서로 '뒤집혀' 있는지 판단 (90° 회전 관계)."""
    src_ar = src_w / src_h
    dst_ar = dst_w / dst_h
    # dst_ar이 1/src_ar에 가까우면 뒤집힌 것
    return abs(dst_ar - 1.0 / src_ar) < tol * (1.0 / src_ar)


def _rotate_intrinsics_90(params: dict) -> dict:
    """
    캘리브레이션 파라미터를 90° 회전시킨다.
    이미지가 시계방향 90° 회전되었다고 가정 (세로→가로 또는 가로→세로).
    
    회전 규칙 (CW 90°: (x, y) -> (H-1-y, x) 기준):
      - fx ↔ fy 스왑
      - cx_new = (H - 1) - cy_old   (근사: H - cy)
      - cy_new = cx_old
      - 왜곡 계수: k1, k2, k3는 반경 방향이라 불변
      - 접선 왜곡 p1, p2는 좌표축 방향에 따라 스왑 및 부호 변경
      - 해상도: W ↔ H 스왑
    
    주의: 실제 이미지가 CW인지 CCW 90°인지에 따라 cx/cy 공식이 달라집니다.
          여기서는 대칭성 덕분에 cx≈W/2, cy≈H/2 근처라 어느 쪽이든 큰 차이 없음.
    """
    rotated = dict(params)
    rotated["fx"] = params["fy"]
    rotated["fy"] = params["fx"]
    # 중앙 근처 가정하에 단순 스왑 (엄밀히는 H-1-cy, 하지만 거의 같음)
    rotated["cx"] = params["image_height"] - params["cy"]
    rotated["cy"] = params["cx"]
    # 접선 왜곡 계수 스왑 (p1, p2가 x/y 방향에 각각 대응하므로 회전 시 스왑)
    rotated["p1"] = params.get("p2", 0.0)
    rotated["p2"] = params.get("p1", 0.0)
    # 반경 왜곡 k1, k2, k3는 회전 불변이라 그대로
    rotated["image_width"]  = params["image_height"]
    rotated["image_height"] = params["image_width"]
    return rotated


def _scale_intrinsics(params: dict, src_w: int, src_h: int,
                      dst_w: int, dst_h: int) -> dict:
    """
    캘리브레이션 파라미터를 타겟 해상도에 맞게 조정.
    종횡비가 뒤집혀있으면 자동으로 90° 회전 후 스케일링.
    """
    # 1) 종횡비 뒤집힘 감지 → 회전
    if _is_aspect_flipped(src_w, src_h, dst_w, dst_h):
        log.info(
            f"종횡비 뒤집힘 감지: ({src_w}×{src_h}) vs ({dst_w}×{dst_h}) "
            f"→ 캘리브레이션 파라미터를 90° 회전"
        )
        params = _rotate_intrinsics_90(params)
        src_w, src_h = params["image_width"], params["image_height"]
        log.info(
            f"  회전 후: fx={params['fx']:.2f}, fy={params['fy']:.2f}, "
            f"cx={params['cx']:.2f}, cy={params['cy']:.2f}, "
            f"해상도={src_w}×{src_h}"
        )
    
    # 2) 해상도가 동일하면 그대로 반환
    if src_w == dst_w and src_h == dst_h:
        return params
    
    # 3) 비례 스케일링
    sx = dst_w / src_w
    sy = dst_h / src_h
    
    if abs(sx - sy) > 1e-2:
        log.warning(
            f"가로/세로 스케일 비율이 다릅니다 (sx={sx:.4f}, sy={sy:.4f}). "
            f"캘리브레이션 해상도와 타겟 해상도의 종횡비가 다릅니다."
        )
    
    scaled = dict(params)
    scaled["fx"] = params["fx"] * sx
    scaled["fy"] = params["fy"] * sy
    scaled["cx"] = params["cx"] * sx
    scaled["cy"] = params["cy"] * sy
    scaled["image_width"]  = dst_w
    scaled["image_height"] = dst_h
    
    log.info(
        f"파라미터 스케일링: ({src_w}×{src_h}) → ({dst_w}×{dst_h}), "
        f"sx={sx:.4f}, sy={sy:.4f}"
    )
    log.info(
        f"  fx: {params['fx']:.2f} → {scaled['fx']:.2f}, "
        f"fy: {params['fy']:.2f} → {scaled['fy']:.2f}, "
        f"cx: {params['cx']:.2f} → {scaled['cx']:.2f}, "
        f"cy: {params['cy']:.2f} → {scaled['cy']:.2f}"
    )
    return scaled


# ─────────────────────────────────────────────────────────────────────────────
# 체커보드 캘리브레이션
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_from_checkerboard(
    calib_source: str,
    output_json: str,
    board_size: tuple = DEFAULT_BOARD_SIZE,
    square_size_mm: float = DEFAULT_SQUARE_SIZE_MM,
    n_calib_frames: int = DEFAULT_N_CALIB_FRAMES,
    calib_fps: float = None,
) -> dict:
    """
    체커보드 소스(비디오 또는 이미지 폴더)로 카메라 캘리브레이션을 수행하고 JSON에 저장한다.

    Args:
        calib_source:    체커보드 비디오 파일 경로 또는 이미지 폴더 경로
        output_json:     캘리브레이션 결과 저장 JSON 경로
        board_size:      내부 코너 수 (cols, rows). 체커보드 인쇄물에 맞게 지정.
        square_size_mm:  체커보드 정사각형 한 변 크기 (mm)
        n_calib_frames:  비디오를 N등분하여 추출할 프레임 수 (기본 15).
                         calib_fps가 지정되면 무시됨.
        calib_fps:       FPS 기반 추출 (지정 시 n_calib_frames 대신 사용)

    Returns:
        캘리브레이션 파라미터 딕셔너리
    """
    if os.path.isfile(calib_source):
        images = _extract_frames_from_video(
            calib_source,
            n_frames=None if calib_fps else n_calib_frames,
            fps=calib_fps,
        )
    elif os.path.isdir(calib_source):
        images = _load_images_from_dir(calib_source)
    else:
        raise RuntimeError(f"calib_source를 찾을 수 없음: {calib_source}")

    if not images:
        raise RuntimeError("캘리브레이션용 이미지가 없습니다.")

    h, w = images[0].shape[:2]

    # 체커보드 3D 객체점 생성 (Z=0 평면)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points, img_points = [], []

    log.info(
        f"체커보드 코너 탐색 시작 "
        f"(내부코너 {board_size[0]}×{board_size[1]}, 정사각형 {square_size_mm}mm)"
    )

    found_count = 0
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
            found_count += 1
            log.info(f"  [{i+1:3d}/{len(images)}] 체커보드 검출 성공 (누적 {found_count}개)")
        else:
            log.info(f"  [{i+1:3d}/{len(images)}] 체커보드 미검출")

    if found_count < 3:
        raise RuntimeError(
            f"체커보드 검출 수 부족 ({found_count}개, 최소 3개 필요). "
            f"board_size={board_size} 설정이 실제 보드와 맞는지 확인하세요."
        )

    log.info(f"캘리브레이션 실행 중 ({found_count}/{len(images)}장 사용)...")

    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )

    log.info(f"RMS 재투영 오차: {rms:.4f} px")
    if rms > 2.0:
        log.warning(
            f"RMS={rms:.4f}가 높습니다 (권장: < 1.0). "
            "체커보드 이미지 품질이나 board_size 설정을 확인하세요."
        )

    params = {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "k1": float(dist[0, 0]),
        "k2": float(dist[0, 1]),
        "p1": float(dist[0, 2]),
        "p2": float(dist[0, 3]),
        "k3": float(dist[0, 4]) if dist.shape[1] > 4 else 0.0,
        "rms": float(rms),
        "image_width":  w,
        "image_height": h,
        "n_images_used": found_count,
        "board_size":   list(board_size),
        "square_size_mm": square_size_mm,
        "source": calib_source,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    log.info(f"캘리브레이션 결과 저장: {output_json}")

    return params


# ─────────────────────────────────────────────────────────────────────────────
# 왜곡 보정 적용
# ─────────────────────────────────────────────────────────────────────────────

def undistort_images(
    input_dir: str,
    output_dir: str,
    params: dict,
) -> dict:
    """
    캘리브레이션 파라미터로 이미지 폴더 전체에 왜곡 보정을 적용한다.

    - Focal length 보존: newCameraMatrix=K 고정 (getOptimalNewCameraMatrix 미사용)
    - 캘리브레이션 해상도와 타겟 해상도가 다를 경우 자동 스케일링

    Args:
        input_dir:  원본 이미지 폴더
        output_dir: 보정 이미지 저장 폴더
        params:     캘리브레이션 파라미터 딕셔너리

    Returns:
        {"processed_count": int, "output_dir": str}
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        p for p in glob.glob(os.path.join(input_dir, "*"))
        if os.path.splitext(p)[1] in SUPPORTED_EXTENSIONS
    ])
    if not image_files:
        raise RuntimeError(f"이미지 파일 없음: {input_dir}")

    # 첫 이미지로 타겟 해상도 확인 후 파라미터 스케일링
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        raise RuntimeError(f"이미지 읽기 실패: {image_files[0]}")
    dst_h, dst_w = first_img.shape[:2]

    src_w = params.get("image_width",  dst_w)
    src_h = params.get("image_height", dst_h)
    scaled = _scale_intrinsics(params, src_w, src_h, dst_w, dst_h)

    K = np.array([
        [scaled["fx"], 0.0,          scaled["cx"]],
        [0.0,          scaled["fy"], scaled["cy"]],
        [0.0,          0.0,          1.0         ],
    ], dtype=np.float64)

    # OpenCV undistort 계수 순서: [k1, k2, p1, p2, k3]
    dist_coeffs = np.array([
        scaled.get("k1", 0.0),
        scaled.get("k2", 0.0),
        scaled.get("p1", 0.0),
        scaled.get("p2", 0.0),
        scaled.get("k3", 0.0),
    ], dtype=np.float64)

    log.info(
        f"보정 파라미터: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, "
        f"cx={K[0,2]:.2f}, cy={K[1,2]:.2f}"
    )
    log.info(
        f"왜곡 계수: k1={dist_coeffs[0]:.5f}, k2={dist_coeffs[1]:.5f}, "
        f"k3={dist_coeffs[4]:.5f}, p1={dist_coeffs[2]:.5f}, p2={dist_coeffs[3]:.5f}"
    )
    log.info(f"처리할 이미지: {len(image_files)}개")

    processed_count = 0
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            log.warning(f"읽기 실패, 건너뜀: {img_path}")
            continue

        # focal length 보존: newCameraMatrix=K 고정
        undistorted = cv2.undistort(img, K, dist_coeffs, newCameraMatrix=K)

        out_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, out_name)
        ext = os.path.splitext(out_name)[1].lower()
        if ext in {".jpg", ".jpeg"}:
            cv2.imwrite(out_path, undistorted, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        else:
            cv2.imwrite(out_path, undistorted)

        processed_count += 1

    log.info(f"보정 완료: {processed_count}개 → {output_dir}")
    return {"processed_count": processed_count, "output_dir": output_dir}


def undistort_video(
    input_video: str,
    output_path: str,
    params: dict,
) -> dict:
    """
    캘리브레이션 파라미터로 동영상 전체에 왜곡 보정을 적용한다.

    - 입력 동영상과 동일한 FPS, 해상도, 코덱으로 출력
    - DJI 드론 비디오의 rotation 메타데이터를 감지하여 적용

    Args:
        input_video: 원본 동영상 파일 경로
        output_path: 보정 동영상 저장 경로 (파일 경로 또는 디렉토리)
        params:      캘리브레이션 파라미터 딕셔너리

    Returns:
        {"processed_frames": int, "output_path": str}
    """
    # output_path가 디렉토리면 동일 파일명으로 저장
    if os.path.isdir(output_path) or not os.path.splitext(output_path)[1]:
        os.makedirs(output_path, exist_ok=True)
        out_name = os.path.basename(input_video)
        output_path = os.path.join(output_path, out_name)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    rotation = _get_video_rotation(input_video)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"동영상 열기 실패: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 첫 프레임으로 해상도 확인 (rotation 적용 후)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError(f"동영상 프레임 읽기 실패: {input_video}")
    first_frame = _apply_rotation(first_frame, rotation)
    dst_h, dst_w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 파라미터 스케일링
    src_w = params.get("image_width", dst_w)
    src_h = params.get("image_height", dst_h)
    scaled = _scale_intrinsics(params, src_w, src_h, dst_w, dst_h)

    K = np.array([
        [scaled["fx"], 0.0,          scaled["cx"]],
        [0.0,          scaled["fy"], scaled["cy"]],
        [0.0,          0.0,          1.0         ],
    ], dtype=np.float64)

    dist_coeffs = np.array([
        scaled.get("k1", 0.0),
        scaled.get("k2", 0.0),
        scaled.get("p1", 0.0),
        scaled.get("p2", 0.0),
        scaled.get("k3", 0.0),
    ], dtype=np.float64)

    # undistort map 사전 계산 (프레임마다 재계산하지 않도록)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist_coeffs, None, K, (dst_w, dst_h), cv2.CV_16SC2)

    # VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (dst_w, dst_h))
    if not writer.isOpened():
        raise RuntimeError(f"동영상 쓰기 실패: {output_path}")

    log.info(
        f"동영상 왜곡 보정: {input_video} → {output_path}\n"
        f"  해상도: {dst_w}x{dst_h}, FPS: {fps:.2f}, "
        f"총 프레임: {total_frames}, rotation: {rotation}°"
    )

    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = _apply_rotation(frame, rotation)
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        writer.write(undistorted)
        processed += 1
        if processed % 500 == 0:
            log.info(f"  진행: {processed}/{total_frames} ({processed/total_frames*100:.0f}%)")

    cap.release()
    writer.release()

    log.info(f"동영상 보정 완료: {processed}프레임 → {output_path}")
    return {"processed_frames": processed, "output_path": output_path}


# ─────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    output_path: str,
    calib_json: str = None,
    calib_video: str = None,
    calib_images: str = None,
    board_size: tuple = DEFAULT_BOARD_SIZE,
    square_size_mm: float = DEFAULT_SQUARE_SIZE_MM,
    n_calib_frames: int = DEFAULT_N_CALIB_FRAMES,
    calib_fps: float = None,
) -> dict:
    """
    왜곡 보정 전체 파이프라인.

    입력이 이미지 폴더이면 이미지 일괄 보정, 동영상 파일이면 동영상 보정.

    우선순위:
      1. calib_json 파일이 존재하면 → 즉시 로드하여 undistort
      2. calib_json이 없고 calib_video 또는 calib_images가 있으면
         → 체커보드 캘리브레이션 → JSON 저장 → undistort

    캘리브레이션 해상도와 입력 해상도가 다를 경우 fx/fy/cx/cy를 자동 스케일링.

    Args:
        input_path:      왜곡 보정할 이미지 폴더 또는 동영상 파일
        output_path:     보정 결과 저장 경로 (폴더 또는 동영상 파일)
        calib_json:      기존 캘리브레이션 JSON 경로 (없으면 새로 캘리브레이션 후 저장)
        calib_video:     체커보드 캘리브레이션 비디오 경로 (calib_json 미존재 시 사용)
        calib_images:    체커보드 캘리브레이션 이미지 폴더 (calib_json 미존재 시 사용)
        board_size:      체커보드 내부 코너 수 (cols, rows)
        square_size_mm:  체커보드 정사각형 크기 (mm)
        n_calib_frames:  비디오를 N등분하여 추출할 프레임 수 (calib_fps 미지정 시 사용)
        calib_fps:       FPS 기반 프레임 추출 (지정 시 n_calib_frames 대신 사용)

    Returns:
        {"calibration_params": dict, "undistort_result": dict}
    """
    # ── 캘리브레이션 파라미터 결정 ────────────────────────────────────────────
    if calib_json and os.path.exists(calib_json):
        log.info(f"[시나리오 1] 기존 캘리브레이션 JSON 사용: {calib_json}")
        with open(calib_json, "r", encoding="utf-8") as f:
            params = json.load(f)
        log.info(
            f"  RMS={params.get('rms', 'N/A')}, "
            f"fx={params['fx']:.2f}, fy={params['fy']:.2f}, "
            f"해상도={params.get('image_width','?')}×{params.get('image_height','?')}"
        )

    elif calib_video or calib_images:
        calib_source = calib_video if calib_video else calib_images
        # output_path가 파일이면 같은 디렉토리에, 폴더면 그 안에 저장
        if os.path.isfile(input_path):
            out_dir = os.path.dirname(os.path.abspath(output_path)) if os.path.splitext(output_path)[1] else output_path
        else:
            out_dir = output_path
        save_json = calib_json if calib_json else os.path.join(out_dir, "calibration.json")

        log.info(f"[시나리오 2] 체커보드 캘리브레이션 실행: {calib_source}")
        params = calibrate_from_checkerboard(
            calib_source=calib_source,
            output_json=save_json,
            board_size=board_size,
            square_size_mm=square_size_mm,
            n_calib_frames=n_calib_frames,
            calib_fps=calib_fps,
        )

    else:
        raise RuntimeError(
            "캘리브레이션 소스가 없습니다.\n"
            "  --calib_json (기존 JSON) 또는\n"
            "  --calib_video / --calib_images (체커보드 소스) 중 하나를 지정하세요."
        )

    # ── 왜곡 보정 적용 (이미지 폴더 또는 동영상) ─────────────────────────────
    is_video = (os.path.isfile(input_path) and
                os.path.splitext(input_path)[1].lower() in
                {e.lower() for e in VIDEO_EXTENSIONS})

    if is_video:
        log.info(f"동영상 왜곡 보정: {input_path} → {output_path}")
        result = undistort_video(input_path, output_path, params)
    else:
        log.info(f"이미지 왜곡 보정: {input_path} → {output_path}")
        os.makedirs(output_path, exist_ok=True)
        result = undistort_images(input_path, output_path, params)

    return {"calibration_params": params, "undistort_result": result}


# ─────────────────────────────────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "드론 카메라 렌즈 왜곡 보정 파이프라인\n"
            "체커보드 캘리브레이션(또는 기존 JSON) → 왜곡 보정 이미지/동영상 출력"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 이미지 폴더 보정
  python calibrate_undistort.py images/ output/ --calib_json calib.json

  # 동영상 파일 보정 (→ 동영상으로 출력)
  python calibrate_undistort.py input.MP4 output.mp4 --calib_json calib.json

  # 동영상 파일 보정 (출력 폴더 지정 → 동일 파일명으로 저장)
  python calibrate_undistort.py input.MP4 output_dir/ --calib_json calib.json

  # 체커보드 비디오로 캘리브레이션 후 보정
  python calibrate_undistort.py images/ output/ --calib_video calib.MP4

  # 체커보드 이미지 폴더로 캘리브레이션 후 보정
  python calibrate_undistort.py images/ output/ --calib_images calib_frames/
        """,
    )
    parser.add_argument("input_path",  help="왜곡 보정할 이미지 폴더 또는 동영상 파일")
    parser.add_argument("output_path", help="보정 결과 저장 경로 (폴더 또는 동영상 파일)")

    calib_group = parser.add_argument_group("캘리브레이션 소스 (하나 이상 지정)")
    calib_group.add_argument(
        "--calib_json", default=None,
        help="기존 캘리브레이션 JSON 경로. 파일이 존재하면 즉시 사용; "
             "없으면 캘리브레이션 후 이 경로에 저장."
    )
    calib_group.add_argument(
        "--calib_video", default=None,
        help="체커보드 캘리브레이션 비디오 파일 경로 (동영상 촬영 모드 권장)"
    )
    calib_group.add_argument(
        "--calib_images", default=None,
        help="체커보드 캘리브레이션 이미지 폴더 경로"
    )

    board_group = parser.add_argument_group("체커보드 설정 (캘리브레이션 시에만 사용)")
    board_group.add_argument(
        "--board_cols", type=int, default=DEFAULT_BOARD_SIZE[0],
        help=f"체커보드 내부 코너 가로 수 (기본: {DEFAULT_BOARD_SIZE[0]})"
    )
    board_group.add_argument(
        "--board_rows", type=int, default=DEFAULT_BOARD_SIZE[1],
        help=f"체커보드 내부 코너 세로 수 (기본: {DEFAULT_BOARD_SIZE[1]})"
    )
    board_group.add_argument(
        "--square_size", type=float, default=DEFAULT_SQUARE_SIZE_MM,
        help=f"체커보드 정사각형 크기 (mm, 기본: {DEFAULT_SQUARE_SIZE_MM})"
    )
    board_group.add_argument(
        "--n_calib_frames", type=int, default=DEFAULT_N_CALIB_FRAMES,
        help=(
            f"비디오를 N등분하여 추출할 프레임 수 (기본: {DEFAULT_N_CALIB_FRAMES}). "
            "천천히 긴 시간 찍은 캘리브레이션 영상에 적합. "
            "--calib_fps 지정 시 무시됨."
        )
    )
    board_group.add_argument(
        "--calib_fps", type=float, default=None,
        help="FPS 기반 프레임 추출 (지정 시 --n_calib_frames 대신 사용)"
    )

    args = parser.parse_args()

    result = run_pipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        calib_json=args.calib_json,
        calib_video=args.calib_video,
        calib_images=args.calib_images,
        board_size=(args.board_cols, args.board_rows),
        square_size_mm=args.square_size,
        n_calib_frames=args.n_calib_frames,
        calib_fps=args.calib_fps,
    )

    calib = result["calibration_params"]
    undist = result["undistort_result"]
    print(f"\n{'='*60}")
    print(f"[완료] 왜곡 보정 파이프라인 종료")
    if "processed_count" in undist:
        print(f"  보정 이미지 수:   {undist['processed_count']}개")
        print(f"  출력 폴더:        {undist['output_dir']}")
    elif "processed_frames" in undist:
        print(f"  보정 프레임 수:   {undist['processed_frames']}개")
        print(f"  출력 파일:        {undist['output_path']}")
    print(f"  캘리브레이션 RMS: {calib.get('rms', 'N/A')}")
    print(f"  fx={calib['fx']:.2f}, fy={calib['fy']:.2f}, "
          f"cx={calib['cx']:.2f}, cy={calib['cy']:.2f}")
    print(f"{'='*60}")
