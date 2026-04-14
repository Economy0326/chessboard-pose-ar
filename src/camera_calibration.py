import cv2
import numpy as np
import os

# ============================================
# 카메라 캘리브레이션 스크립트
# --------------------------------------------
# 기능:
# 1. data 폴더의 캘리브레이션 영상들을 읽음
# 2. 일정 간격으로 프레임을 추출함
# 3. 체스보드 코너를 검출함
# 4. 카메라 캘리브레이션을 수행함
# 5. 결과를 output/calibration_result.npz 로 저장함
#
# 사용 체스보드 정보:
# - 내부 코너 수: 7 x 10
# - 한 칸 크기: 25 mm
# ============================================


# -----------------------------
# 사용자 설정값
# -----------------------------
PATTERN_SIZE = (7, 10)      # 체스보드 내부 코너 수 (가로, 세로)
SQUARE_SIZE = 25.0          # 한 칸 크기 (mm)

VIDEO_PATHS = [
    "data/calibration_video_01.mp4",
    "data/calibration_video_02.mp4",
]

FRAME_INTERVAL = 10         # 몇 프레임마다 1장씩 사용할지
OUTPUT_DIR = "output"
PREVIEW_DIR = os.path.join(OUTPUT_DIR, "corners_preview")
SAVE_FILE = os.path.join(OUTPUT_DIR, "calibration_result.npz")

# 코너 정밀화 종료 조건
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


def create_folders():
    """결과 저장 폴더를 생성한다."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)


def make_object_points(pattern_size, square_size):
    """
    체스보드의 3D 기준 좌표를 생성한다.
    체스보드는 z=0 평면 위에 놓여 있다고 가정한다.

    예:
    (0, 0, 0), (25, 0, 0), (50, 0, 0), ...
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def process_video(video_path, objp, objpoints, imgpoints, preview_start_idx):
    """
    하나의 영상에서 프레임을 추출하고 체스보드 코너를 검출한다.

    매개변수:
    - video_path: 입력 영상 경로
    - objp: 체스보드 3D 기준 좌표
    - objpoints: 누적 3D 좌표 리스트
    - imgpoints: 누적 2D 이미지 좌표 리스트
    - preview_start_idx: 미리보기 이미지 저장 시작 번호

    반환값:
    - image_size: 이미지 크기 (width, height)
    - success_count: 성공한 프레임 수
    - preview_idx: 저장된 마지막 미리보기 번호
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[경고] 영상을 열 수 없습니다: {video_path}")
        return None, 0, preview_start_idx

    frame_idx = 0
    success_count = 0
    preview_idx = preview_start_idx
    image_size = None

    print(f"\n[정보] 영상 처리 시작: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 모든 프레임을 다 쓰지 않고 일정 간격으로만 사용
        if frame_idx % FRAME_INTERVAL != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        # 체스보드 코너 검출
        found, corners = cv2.findChessboardCorners(
            gray,
            PATTERN_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            # 코너를 더 정밀하게 보정
            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                CRITERIA
            )

            # 캘리브레이션용 3D/2D 점 저장
            objpoints.append(objp)
            imgpoints.append(corners2)
            success_count += 1

            # 검출 결과를 미리보기 이미지로 저장
            preview = frame.copy()
            cv2.drawChessboardCorners(preview, PATTERN_SIZE, corners2, found)

            preview_name = os.path.join(PREVIEW_DIR, f"corners_{preview_idx:03d}.jpg")
            cv2.imwrite(preview_name, preview)
            preview_idx += 1

        frame_idx += 1

    cap.release()
    print(f"[정보] 코너 검출 성공 프레임 수: {success_count}")
    return image_size, success_count, preview_idx


def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    """
    재투영 오차(RMSE)를 계산한다.
    """
    total_error = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        projected_imgpoints, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], projected_imgpoints, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(projected_imgpoints)

    rmse = np.sqrt(total_error / total_points)
    return rmse


def main():
    create_folders()

    # 체스보드의 3D 기준 좌표 생성
    objp = make_object_points(PATTERN_SIZE, SQUARE_SIZE)

    # 여러 장면에서 누적될 3D/2D 대응점 리스트
    objpoints = []   # 실제 세계 좌표
    imgpoints = []   # 영상 이미지 좌표

    image_size = None
    total_success = 0
    preview_idx = 0

    # 모든 영상에 대해 코너 검출 수행
    for video_path in VIDEO_PATHS:
        size, success_count, preview_idx = process_video(
            video_path, objp, objpoints, imgpoints, preview_idx
        )
        if size is not None:
            image_size = size
        total_success += success_count

    # 유효한 프레임 수가 너무 적으면 캘리브레이션이 불안정함
    if total_success < 8:
        print("\n[오류] 코너 검출에 성공한 프레임 수가 너무 적습니다.")
        print("체스보드가 전체적으로 잘 보이도록 다시 촬영하거나, FRAME_INTERVAL 값을 줄여보세요.")
        return

    if image_size is None:
        print("\n[오류] 유효한 이미지 크기를 얻지 못했습니다.")
        return

    print("\n[정보] 카메라 캘리브레이션 수행 중...")

    # 카메라 캘리브레이션 수행
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    # 재투영 오차 계산
    rmse = compute_reprojection_error(
        objpoints,
        imgpoints,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs
    )

    # 결과 저장
    np.savez(
        SAVE_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rmse=rmse,
        pattern_size=np.array(PATTERN_SIZE),
        square_size=np.array(SQUARE_SIZE)
    )

    # 결과 출력
    print("\n========== 캘리브레이션 결과 ==========")
    print(f"성공 프레임 수: {total_success}")
    print(f"RMSE: {rmse:.6f}")
    print("\n[카메라 내부 파라미터 행렬(camera_matrix)]")
    print(camera_matrix)
    print("\n[왜곡 계수(dist_coeffs)]")
    print(dist_coeffs.ravel())
    print("\n저장 파일:")
    print(SAVE_FILE)
    print("======================================")


if __name__ == "__main__":
    main()