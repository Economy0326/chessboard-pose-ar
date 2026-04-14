import cv2
import numpy as np
import os

# ============================================
# 카메라 자세 추정 + AR 채워진 3D 나무 오브젝트 시각화
# --------------------------------------------
# 기능:
# 1. output/calibration_result.npz 를 불러옴
# 2. 체스보드 코너를 검출함
# 3. solvePnP로 카메라 자세(rvec, tvec)를 구함
# 4. 채워진 3D 나무 오브젝트를 영상 위에 투영함
# 5. 결과 영상을 output/ar_result.mp4 로 저장함
#
# 사용 체스보드 정보:
# - 내부 코너 수: 7 x 10
# - 한 칸 크기: 25 mm
# ============================================


# -----------------------------
# 사용자 설정값
# -----------------------------
PATTERN_SIZE = (7, 10)             # 체스보드 내부 코너 수
SQUARE_SIZE = 25.0                 # 한 칸 크기 (mm)

INPUT_VIDEO = "data/calibration_video_02.mp4"
OUTPUT_DIR = "output"
CALIB_FILE = os.path.join(OUTPUT_DIR, "calibration_result.npz")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "ar_result.mp4")

# 결과 프레임 샘플 저장용
DEMO_FRAME_1 = os.path.join(OUTPUT_DIR, "demo_frame_01.jpg")
DEMO_FRAME_2 = os.path.join(OUTPUT_DIR, "demo_frame_02.jpg")

# 코너 정밀화 종료 조건
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


def make_object_points(pattern_size, square_size):
    """
    체스보드의 3D 기준 좌표를 생성한다.
    체스보드는 z=0 평면 위에 있다고 가정한다.
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def make_tree_points(square_size):
    """
    체스보드 위에 올릴 3D 나무의 기준 점들을 생성한다.

    구성:
    1. 줄기: 직육면체
    2. 잎 1층: 큰 피라미드
    3. 잎 2층: 작은 피라미드

    좌표계 설명:
    - 체스보드 평면은 z=0
    - z가 음수일수록 체스보드 위로 솟아오른 물체처럼 보임
    """
    s = square_size

    # -----------------------------
    # 줄기(직육면체) 점 8개
    # -----------------------------
    # 바닥면
    t0 = [2.8 * s, 4.0 * s, 0]
    t1 = [3.6 * s, 4.0 * s, 0]
    t2 = [3.6 * s, 4.8 * s, 0]
    t3 = [2.8 * s, 4.8 * s, 0]

    # 윗면
    t4 = [2.8 * s, 4.0 * s, -1.8 * s]
    t5 = [3.6 * s, 4.0 * s, -1.8 * s]
    t6 = [3.6 * s, 4.8 * s, -1.8 * s]
    t7 = [2.8 * s, 4.8 * s, -1.8 * s]

    # -----------------------------
    # 잎 1층(큰 피라미드)
    # 밑면 4개 + 꼭대기 1개
    # -----------------------------
    l10 = [1.8 * s, 3.0 * s, -1.2 * s]
    l11 = [4.6 * s, 3.0 * s, -1.2 * s]
    l12 = [4.6 * s, 5.8 * s, -1.2 * s]
    l13 = [1.8 * s, 5.8 * s, -1.2 * s]
    l14 = [3.2 * s, 4.4 * s, -3.8 * s]

    # -----------------------------
    # 잎 2층(작은 피라미드)
    # 밑면 4개 + 꼭대기 1개
    # -----------------------------
    l20 = [2.2 * s, 3.4 * s, -2.3 * s]
    l21 = [4.2 * s, 3.4 * s, -2.3 * s]
    l22 = [4.2 * s, 5.4 * s, -2.3 * s]
    l23 = [2.2 * s, 5.4 * s, -2.3 * s]
    l24 = [3.2 * s, 4.4 * s, -5.0 * s]

    tree_points = np.float32([
        t0, t1, t2, t3, t4, t5, t6, t7,
        l10, l11, l12, l13, l14,
        l20, l21, l22, l23, l24
    ])
    return tree_points


def fill_polygon(img, pts, color):
    """
    주어진 다각형을 색으로 채운다.
    """
    pts = np.int32(pts).reshape(-1, 1, 2)
    cv2.fillConvexPoly(img, pts, color)


def draw_polygon_line(img, pts, color, thickness=2):
    """
    다각형 외곽선을 그린다.
    """
    pts = np.int32(pts).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, thickness)


def draw_tree(img, imgpts):
    """
    투영된 2D 점들을 이용해 채워진 3D 나무를 그린다.
    면을 채워서 속이 빈 와이어프레임이 아니라
    가득 찬 물체처럼 보이게 만든다.
    """
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # -----------------------------
    # 점 이름 정리
    # -----------------------------
    # 줄기
    t0, t1, t2, t3, t4, t5, t6, t7 = imgpts[0:8]

    # 잎 1층
    l10, l11, l12, l13, l14 = imgpts[8:13]

    # 잎 2층
    l20, l21, l22, l23, l24 = imgpts[13:18]

    # -----------------------------
    # 색상(BGR)
    # -----------------------------
    trunk_color_front = (40, 90, 160)      # 줄기 밝은 갈색
    trunk_color_side = (30, 70, 120)       # 줄기 어두운 갈색
    trunk_color_top = (60, 120, 190)       # 줄기 윗면

    leaf_color_1 = (40, 160, 40)           # 잎 1층 초록
    leaf_color_1_dark = (30, 120, 30)      # 잎 1층 어두운 면
    leaf_color_2 = (60, 200, 60)           # 잎 2층 밝은 초록
    leaf_color_2_dark = (40, 150, 40)      # 잎 2층 어두운 면

    outline_color = (20, 60, 20)           # 외곽선 색

    # -----------------------------
    # 1. 줄기 그리기
    # -----------------------------
    # 앞/옆/윗면을 채워서 직육면체처럼 보이게 함
    fill_polygon(img, [t0, t1, t5, t4], trunk_color_front)
    fill_polygon(img, [t1, t2, t6, t5], trunk_color_side)
    fill_polygon(img, [t4, t5, t6, t7], trunk_color_top)

    draw_polygon_line(img, [t0, t1, t5, t4], (20, 40, 80), 2)
    draw_polygon_line(img, [t1, t2, t6, t5], (20, 40, 80), 2)
    draw_polygon_line(img, [t4, t5, t6, t7], (20, 40, 80), 2)

    # -----------------------------
    # 2. 잎 1층(큰 피라미드) 그리기
    # -----------------------------
    # 밑면
    fill_polygon(img, [l10, l11, l12, l13], leaf_color_1_dark)

    # 옆면 4개
    fill_polygon(img, [l10, l11, l14], leaf_color_1)
    fill_polygon(img, [l11, l12, l14], leaf_color_1_dark)
    fill_polygon(img, [l12, l13, l14], leaf_color_1)
    fill_polygon(img, [l13, l10, l14], leaf_color_1_dark)

    draw_polygon_line(img, [l10, l11, l12, l13], outline_color, 2)
    draw_polygon_line(img, [l10, l11, l14], outline_color, 2)
    draw_polygon_line(img, [l11, l12, l14], outline_color, 2)
    draw_polygon_line(img, [l12, l13, l14], outline_color, 2)
    draw_polygon_line(img, [l13, l10, l14], outline_color, 2)

    # -----------------------------
    # 3. 잎 2층(작은 피라미드) 그리기
    # -----------------------------
    fill_polygon(img, [l20, l21, l22, l23], leaf_color_2_dark)

    fill_polygon(img, [l20, l21, l24], leaf_color_2)
    fill_polygon(img, [l21, l22, l24], leaf_color_2_dark)
    fill_polygon(img, [l22, l23, l24], leaf_color_2)
    fill_polygon(img, [l23, l20, l24], leaf_color_2_dark)

    draw_polygon_line(img, [l20, l21, l22, l23], outline_color, 2)
    draw_polygon_line(img, [l20, l21, l24], outline_color, 2)
    draw_polygon_line(img, [l21, l22, l24], outline_color, 2)
    draw_polygon_line(img, [l22, l23, l24], outline_color, 2)
    draw_polygon_line(img, [l23, l20, l24], outline_color, 2)

    return img


def draw_info_text(img, found, frame_idx):
    """
    디버깅용 텍스트를 화면에 출력한다.
    """
    status_text = f"Chessboard detected: {found}"
    frame_text = f"Frame: {frame_idx}"

    cv2.putText(img, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(img, frame_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 캘리브레이션 결과 파일 불러오기
    if not os.path.exists(CALIB_FILE):
        print("[오류] calibration_result.npz 파일이 없습니다.")
        print("먼저 camera_calibration.py를 실행하세요.")
        return

    calib_data = np.load(CALIB_FILE)
    camera_matrix = calib_data["camera_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]

    print("[정보] 캘리브레이션 결과 로드 완료")
    print("[정보] 입력 영상:", INPUT_VIDEO)

    # 체스보드 3D 기준 좌표
    objp = make_object_points(PATTERN_SIZE, SQUARE_SIZE)

    # AR 나무 3D 좌표
    tree_points = make_tree_points(SQUARE_SIZE)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[오류] 영상을 열 수 없습니다: {INPUT_VIDEO}")
        return

    # 입력 영상 정보 읽기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 일부 영상은 fps가 0으로 읽히는 경우가 있으므로 예외 처리
    if fps <= 0:
        fps = 30.0

    # 결과 영상 저장 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_idx = 0
    detected_count = 0
    saved_demo_1 = False
    saved_demo_2 = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 체스보드 코너 검출
        found, corners = cv2.findChessboardCorners(
            gray,
            PATTERN_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            # 코너 위치를 더 정밀하게 보정
            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                CRITERIA
            )

            # 카메라 자세 추정
            success, rvec, tvec = cv2.solvePnP(
                objp,
                corners2,
                camera_matrix,
                dist_coeffs
            )

            if success:
                # 나무 3D 점들을 2D 영상 좌표로 투영
                imgpts, _ = cv2.projectPoints(
                    tree_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs
                )

                # 체스보드 코너 표시
                cv2.drawChessboardCorners(draw_frame, PATTERN_SIZE, corners2, found)

                # 채워진 3D 나무 그리기
                draw_frame = draw_tree(draw_frame, imgpts)

                detected_count += 1

                # README용 샘플 프레임 저장
                if not saved_demo_1 and frame_idx > 10:
                    cv2.imwrite(DEMO_FRAME_1, draw_frame)
                    saved_demo_1 = True

                if not saved_demo_2 and frame_idx > 60:
                    cv2.imwrite(DEMO_FRAME_2, draw_frame)
                    saved_demo_2 = True

        # 상태 텍스트 출력
        draw_frame = draw_info_text(draw_frame, found, frame_idx)

        # 결과 영상에 기록
        writer.write(draw_frame)

        # 화면으로 확인하고 싶으면 주석 해제
        # cv2.imshow("AR Pose Estimation", draw_frame)
        # key = cv2.waitKey(1)
        # if key == 27:
        #     break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\n========== AR 결과 ==========")
    print(f"총 처리 프레임 수: {frame_idx}")
    print(f"체스보드 검출 성공 프레임 수: {detected_count}")
    print(f"결과 영상 저장: {OUTPUT_VIDEO}")
    if os.path.exists(DEMO_FRAME_1):
        print(f"샘플 이미지 저장: {DEMO_FRAME_1}")
    if os.path.exists(DEMO_FRAME_2):
        print(f"샘플 이미지 저장: {DEMO_FRAME_2}")
    print("============================")


if __name__ == "__main__":
    main()