"""
파노라마 스티칭 (Panorama Stitching)
=====================================
input_1.jpg ~ input_N.jpg (좌→우 순서) N장을 스티칭하여 wide-FOV 파노라마를 생성한다.

파이프라인:
  A. Load            : glob 기반 robust 로드
  B. Cylindrical warp: focal length f 가정으로 원통 투영 (P03 backward warping 패턴)
  C. Pairwise match  : SIFT + ratio test + cross-check (P02 매칭 코드 재사용)
  D. Estimate transform: RANSAC 기반 affine 추정
  E. Chain transforms: 기준 이미지(가운데) 기준 누적 체이닝 + 캔버스 크기 산출
  F. Composite       : backward warping 좌표 맵 계산 (P03) + cv2.remap + feathering blend
  G. Crop            : 검은 여백 제거 + 상하 트림
  H. Save            : focal length 후보별 결과 저장 + best 자동 선택

수업 실습 코드 재사용:
  - P02 Practice 5: ratio test (0.75) + cross-check 매칭 로직
  - P03 Practice 3: backward warping — homogeneous coord grid → 역변환 → 좌표 맵 계산
"""

import os
import re
import glob
import time
import numpy as np
import cv2

# ============================= CONFIG =============================
FOCAL_MM_CANDIDATES = [24, 35, 50, 85]    # 35mm 환산 초점거리 후보 (mm)
RATIO_THRESHOLD = 0.75                     # Lowe's ratio test threshold (P02)
RANSAC_REPROJ_THRESHOLD = 3.0             # RANSAC reprojection threshold (px)
MIN_MATCH_COUNT = 10                       # 최소 매칭 수
SENSOR_WIDTH_MM = 36.0                     # full-frame 센서 폭 (mm)
MAX_CANVAS_DIM = 20000                     # 캔버스 최대 크기 제한 (px)
CROP_ROW_VALID_RATIO = 0.95               # 상하 트림 — 행별 유효 픽셀 비율 임계값
FALLBACK_AFFINE_RATIO = 0.3               # Similarity inlier 비율 < 이 값 → fallback 진입
FALLBACK_HOMOGRAPHY_MIN_RATIO = 0.5      # Homography 채택 조건 ①: inlier/n_matches ≥ 이 값
FALLBACK_HOMOGRAPHY_MARGIN = 1.3         # Homography 채택 조건 ②: inlier ≥ best_ic × 이 값
MAX_INPUT_WIDTH = 2000                     # 입력 이미지 최대 폭 (초과 시 자동 downscale)
DEBUG = False                              # True: 중간 시각화 출력
# ==================================================================


# =====================================================================
#  A. 이미지 로드 — glob 기반 robust 탐색
# =====================================================================
def load_images():
    """
    스크립트와 같은 디렉터리에서 input_*.jpg (또는 .jpeg)를 찾아 숫자 순으로 정렬 후 로드.
    절대경로 하드코딩 없이 BASE_DIR 상대경로 사용.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # jpg, jpeg 모두 탐색
    patterns = [os.path.join(base_dir, "input_*.jpg"),
                os.path.join(base_dir, "input_*.jpeg")]
    found_files = []
    for pattern in patterns:
        found_files.extend(glob.glob(pattern))

    # 중복 제거 (경로 정규화)
    found_files = list({os.path.normpath(f) for f in found_files})

    if len(found_files) == 0:
        raise FileNotFoundError(
            f"'{base_dir}' 에서 input_*.jpg / input_*.jpeg 파일을 찾을 수 없습니다."
        )

    # 파일명에서 숫자를 추출하여 정렬 (input_1, input_2, ..., input_10 올바르게)
    def extract_number(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r"input_(\d+)", basename)
        return int(match.group(1)) if match else 0

    found_files.sort(key=extract_number)

    images = []
    for fpath in found_files:
        img = cv2.imread(fpath)
        if img is None:
            print(f"[경고] '{fpath}' 로드 실패 — 건너뜁니다.")
            continue
        images.append(img)

    if len(images) < 2:
        raise ValueError(f"스티칭하려면 최소 2장이 필요합니다. (로드된 이미지: {len(images)}장)")

    print(f"[A] {len(images)}장 로드 완료  ({images[0].shape[1]}x{images[0].shape[0]})")
    for i, fpath in enumerate(found_files):
        print(f"     [{i+1}] {os.path.basename(fpath)}")
    return images


# =====================================================================
#  B. Cylindrical Warping — P03 backward warping 패턴 + cv2.remap
# =====================================================================
def cylindrical_warp(img, f_px):
    """
    원통 투영 (cylindrical projection).

    출력 좌표 (x_cyl, y_cyl) → 입력 좌표 (x_img, y_img) 역변환:
        θ = (x_cyl - cx) / f
        h = (y_cyl - cy) / f
        x_img = f * tan(θ) + cx
        y_img = f * h / cos(θ) + cy

    P03 backward warping 패턴:
        1. 출력 좌표 그리드 생성 (np.indices)
        2. 역변환 수식으로 입력 좌표 맵(map_x, map_y) 계산
        3. cv2.remap으로 bilinear interpolation 수행
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # --- P03 backward warping 패턴: 출력 좌표 그리드 ---
    out_y_grid, out_x_grid = np.indices((h, w), dtype=np.float32)

    # --- Cylindrical → Planar 역변환 (직접 구현) ---
    theta = (out_x_grid - cx) / f_px          # 수평 각도
    h_cyl = (out_y_grid - cy) / f_px          # 수직 정규화 높이

    cos_theta = np.cos(theta)
    # cos(θ)=0 방지 (극단적 각도)
    cos_theta[cos_theta == 0] = 1e-10

    map_x = (f_px * np.tan(theta) + cx).astype(np.float32)
    map_y = (f_px * h_cyl / cos_theta + cy).astype(np.float32)

    # --- cv2.remap으로 bilinear interpolation (성능 위임) ---
    result = cv2.remap(img, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0))
    return result


# =====================================================================
#  C. 특징점 매칭 — P02 Practice 5 코드 재사용
#     ratio test (0.75) + cross-check
# =====================================================================
def detect_sift(img):
    """SIFT 특징점 + descriptor 추출 (cv2.SIFT_create 사용)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def match_pair(des1, des2, kp1, kp2):
    """
    P02 Practice 5 매칭 로직 재사용:
      1) BFMatcher knnMatch (k=2) → ratio test (threshold 0.75)
      2) 역방향 knnMatch → cross-check
    원본 P02 코드는 전체 distance_matrix를 직접 계산했으나,
    고해상도 이미지의 descriptor 수가 많을 때 메모리 이슈 발생 가능 →
    BFMatcher.knnMatch로 대체하되, ratio test + cross-check 로직은 동일하게 유지.
    """
    if des1 is None or des2 is None:
        print("  [경고] descriptor가 None — 매칭 불가")
        return None

    if len(des1) < 2 or len(des2) < 2:
        print(f"  [경고] descriptor 부족 (img1: {len(des1)}, img2: {len(des2)})")
        return None

    bf = cv2.BFMatcher(cv2.NORM_L2)

    # ---- P02 Ratio Test (forward 방향) ----
    # P02 원본: sorted_indices = np.argsort(distance_matrix[i])
    #           if best_distance < ratio_threshold * second_best_distance:
    knn_12 = bf.knnMatch(des1, des2, k=2)
    ratio_matches_forward = {}   # queryIdx → DMatch
    for match_pair_result in knn_12:
        if len(match_pair_result) < 2:
            continue
        m, n = match_pair_result
        # P02 코드: if best_distance < ratio_threshold * second_best_distance
        if m.distance < RATIO_THRESHOLD * n.distance:
            ratio_matches_forward[m.queryIdx] = m

    # ---- P02 Cross-Check (reverse 방향) ----
    # P02 원본: reverse_best_index = int(np.argmin(distance_matrix[:, j]))
    #           if reverse_best_index == match.queryIdx
    knn_21 = bf.knnMatch(des2, des1, k=2)
    ratio_matches_reverse = {}   # des2 idx → des1 idx (P02 원본과 동일한 방향)
    for match_pair_result in knn_21:
        if len(match_pair_result) < 2:
            continue
        m, n = match_pair_result
        if m.distance < RATIO_THRESHOLD * n.distance:
            # m.queryIdx → des2의 인덱스,  m.trainIdx → des1의 인덱스
            # P02 원본: reverse_best_index = argmin(distance_matrix[:, j])
            #   → des2[j]의 best match가 des1의 어떤 인덱스인지 저장
            ratio_matches_reverse[m.queryIdx] = m.trainIdx

    # Cross-check: 양방향 모두 매칭된 쌍만 최종 선택
    cross_matches = []
    for q_idx, fwd_match in ratio_matches_forward.items():
        t_idx = fwd_match.trainIdx  # des2에서의 인덱스
        # 역방향에서 des2[t_idx]의 best가 des1[q_idx]를 가리키는지 확인
        # P02 원본: if reverse_best_index == match.queryIdx
        if t_idx in ratio_matches_reverse and ratio_matches_reverse[t_idx] == q_idx:
            cross_matches.append(fwd_match)

    cross_matches = sorted(cross_matches, key=lambda m: m.distance)


    if len(cross_matches) < MIN_MATCH_COUNT:
        print(f"  [경고] 매칭 수 부족 ({len(cross_matches)} < {MIN_MATCH_COUNT})")
        return None

    return cross_matches


# =====================================================================
#  D. 변환 추정 — RANSAC (similarity → affine/homography 병렬 fallback)
# =====================================================================
def estimate_transform(kp1, kp2, matches):
    """
    인접쌍 변환행렬 추정 (img1 → img2 방향).
    Fallback 전략:
      1차: Similarity (4-DOF) — cylindrical warp 후 이상적 경우
      2차: Similarity 부족 시 Affine(6-DOF) + Homography(8-DOF) 둘 다 시도 → 최선 채택
           (Homography는 DOF가 높아 overfit 가능 → inlier 1.3배 이상 & det 안정일 때만)
    반환: 3×3 homogeneous 행렬 + inlier 수.
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    n_matches = len(matches)

    # --- 1차: Similarity (4-DOF: 회전 + 균일 스케일 + 이동) ---
    M_sim, inliers_sim = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD
    )
    ic_sim = int(np.sum(inliers_sim)) if inliers_sim is not None else 0
    ratio_sim = ic_sim / n_matches if n_matches > 0 else 0

    # Similarity가 충분히 좋으면 바로 반환 (대부분의 정상 케이스)
    if M_sim is not None and ratio_sim >= FALLBACK_AFFINE_RATIO:
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = M_sim
        return H, ic_sim

    # --- Similarity 부족: Affine + Homography 병렬 시도 → 최선 채택 ---
    best_ic, best_M, best_method, best_is_3x3 = ic_sim, M_sim, "Similarity(4-DOF)", False

    # Affine (6-DOF)
    M_aff, inliers_aff = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD
    )
    ic_aff = int(np.sum(inliers_aff)) if inliers_aff is not None else 0
    if M_aff is not None and ic_aff > best_ic:
        best_ic, best_M, best_method, best_is_3x3 = ic_aff, M_aff, "Affine(6-DOF)", False

    # Homography (8-DOF) — overfit 방지 가드 포함
    H_hom, inliers_hom = cv2.findHomography(
        src_pts, dst_pts,
        cv2.RANSAC,
        RANSAC_REPROJ_THRESHOLD
    )
    ic_hom = int(np.sum(inliers_hom)) if inliers_hom is not None else 0
    if H_hom is not None and ic_hom > best_ic:
        # Homography 채택 조건: (1) inlier 1.3배 이상, (2) det 안정성
        det = np.linalg.det(H_hom[:2, :2])
        if (ic_hom >= best_ic * FALLBACK_HOMOGRAPHY_MARGIN
                or ic_hom / n_matches >= FALLBACK_HOMOGRAPHY_MIN_RATIO) \
                and 0.1 < abs(det) < 10:
            best_ic, best_M, best_method, best_is_3x3 = ic_hom, H_hom, "Homography(8-DOF)", True

    if best_M is None:
        print("  [경고] 변환행렬 추정 실패 (모든 방법)")
        return None, 0

    if best_method != "Similarity(4-DOF)":
        print(f"    → {best_method} fallback 사용 (similarity {ic_sim} → {best_ic} inliers)")

    if best_is_3x3:
        return best_M, best_ic

    # 2×3 → 3×3 homogeneous 행렬로 확장 (체이닝 용)
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = best_M
    return H, best_ic


# =====================================================================
#  E. 변환 체이닝 — 기준 이미지 기준 누적 + 캔버스 크기 산출
# =====================================================================
def chain_transforms(pairwise_H, num_images, image_shapes):
    """
    인접쌍 변환행렬을 기준 이미지(가운데)로 누적 체이닝.

    pairwise_H[i]: img_i → img_{i+1} 방향 변환.
    ref_idx = num_images // 2 (가운데 이미지).

    global_H[ref]: Identity
    global_H[i < ref]: H_i→i+1 @ H_{i+1→i+2} @ ... @ H_{ref-1→ref}  (순방향 누적)
    global_H[i > ref]: inv(H_{ref→ref+1}) @ inv(H_{ref+1→ref+2}) @ ... (역변환 누적)

    캔버스 크기: 모든 이미지 코너를 global_H로 변환 → min/max → offset 보정.
    """
    ref_idx = num_images // 2
    global_H = [None] * num_images
    global_H[ref_idx] = np.eye(3, dtype=np.float64)

    # ref 왼쪽 (i < ref): img_i → ref 방향으로 순방향 체이닝
    for i in range(ref_idx - 1, -1, -1):
        # pairwise_H[i]: img_i → img_{i+1}
        # global_H[i] = pairwise_H[i] @ global_H[i+1]  → img_i를 ref 좌표계로
        if pairwise_H[i] is not None and global_H[i + 1] is not None:
            global_H[i] = global_H[i + 1] @ pairwise_H[i]
        else:
            print(f"  [경고] 이미지 {i+1} 체이닝 실패 — Identity 대체")
            global_H[i] = np.eye(3, dtype=np.float64)

    # ref 오른쪽 (i > ref): inv(pairwise_H[i-1]) 누적
    for i in range(ref_idx + 1, num_images):
        # pairwise_H[i-1]: img_{i-1} → img_i
        # global_H[i] = inv(pairwise_H[i-1]) @ global_H[i-1]  → img_i를 ref 좌표계로
        if pairwise_H[i - 1] is not None and global_H[i - 1] is not None:
            H_inv = np.linalg.inv(pairwise_H[i - 1])
            global_H[i] = global_H[i - 1] @ H_inv
        else:
            print(f"  [경고] 이미지 {i+1} 체이닝 실패 — Identity 대체")
            global_H[i] = np.eye(3, dtype=np.float64)

    # 캔버스 크기 계산: 모든 이미지 코너를 변환
    all_corners = []
    for i in range(num_images):
        h, w = image_shapes[i][:2]
        corners = np.array([
            [0,   0,   1],
            [w-1, 0,   1],
            [w-1, h-1, 1],
            [0,   h-1, 1]
        ], dtype=np.float64).T  # 3×4

        transformed = global_H[i] @ corners  # 3×4
        transformed /= transformed[2:3, :]   # homogeneous divide
        all_corners.append(transformed[:2, :].T)  # 4×2

    all_corners = np.vstack(all_corners)

    x_min = int(np.floor(all_corners[:, 0].min()))
    x_max = int(np.ceil(all_corners[:, 0].max()))
    y_min = int(np.floor(all_corners[:, 1].min()))
    y_max = int(np.ceil(all_corners[:, 1].max()))

    canvas_w = x_max - x_min + 1
    canvas_h = y_max - y_min + 1

    # offset 행렬: 음수 좌표를 보정
    offset = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0,  1]
    ], dtype=np.float64)

    # 캔버스 크기 상한선 — 초과 시 scale matrix도 함께 적용
    if canvas_w > MAX_CANVAS_DIM or canvas_h > MAX_CANVAS_DIM:
        print(f"  [경고] 캔버스 크기 초과 ({canvas_w}x{canvas_h}) → 스케일 적용")
        scale = min(MAX_CANVAS_DIM / canvas_w, MAX_CANVAS_DIM / canvas_h)
        canvas_w = int(canvas_w * scale)
        canvas_h = int(canvas_h * scale)
        scale_matrix = np.array([
            [scale, 0,     0],
            [0,     scale, 0],
            [0,     0,     1]
        ], dtype=np.float64)
        offset = scale_matrix @ offset  # offset에 scale 결합

    # 모든 global_H에 offset(+scale) 적용
    for i in range(num_images):
        global_H[i] = offset @ global_H[i]

    canvas_size = (canvas_h, canvas_w)
    print(f"  [E] 캔버스 크기: {canvas_w}x{canvas_h},  기준 이미지: input_{ref_idx+1}")
    return global_H, canvas_size


# =====================================================================
#  F. 합성 — P03 backward warping 좌표 맵 + cv2.remap + feathering
# =====================================================================
def composite_panorama(images, global_H, canvas_size):
    """
    각 이미지를 캔버스에 합성.

    P03 backward warping 패턴:
        1. 출력(캔버스) 좌표 그리드 생성: np.indices
        2. 각 이미지의 global_H 역행렬로 입력 좌표 맵 계산:
           M_inv = np.linalg.inv(H)
           in_coords = M_inv @ out_coords   ← P03 코드 구조
        3. cv2.remap으로 bilinear interpolation 수행

    Blending: distance-transform 기반 feathering
        각 warped 이미지의 유효 마스크에 distanceTransform 적용 →
        가중치로 사용하여 seam 최소화.
    """
    canvas_h, canvas_w = canvas_size

    # 누적 가중치 합 + 가중 픽셀 합 (float32로 메모리 절약, blending에 충분한 정밀도)
    weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    color_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)

    # --- P03 backward warping 패턴: 출력 좌표 그리드 (한 번만 생성) ---
    out_y_grid, out_x_grid = np.indices((canvas_h, canvas_w), dtype=np.float32)

    for i, (img, H) in enumerate(zip(images, global_H)):
        h, w = img.shape[:2]

        # --- P03 핵심: 역행렬로 입력 좌표 계산 ---
        # P03 원본: M_inv = np.linalg.inv(M)
        #           in_coords = M_inv @ out_coords
        H_inv = np.linalg.inv(H)

        # homogeneous 좌표 → 역변환 → 입력 좌표 맵
        # P03 원본: out_coords = np.stack((out_x.flatten(), out_y.flatten(), ones))
        #           in_coords = M_inv @ out_coords
        #           in_x = in_coords[0] / in_coords[2]
        #           in_y = in_coords[1] / in_coords[2]
        map_x = (H_inv[0, 0] * out_x_grid + H_inv[0, 1] * out_y_grid + H_inv[0, 2]) / \
                (H_inv[2, 0] * out_x_grid + H_inv[2, 1] * out_y_grid + H_inv[2, 2])
        map_y = (H_inv[1, 0] * out_x_grid + H_inv[1, 1] * out_y_grid + H_inv[1, 2]) / \
                (H_inv[2, 0] * out_x_grid + H_inv[2, 1] * out_y_grid + H_inv[2, 2])

        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # --- cv2.remap으로 bilinear interpolation (성능 위임) ---
        warped = cv2.remap(img, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0))

        # 유효 마스크: warped 이미지에서 검은색이 아닌 영역
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = (gray_warped > 1).astype(np.uint8)  # > 1: crop_black_borders와 threshold 통일

        # --- Feathering: distance-transform 기반 가중치 ---
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist = dist.astype(np.float32)

        # 가중 누적
        weight_sum += dist
        color_sum += dist[:, :, np.newaxis] * warped.astype(np.float32)

        print(f"    이미지 {i+1}/{len(images)} 합성 완료")

    # --- 정규화: result = Σ(w_i * img_i) / Σ(w_i) ---
    # 분모 0 방지
    valid = weight_sum > 0
    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for c in range(3):
        channel = color_sum[:, :, c]
        out_channel = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        out_channel[valid] = channel[valid] / weight_sum[valid]
        result[:, :, c] = np.clip(out_channel, 0, 255).astype(np.uint8)

    return result


# =====================================================================
#  G. 검은 여백 제거 + 상하 트림
# =====================================================================
def crop_black_borders(panorama):
    """
    검은 여백 제거:
      1) 그레이스케일 → threshold → non-zero 영역의 bounding rect
      2) 상하 추가 트림: 행별 유효 픽셀 비율 < CROP_ROW_VALID_RATIO 인 상·하단 행 제거
    """
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 1단계: non-zero column/row 범위
    coords = cv2.findNonZero(thresh)
    if coords is None:
        print("  [경고] 유효 픽셀 없음 — crop 생략")
        return panorama

    x, y, w, h = cv2.boundingRect(coords)
    cropped = panorama[y:y+h, x:x+w]

    # 2단계: 상하 추가 트림 — 유효 픽셀 비율이 낮은 행 제거
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    row_valid = np.mean(gray_cropped > 0, axis=1)  # 행별 유효 비율

    valid_rows = np.where(row_valid >= CROP_ROW_VALID_RATIO)[0]
    if len(valid_rows) > 0:
        top = valid_rows[0]
        bottom = valid_rows[-1] + 1
        cropped = cropped[top:bottom, :]

    print(f"  [G] Crop 완료: {panorama.shape[1]}x{panorama.shape[0]} → {cropped.shape[1]}x{cropped.shape[0]}")
    return cropped


# =====================================================================
#  H. 결과 평가 + best 선택
# =====================================================================
def evaluate_result(panorama, inlier_counts):
    """
    평가 지표:
      1) RANSAC inlier 합계 (정규화)
      2) Crop 후 유효 픽셀 비율
      3) 가중 합산 → 단일 score
    """
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    fill_ratio = np.mean(gray > 0)
    total_inliers = sum(inlier_counts)
    # score: inlier 비중 70% + fill 비중 30%
    score = 0.7 * (total_inliers / max(1, len(inlier_counts) * 100)) + 0.3 * fill_ratio
    return score, total_inliers, fill_ratio


# =====================================================================
#  디버그 시각화 (DEBUG=True일 때만)
# =====================================================================
def debug_show(title, img, max_width=1200):
    """디버그용 이미지 표시."""
    if not DEBUG:
        return
    try:
        import matplotlib.pyplot as plt
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            img = cv2.resize(img, (max_width, int(h * scale)))
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
    except ImportError:
        pass


# =====================================================================
#  main — 전체 파이프라인 오케스트레이션
# =====================================================================
def main():
    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  파노라마 스티칭 (Panorama Stitching)")
    print("=" * 60)

    # ---- A. 이미지 로드 ----
    images = load_images()
    num_images = len(images)
    img_h, img_w = images[0].shape[:2]

    # ---- 고해상도 입력 자동 downscale (메모리/속도 최적화) ----
    if img_w > MAX_INPUT_WIDTH:
        scale_input = MAX_INPUT_WIDTH / img_w
        images = [cv2.resize(img, None, fx=scale_input, fy=scale_input,
                             interpolation=cv2.INTER_AREA) for img in images]
        img_h, img_w = images[0].shape[:2]
        print(f"  [A] 고해상도 자동 다운스케일 → {img_w}x{img_h}")

    # ---- 원본 이미지에서 SIFT는 focal마다 다시 계산 (cylindrical warp 결과가 다르므로) ----

    results = {}  # f_mm → (panorama, score, inliers, fill)

    for f_idx, f_mm in enumerate(FOCAL_MM_CANDIDATES):
        print(f"\n{'─' * 60}")
        print(f"  [{f_idx+1}/{len(FOCAL_MM_CANDIDATES)}] Focal length: {f_mm}mm")
        print(f"{'─' * 60}")

        # ---- B. Cylindrical Warping ----
        f_px = img_w * f_mm / SENSOR_WIDTH_MM
        print(f"  [B] f_px = {f_px:.1f}  ({f_mm}mm → {img_w}px 센서)")

        warped_images = []
        for i, img in enumerate(images):
            w_img = cylindrical_warp(img, f_px)
            warped_images.append(w_img)

        print(f"  [B] Cylindrical warp 완료 ({num_images}장)")
        debug_show(f"Cylindrical Warp (f={f_mm}mm) - Image 1", warped_images[0])

        # ---- C. Pairwise SIFT 매칭 (P02 재사용) ----
        print(f"  [C] SIFT 매칭 시작...")

        # SIFT 검출
        sift_results = []
        for i, w_img in enumerate(warped_images):
            kp, des = detect_sift(w_img)
            sift_results.append((kp, des))
            print(f"    이미지 {i+1}: {len(kp)} keypoints")

        # 인접쌍 매칭
        pairwise_matches = []
        for i in range(num_images - 1):
            kp1, des1 = sift_results[i]
            kp2, des2 = sift_results[i + 1]
            matches = match_pair(des1, des2, kp1, kp2)
            pairwise_matches.append(matches)
            if matches is not None:
                print(f"    쌍 ({i+1},{i+2}): {len(matches)} matches")
            else:
                print(f"    쌍 ({i+1},{i+2}): 매칭 실패")

        # ---- D. 변환 추정 (RANSAC) ----
        print(f"  [D] 변환 추정...")
        pairwise_H = []
        inlier_counts = []
        all_ok = True

        for i in range(num_images - 1):
            if pairwise_matches[i] is None:
                pairwise_H.append(None)
                inlier_counts.append(0)
                all_ok = False
                continue

            kp1, _ = sift_results[i]
            kp2, _ = sift_results[i + 1]
            H, inliers = estimate_transform(kp1, kp2, pairwise_matches[i])
            pairwise_H.append(H)
            inlier_counts.append(inliers)
            if H is not None:
                print(f"    쌍 ({i+1},{i+2}): inliers = {inliers}")
            else:
                all_ok = False

        if not all_ok:
            print(f"  [경고] 일부 쌍에서 매칭/변환 실패 — 결과 품질 저하 가능")

        # ---- E. 변환 체이닝 + 캔버스 계산 ----
        image_shapes = [img.shape for img in warped_images]
        global_H, canvas_size = chain_transforms(pairwise_H, num_images, image_shapes)

        # ---- F. 합성 (backward warp + feathering) ----
        print(f"  [F] 합성 시작...")
        panorama = composite_panorama(warped_images, global_H, canvas_size)
        debug_show(f"Composite (f={f_mm}mm)", panorama)

        # ---- G. Crop ----
        panorama = crop_black_borders(panorama)
        debug_show(f"Cropped (f={f_mm}mm)", panorama)

        # ---- 평가 ----
        score, total_inliers, fill_ratio = evaluate_result(panorama, inlier_counts)

        # ---- 저장 ----
        out_path = os.path.join(base_dir, f"result_f{f_mm}.jpg")
        cv2.imwrite(out_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [H] 저장: {os.path.basename(out_path)}")
        print(f"       Score={score:.4f}  Inliers={total_inliers}  Fill={fill_ratio:.2%}")

        results[f_mm] = (panorama, score, total_inliers, fill_ratio)

    # ---- Best 선택 + result_panorama.jpg 저장 ----
    print(f"\n{'=' * 60}")
    print(f"  결과 요약")
    print(f"{'=' * 60}")
    print(f"  {'f_mm':>6}  {'Score':>8}  {'Inliers':>8}  {'Fill':>8}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")

    best_mm = None
    best_score = -1
    for f_mm, (pano, score, inliers, fill) in results.items():
        marker = ""
        if score > best_score:
            best_score = score
            best_mm = f_mm
        print(f"  {f_mm:>4}mm  {score:>8.4f}  {inliers:>8}  {fill:>7.2%}")

    if best_mm is not None:
        best_pano = results[best_mm][0]
        best_path = os.path.join(base_dir, "result_panorama.jpg")
        cv2.imwrite(best_path, best_pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\n  ★ Best: f={best_mm}mm (score={best_score:.4f})")
        print(f"  ★ 저장: result_panorama.jpg")

    # ===== 보고서 기입용 요약 (표1 + 분석1) =====
    print(f"\n{'='*60}")
    print("  [보고서 기입용] 표 1 - 초점거리별 결과")
    print(f"{'='*60}")
    print(f"  {'f(mm)':>6} {'f_px':>9} {'Score':>9} {'Inliers':>9} {'Fill(%)':>9}")
    for f_mm in FOCAL_MM_CANDIDATES:
        if f_mm not in results:
            continue
        f_px = img_w * f_mm / SENSOR_WIDTH_MM
        _, score, inliers, fill = results[f_mm]
        print(f"  {f_mm:>6} {f_px:>9.1f} {score:>9.4f} {inliers:>9} {fill*100:>8.2f}")
    if best_mm is not None:
        print(f"\n  분석(1) Best 초점거리 = {best_mm}mm,  Inlier 합계 = {results[best_mm][2]}")
    print(f"{'='*60}")


    elapsed = time.time() - start_time
    print(f"\n  총 실행 시간: {elapsed:.1f}초")
    print("=" * 60)


if __name__ == "__main__":
    main()
