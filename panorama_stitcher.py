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
  - P03 Practice 3: backward warping -- homogeneous coord grid → 역변환 → 좌표 맵 계산
"""

import os
import re
import glob
import json
import time
import numpy as np
import cv2

# ============================= CONFIG =============================
FOCAL_MM_CANDIDATES = [28, 35, 50]            # 35mm 환산 초점거리 후보 (EXIF 없을 때 fallback)
RATIO_THRESHOLD = 0.75                     # Lowe's ratio test threshold (P02)
RANSAC_REPROJ_THRESHOLD = 2.5             # RANSAC reprojection threshold (px) -- 정밀 inlier 선택
MIN_MATCH_COUNT = 10                       # 최소 매칭 수
SENSOR_WIDTH_MM = 36.0                     # full-frame 센서 폭 (mm)
MAX_CANVAS_DIM = 15000                     # 캔버스 최대 크기 제한 (px) -- 5장 스티칭 허용
MAX_CANVAS_PIXELS = 10_000_000             # 합성 메모리 예산: float pyramid 기준 약 1,000만 px
CROP_ROW_VALID_RATIO = 0.85               # 상하 트림 -- 행별 유효 픽셀 비율 임계값 (완화)
FALLBACK_AFFINE_RATIO = 0.3               # Similarity inlier 비율 < 이 값 → fallback 진입
FALLBACK_HOMOGRAPHY_MIN_RATIO = 0.5      # Homography 채택 조건 ①: inlier/n_matches ≥ 이 값
FALLBACK_HOMOGRAPHY_MARGIN = 1.3         # Homography 채택 조건 ②: inlier ≥ best_ic × 이 값
MAX_INPUT_WIDTH = 3000                     # 입력 이미지 최대 폭 -- 화질 보존 위해 상향
BLEND_BANDS = 6                            # Laplacian pyramid 블렌딩 레벨 수 -- 세밀한 주파수 분리
BLEND_BLUR_KSIZE = 21                     # Distance transform 후 Gaussian blur 커널 크기
MASK_ERODE_PX = 2                         # 마스크 erode 크기 -- remap 경계 artifact 제거
CLAHE_CLIP_LIMIT = 1.5                    # CLAHE 대비 제한 (SIFT 전용 -- 합성엔 미적용)
CLAHE_GRID_SIZE = 8                       # CLAHE 타일 크기 (px)
UNSHARP_AMOUNT = 0.0                      # Unsharp Masking 강도 (0=OFF -- 원본 화질 유지)
UNSHARP_SIGMA = 1.5                       # Unsharp Masking 가우시안 시그마
BILATERAL_D = 0                           # Bilateral filter diameter (0=OFF -- 원본 화질 유지)
BILATERAL_SIGMA_COLOR = 50                # Bilateral 색상 공간 시그마
BILATERAL_SIGMA_SPACE = 50                # Bilateral 좌표 공간 시그마
SATURATION_BOOST = 1.0                    # 채도 boost (1.0=OFF -- 원본 색감 유지)
GAIN_CLIP_MIN = 0.80                      # Gain compensation 하한 -- 확장
GAIN_CLIP_MAX = 1.25                      # Gain compensation 상한 -- 확장
INTERP_METHOD = cv2.INTER_LANCZOS4        # 리샘플링 interpolation -- 고품질 Lanczos
DEBUG = False                              # True: 중간 시각화 출력
# ==================================================================


# =====================================================================
#  A. 이미지 로드 -- glob 기반 robust 탐색
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
            print(f"[경고] '{fpath}' 로드 실패 -- 건너뜁니다.")
            continue
        images.append(img)

    if len(images) < 2:
        raise ValueError(f"스티칭하려면 최소 2장이 필요합니다. (로드된 이미지: {len(images)}장)")

    print(f"[A] {len(images)}장 로드 완료  ({images[0].shape[1]}x{images[0].shape[0]})")
    for i, fpath in enumerate(found_files):
        print(f"     [{i+1}] {os.path.basename(fpath)}")
    return images, found_files


# =====================================================================
#  A-0. EXIF Focal Length 추출 -- 자동 초점거리 감지
# =====================================================================
def get_exif_focal_length(filepath):
    """
    JPEG EXIF에서 35mm 환산 focal length를 추출.
    1차: FocalLengthIn35mmFilm (tag 41989) -- 직접 35mm 환산값
    2차: FocalLength (tag 37386) + crop factor 계산
    실패 시 None 반환.
    """
    try:
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS

        pil_img = PILImage.open(filepath)
        exif_data = pil_img._getexif()
        if exif_data is None:
            return None

        # Tag 번호 → 이름 매핑
        exif_dict = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            exif_dict[tag_name] = value

        # 1차: FocalLengthIn35mmFilm (이미 35mm 환산)
        if 'FocalLengthIn35mmFilm' in exif_dict:
            f35 = float(exif_dict['FocalLengthIn35mmFilm'])
            if f35 > 0:
                return f35

        # 2차: FocalLength (실제 초점거리) -- crop factor 필요
        if 'FocalLength' in exif_dict:
            fl = exif_dict['FocalLength']
            # IFDRational 또는 tuple 형태 처리
            if hasattr(fl, 'numerator'):
                focal_mm = float(fl.numerator) / float(fl.denominator) if fl.denominator != 0 else 0
            elif isinstance(fl, tuple):
                focal_mm = float(fl[0]) / float(fl[1]) if fl[1] != 0 else 0
            else:
                focal_mm = float(fl)

            if focal_mm > 0:
                # ExifImageWidth로 crop factor 추정 시도
                # 대부분의 스마트폰: crop factor ≈ 5~7
                # 기본 crop factor = 5.6 (일반적인 스마트폰 1/2.55" 센서)
                crop_factor = 5.6
                f35_equiv = focal_mm * crop_factor
                # 합리적 범위 체크 (20~200mm)
                if 20 <= f35_equiv <= 200:
                    return f35_equiv

        return None

    except Exception as e:
        print(f"  [EXIF] 읽기 실패: {e}")
        return None


def detect_exif_focal_lengths(file_paths):
    """
    모든 입력 이미지에서 EXIF focal length를 추출하여 대표값 결정.
    대부분의 카메라는 같은 렌즈로 촬영하므로 중앙값 사용.
    """
    focal_lengths = []
    for fpath in file_paths:
        f = get_exif_focal_length(fpath)
        if f is not None:
            focal_lengths.append(f)

    if len(focal_lengths) == 0:
        return None

    # 중앙값 사용 (이상치 제거)
    median_f = float(np.median(focal_lengths))
    print(f"  [EXIF] {len(focal_lengths)}/{len(file_paths)}장에서 focal length 검출")
    print(f"  [EXIF] 추출값: {focal_lengths}")
    print(f"  [EXIF] 대표값: {median_f:.1f}mm (35mm 환산)")
    return median_f


# =====================================================================
#  A-1. CLAHE 대비 향상 -- LAB L채널 국소 대비 균일화
# =====================================================================
def apply_clahe(img):
    """
    BGR → LAB → L채널 CLAHE → LAB → BGR 변환.
    색상(a, b 채널)은 유지하고 명도(L)만 국소 히스토그램 균일화.
    입력 전처리와 최종 파노라마 후처리 양쪽에 재사용.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                             tileGridSize=(CLAHE_GRID_SIZE, CLAHE_GRID_SIZE))
    l_enhanced = clahe.apply(l)
    merged = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# =====================================================================
#  A-2. Unsharp Masking -- 선명도 향상 (CLAHE 대비 노이즈 증폭 없음)
# =====================================================================
def apply_unsharp_mask(img, amount=UNSHARP_AMOUNT, sigma=UNSHARP_SIGMA):
    """
    Unsharp Masking: blurred를 빼서 high-frequency detail을 강조.
    amount: 강도 (0=효과 없음, 1=최대 강조)
    sigma: Gaussian blur 시그마 (클수록 넓은 범위 blur)
    CLAHE와 달리 노이즈를 증폭하지 않고 edge만 살짝 선명하게 만듦.
    """
    blurred = cv2.GaussianBlur(img.astype(np.float32),
                                (0, 0), sigma)
    sharpened = img.astype(np.float32) + amount * (img.astype(np.float32) - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# =====================================================================
#  A-3. Vignetting 보정 -- cylindrical warp 후 양 끝 밝기 저하 보정
# =====================================================================
def apply_vignette_correction(img, f_px):
    """
    Cylindrical warp 후 양 끝 밝기 저하를 코사인 law로 보정.
    cos(θ) 감소에 의한 밝기 감소를 역으로 보상.
    """
    h, w = img.shape[:2]
    cx = w / 2.0
    x = np.arange(w, dtype=np.float32)
    theta = (x - cx) / f_px
    cos_theta = np.cos(theta)
    # cos^2(θ) 감소 보정: gain = 1 / cos^2(θ), 클리핑 (최대 1.2배 -- 과잉 밝기 방지)
    correction = np.clip(1.0 / (cos_theta ** 2 + 1e-6), 1.0, 1.2)
    # 2D broadcast (행 방향으로 복제)
    correction_map = np.tile(correction, (h, 1))
    result = img.astype(np.float32)
    for c in range(3):
        result[:, :, c] *= correction_map
    return np.clip(result, 0, 255).astype(np.uint8)


# =====================================================================
#  A-4. 최종 후처리 파이프라인 -- bilateral + unsharp + saturation
# =====================================================================
def apply_final_postprocess(panorama):
    """
    최종 파노라마 후처리:
      1) Bilateral Filter: edge-preserving denoising (평탄 영역 노이즈 제거)
      2) Unsharp Masking: 선명도 미세 향상
      3) Saturation boost: 채도 미세 증가 (HSV S채널)
    CLAHE 이중 적용 대신 사용하여 노이즈 증폭 없이 시각적 품질 향상.
    """
    result = panorama.copy()

    # 1) Bilateral Filter -- BILATERAL_D=0 이면 skip (원본 화질 유지 모드)
    if BILATERAL_D > 0:
        result = cv2.bilateralFilter(result, BILATERAL_D,
                                     BILATERAL_SIGMA_COLOR,
                                     BILATERAL_SIGMA_SPACE)

    # 2) Unsharp Masking -- 선명도 미세 향상
    result = apply_unsharp_mask(result, UNSHARP_AMOUNT, UNSHARP_SIGMA)

    # 3) Saturation boost -- HSV S채널 미세 증가
    if abs(SATURATION_BOOST - 1.0) > 0.01:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= SATURATION_BOOST
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


# =====================================================================
#  B. Cylindrical Warping -- P03 backward warping 패턴 + cv2.remap
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
        3. cv2.remap으로 LANCZOS4 interpolation 수행
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

    # --- cv2.remap으로 LANCZOS4 interpolation (고품질 리샘플링) ---
    result = cv2.remap(img, map_x, map_y,
                       interpolation=INTERP_METHOD,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0))
    return result


# =====================================================================
#  C. 특징점 매칭 -- P02 Practice 5 코드 재사용
#     ratio test (0.75) + cross-check
# =====================================================================
def detect_sift(img):
    """SIFT 특징점 + descriptor 추출 (cv2.SIFT_create 사용)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.03)
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
        print("  [경고] descriptor가 None -- 매칭 불가")
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
#  D. 변환 추정 -- RANSAC (similarity → affine/homography 병렬 fallback)
# =====================================================================
def estimate_transform(kp1, kp2, matches):
    """
    인접쌍 변환행렬 추정 (img1 → img2 방향).
    Fallback 전략:
      1차: Similarity (4-DOF) -- cylindrical warp 후 이상적 경우
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

    # Similarity가 충분히 좋으면 inlier만으로 재추정 (least-squares refine)
    if M_sim is not None and ratio_sim >= FALLBACK_AFFINE_RATIO:
        # Inlier만으로 재추정 -- 정밀도 향상
        inlier_mask = inliers_sim.ravel().astype(bool)
        if np.sum(inlier_mask) >= 4:
            src_inliers = src_pts[inlier_mask]
            dst_inliers = dst_pts[inlier_mask]
            M_refined, _ = cv2.estimateAffinePartial2D(
                src_inliers, dst_inliers,
                method=cv2.LMEDS  # inlier만이므로 LMEDS로 정밀 추정
            )
            if M_refined is not None:
                M_sim = M_refined

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

    # Homography (8-DOF) -- overfit 방지 가드 포함
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
#  E. 변환 체이닝 -- 기준 이미지 기준 누적 + 캔버스 크기 산출
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
            print(f"  [경고] 이미지 {i+1} 체이닝 실패 -- Identity 대체")
            global_H[i] = np.eye(3, dtype=np.float64)

    # ref 오른쪽 (i > ref): inv(pairwise_H[i-1]) 누적
    for i in range(ref_idx + 1, num_images):
        # pairwise_H[i-1]: img_{i-1} → img_i
        # global_H[i] = inv(pairwise_H[i-1]) @ global_H[i-1]  → img_i를 ref 좌표계로
        if pairwise_H[i - 1] is not None and global_H[i - 1] is not None:
            H_inv = np.linalg.inv(pairwise_H[i - 1])
            global_H[i] = global_H[i - 1] @ H_inv
        else:
            print(f"  [경고] 이미지 {i+1} 체이닝 실패 -- Identity 대체")
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

    # 캔버스 크기 상한선 -- 초과 시 scale matrix도 함께 적용
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
#  F-0. Histogram Matching -- 인접 이미지 색조 통일
# =====================================================================
def histogram_match_lab(src, ref, mask_src, mask_ref):
    """
    LAB 채널별 히스토그램 매칭 (선형 스케일링 방식).
    Overlap 영역의 평균·표준편차를 기반으로 전체 이미지 색조를 ref에 맞춤.
    dst = (src - mean_src) * (std_ref / std_src) + mean_ref
    """
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)

    overlap = (mask_src > 0) & (mask_ref > 0)
    if overlap.sum() < 500:  # overlap이 너무 작으면 skip
        return src

    result_lab = src_lab.copy()
    for ch in range(3):  # L, A, B 각 채널
        src_ch = src_lab[:, :, ch]
        ref_ch = ref_lab[:, :, ch]

        # Overlap 영역에서 통계 추출
        src_mean = src_ch[overlap].mean()
        src_std = src_ch[overlap].std()
        ref_mean = ref_ch[overlap].mean()
        ref_std = ref_ch[overlap].std()

        if src_std < 1e-3:
            continue

        # 선형 변환
        ratio = ref_std / src_std if src_std > 1e-3 else 1.0
        ratio = np.clip(ratio, 0.5, 2.0)  # 극단적 변환 방지
        result_lab[:, :, ch] = (src_ch - src_mean) * ratio + ref_mean

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


# =====================================================================
#  F-1. Gain Compensation -- BGR 채널별 밝기·색온도 균일화
# =====================================================================
def compute_gain_compensation(warped_list, mask_list):
    """
    BGR 채널별 독립 gain 계산으로 밝기 + white balance 동시 보정.
    기준 이미지(가운데)를 [1.0, 1.0, 1.0]으로 놓고 좌/우 양방향 propagation.
    기존 LAB L채널 단일 gain → BGR 3채널 독립 gain으로 확장.
    """
    n = len(warped_list)
    # gains[i] = [gain_B, gain_G, gain_R]
    gains = [np.array([1.0, 1.0, 1.0]) for _ in range(n)]
    ref_idx = n // 2  # 기준 이미지 = 가운데

    # ref → 왼쪽 propagation
    for i in range(ref_idx - 1, -1, -1):
        overlap = (mask_list[i] > 0) & (mask_list[i + 1] > 0)
        if overlap.sum() < 100:
            gains[i] = gains[i + 1].copy()
            continue

        channel_gains = np.array([1.0, 1.0, 1.0])
        for ch in range(3):  # B, G, R
            mean_next = warped_list[i + 1][:, :, ch][overlap].astype(np.float64).mean()
            mean_curr = warped_list[i][:, :, ch][overlap].astype(np.float64).mean()
            g = mean_next / mean_curr if mean_curr > 1e-3 else 1.0
            channel_gains[ch] = g

        gains[i] = gains[i + 1] * channel_gains

    # ref → 오른쪽 propagation
    for i in range(ref_idx + 1, n):
        overlap = (mask_list[i - 1] > 0) & (mask_list[i] > 0)
        if overlap.sum() < 100:
            gains[i] = gains[i - 1].copy()
            continue

        channel_gains = np.array([1.0, 1.0, 1.0])
        for ch in range(3):  # B, G, R
            mean_prev = warped_list[i - 1][:, :, ch][overlap].astype(np.float64).mean()
            mean_curr = warped_list[i][:, :, ch][overlap].astype(np.float64).mean()
            g = mean_prev / mean_curr if mean_curr > 1e-3 else 1.0
            channel_gains[ch] = g

        gains[i] = gains[i - 1] * channel_gains

    # 클리핑 -- BGR 각 채널별 독립 [GAIN_CLIP_MIN, GAIN_CLIP_MAX]
    gains = [np.clip(g, GAIN_CLIP_MIN, GAIN_CLIP_MAX) for g in gains]
    return gains


# =====================================================================
#  F-2. Laplacian Pyramid 블렌딩 -- seam 제거용 다중 주파수 합성
# =====================================================================
def build_laplacian_pyramid(img, num_bands):
    """
    img: float32 (H, W, 3).
    반환: [고주파 residual_0, ..., 저주파 base] -- 총 num_bands개.
    pyrUp dstsize 명시로 홀수 차원 불일치 방지.
    """
    pyr, cur = [], img
    for _ in range(num_bands - 1):
        down = cv2.pyrDown(cur)
        up   = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
        cv2.subtract(cur, up, dst=cur)
        pyr.append(cur)
        cur = down
    pyr.append(cur)
    return pyr


def build_gaussian_pyramid(weight_map, num_bands):
    """weight_map: float32 (H, W) 단일 채널. 반환: Gaussian pyramid."""
    pyr, cur = [weight_map], weight_map
    for _ in range(num_bands - 1):
        cur = cv2.pyrDown(cur)
        pyr.append(cur)
    return pyr


def normalize_weight_maps_in_place(weight_maps):
    """Normalize per-image float32 weight maps without duplicating full canvases."""
    w_sum = np.zeros_like(weight_maps[0])
    for weight in weight_maps:
        w_sum += weight
    w_sum[w_sum == 0] = 1.0
    for weight in weight_maps:
        np.divide(weight, w_sum, out=weight)
    return weight_maps


def accumulate_weighted_pyramid(blended_pyr, lap_pyr, gau_pyr):
    """Weight Laplacian levels in place and accumulate without full-size temporaries."""
    for level in range(len(lap_pyr)):
        np.multiply(
            lap_pyr[level],
            gau_pyr[level][:, :, np.newaxis],
            out=lap_pyr[level],
        )

    if blended_pyr is None:
        return lap_pyr

    for level in range(len(blended_pyr)):
        np.add(blended_pyr[level], lap_pyr[level], out=blended_pyr[level])
    return blended_pyr


def is_canvas_within_memory_budget(canvas_size):
    """Reject transform candidates whose canvas would exceed the blend memory budget."""
    canvas_h, canvas_w = canvas_size
    return canvas_h * canvas_w <= MAX_CANVAS_PIXELS


def reconstruct_from_pyramid(pyr):
    """블렌딩된 Laplacian pyramid를 재구성하여 float32 (H, W, 3) 반환."""
    result = pyr[-1].copy()
    for lvl in range(len(pyr) - 2, -1, -1):
        th, tw = pyr[lvl].shape[:2]
        result = cv2.pyrUp(result, dstsize=(tw, th)) + pyr[lvl]
    return result


# =====================================================================
#  F. 합성 -- P03 backward warping 좌표 맵 + cv2.remap + feathering
# =====================================================================
def composite_panorama(images, global_H, canvas_size):
    """
    각 이미지를 캔버스에 합성.

    P03 backward warping 패턴:
        1. 출력(캔버스) 좌표 그리드 생성: np.indices
        2. 각 이미지의 global_H 역행렬로 입력 좌표 맵 계산:
           M_inv = np.linalg.inv(H)
           in_coords = M_inv @ out_coords   ← P03 코드 구조
        3. cv2.remap으로 LANCZOS4 interpolation 수행

    Blending: Histogram Matching + Gain Compensation + Laplacian Pyramid Multi-band Blending
        Phase 1: P03 backward warp로 모든 이미지 캔버스 좌표계로 수집
        Phase 2: Histogram matching으로 색조 통일
        Phase 3: 인접 overlap 기반 BGR 채널별 gain 보정 (노출+색온도 균일화)
        Phase 4: Laplacian pyramid 다중 주파수 블렌딩 (seam 제거)
    """
    canvas_h, canvas_w = canvas_size

    # --- P03 backward warping 패턴: 출력 좌표 그리드 (한 번만 생성) ---
    out_y_grid, out_x_grid = np.indices((canvas_h, canvas_w), dtype=np.float32)

    # ---- Phase 1: P03 backward warp -- 전체 warped 이미지 수집 ----
    all_warped = []
    all_masks  = []
    for i, (img, H) in enumerate(zip(images, global_H)):
        # --- P03 핵심: 역행렬로 입력 좌표 계산 ---
        # P03 원본: M_inv = np.linalg.inv(M)
        #           in_coords = M_inv @ out_coords
        H_inv = np.linalg.inv(H)

        # homogeneous 좌표 → 역변환 → 입력 좌표 맵
        # P03 원본: out_coords = np.stack((out_x.flatten(), out_y.flatten(), ones))
        #           in_coords = M_inv @ out_coords
        #           in_x = in_coords[0] / in_coords[2]
        #           in_y = in_coords[1] / in_coords[2]
        denom = H_inv[2, 0] * out_x_grid + H_inv[2, 1] * out_y_grid + H_inv[2, 2]
        denom[denom == 0] = 1e-10  # divide-by-zero 방지
        map_x = (H_inv[0, 0] * out_x_grid + H_inv[0, 1] * out_y_grid + H_inv[0, 2]) / denom
        map_y = (H_inv[1, 0] * out_x_grid + H_inv[1, 1] * out_y_grid + H_inv[1, 2]) / denom

        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # --- cv2.remap으로 LANCZOS4 interpolation (고품질 리샘플링) ---
        warped = cv2.remap(img, map_x, map_y,
                           interpolation=INTERP_METHOD,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0))

        # 유효 마스크: warped 이미지에서 검은색이 아닌 영역
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = (gray_warped > 1).astype(np.uint8)  # > 1: crop_black_borders와 threshold 통일

        # 마스크 erode -- remap 경계의 보간 artifact 제거
        if MASK_ERODE_PX > 0:
            erode_kernel = np.ones((MASK_ERODE_PX * 2 + 1, MASK_ERODE_PX * 2 + 1), np.uint8)
            mask = cv2.erode(mask, erode_kernel, iterations=1)

        all_warped.append(warped)
        all_masks.append(mask)
        print(f"    이미지 {i+1}/{len(images)} warp 완료")

    # Full-canvas coordinate maps are no longer needed after the warp phase.
    del out_y_grid, out_x_grid, denom, map_x, map_y, gray_warped

    # ---- Phase 2: Histogram Matching -- 기준 이미지 색조로 통일 ----
    ref_idx = len(all_warped) // 2
    for i in range(len(all_warped)):
        if i == ref_idx:
            continue
        all_warped[i] = histogram_match_lab(
            all_warped[i], all_warped[ref_idx],
            all_masks[i], all_masks[ref_idx]
        )
    print(f"    Histogram matching 완료 (기준: 이미지 {ref_idx+1})")

    # ---- Phase 3: Gain Compensation -- BGR 채널별 밝기·색온도 균일화 ----
    gains = compute_gain_compensation(all_warped, all_masks)
    for i in range(len(all_warped)):
        g = gains[i]
        if np.any(np.abs(g - 1.0) > 0.01):  # gain이 1.0에 가까우면 skip
            # BGR 각 채널에 독립 gain 적용
            img_f = all_warped[i].astype(np.float32)
            for ch in range(3):
                img_f[:, :, ch] *= g[ch]
            all_warped[i] = np.clip(img_f, 0, 255).astype(np.uint8)
    print(f"    Gain compensation 완료: {[f'[{g[0]:.3f},{g[1]:.3f},{g[2]:.3f}]' for g in gains]}")

    # ---- Phase 4: Multi-band Laplacian Pyramid Blending ----
    # Distance transform 기반 가중치 계산 + Gaussian smoothing
    weight_maps = []
    for m in all_masks:
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)
        # Gaussian blur로 weight 경계를 부드럽게 (seam 완화) -- 축소된 커널
        dist = cv2.GaussianBlur(dist, (BLEND_BLUR_KSIZE, BLEND_BLUR_KSIZE), 0)
        weight_maps.append(dist)

    # 픽셀별 가중치 합산 → 정규화 (각 픽셀에서 가중치 합 = 1)
    # Normalize in place so five additional full-canvas float arrays are not retained.
    norm_weights = normalize_weight_maps_in_place(weight_maps)
    del all_masks

    # 캔버스 크기가 pyrDown 레벨보다 충분히 큰지 확인
    min_dim = min(canvas_h, canvas_w)
    num_bands = min(BLEND_BANDS, max(1, int(np.log2(min_dim))))

    # Laplacian pyramid 블렌딩: 각 주파수 레벨별로 가중 합산
    blended_pyr = None
    for i in range(len(all_warped)):
        img_f   = all_warped[i].astype(np.float32)
        all_warped[i] = None
        lap_pyr = build_laplacian_pyramid(img_f, num_bands)
        gau_pyr = build_gaussian_pyramid(norm_weights[i], num_bands)
        norm_weights[i] = None
        blended_pyr = accumulate_weighted_pyramid(blended_pyr, lap_pyr, gau_pyr)
        del img_f, lap_pyr, gau_pyr

    del all_warped, norm_weights, weight_maps
    result = np.clip(reconstruct_from_pyramid(blended_pyr), 0, 255).astype(np.uint8)
    return result


# =====================================================================
#  G. 검은 여백 제거 + 최대 내접 직사각형 Crop
# =====================================================================
def find_largest_inscribed_rect(mask):
    """
    유효 마스크에서 최대 내접 직사각형 찾기 (DP 히스토그램 방식).
    mask: 2D uint8 (0 or 255 or 1).
    반환: (x, y, w, h) -- 최대 직사각형의 좌표와 크기.
    """
    binary = (mask > 0).astype(np.uint8)
    rows, cols = binary.shape

    # 각 행에서의 연속 히스토그램 높이 계산
    heights = np.zeros(cols, dtype=np.int32)
    best_area = 0
    best_rect = (0, 0, cols, rows)

    for r in range(rows):
        for c in range(cols):
            if binary[r, c] > 0:
                heights[c] += 1
            else:
                heights[c] = 0

        # 현재 히스토그램에서 최대 직사각형 (스택 기반)
        stack = []
        for c in range(cols + 1):
            h = heights[c] if c < cols else 0
            start = c
            while stack and stack[-1][1] > h:
                idx, sh = stack.pop()
                w = c - idx
                area = w * sh
                if area > best_area:
                    best_area = area
                    best_rect = (idx, r - sh + 1, w, sh)
                start = idx
            stack.append((start, h))

    return best_rect


def crop_black_borders(panorama):
    """
    검은 여백 제거:
      1) 그레이스케일 → threshold → non-zero 영역의 bounding rect
      2) Largest inscribed rectangle -- 곡선 경계에서도 최적 직사각형 추출
      3) 상하 추가 트림: 행별 유효 픽셀 비율 < CROP_ROW_VALID_RATIO 인 상·하단 행 제거
    """
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 1단계: non-zero column/row 범위 (기본 bounding rect)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        print("  [경고] 유효 픽셀 없음 -- crop 생략")
        return panorama

    x_br, y_br, w_br, h_br = cv2.boundingRect(coords)
    cropped = panorama[y_br:y_br+h_br, x_br:x_br+w_br]

    # 2단계: Largest inscribed rectangle
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    mask_cropped = (gray_cropped > 1).astype(np.uint8)

    lir_x, lir_y, lir_w, lir_h = find_largest_inscribed_rect(mask_cropped)
    lir_area = lir_w * lir_h
    br_area = w_br * h_br

    # LIR이 bounding rect의 70% 이상 면적이면 LIR 사용 (그렇지 않으면 기존 방식)
    if lir_area >= br_area * 0.70 and lir_w > 100 and lir_h > 100:
        cropped = cropped[lir_y:lir_y+lir_h, lir_x:lir_x+lir_w]
        print(f"  [G] LIR Crop: {panorama.shape[1]}x{panorama.shape[0]} → {lir_w}x{lir_h}")
    else:
        # 3단계: 상하 추가 트림 -- 유효 픽셀 비율이 낮은 행 제거
        row_valid = np.mean(gray_cropped > 0, axis=1)
        valid_rows = np.where(row_valid >= CROP_ROW_VALID_RATIO)[0]
        if len(valid_rows) > 0:
            top = valid_rows[0]
            bottom = valid_rows[-1] + 1
            cropped = cropped[top:bottom, :]
        print(f"  [G] Crop 완료: {panorama.shape[1]}x{panorama.shape[0]} → {cropped.shape[1]}x{cropped.shape[0]}")

    return cropped


# =====================================================================
#  H. 결과 평가 + best 선택 (geometric consistency 추가)
# =====================================================================
def compute_geometric_consistency(global_H, image_shapes):
    """
    변환행렬의 기하학적 일관성 점수 (0~1).
    Cylindrical warp 후 이상적으로는 순수 translation만 있어야 함.
    scale/rotation이 identity에 가까울수록 높은 점수.
    """
    n = len(global_H)
    if n < 2:
        return 1.0

    penalties = []
    for i in range(n):
        H = global_H[i]
        # 2x2 affine 부분에서 scale/rotation 추출
        a, b = H[0, 0], H[0, 1]
        c, d = H[1, 0], H[1, 1]

        # scale = sqrt(det), rotation = atan2(c, a)
        det = a * d - b * c
        scale = np.sqrt(abs(det))
        rotation = np.degrees(np.arctan2(c, a))

        # 이상적: scale=1.0, rotation=0.0
        scale_penalty = abs(scale - 1.0)
        rotation_penalty = abs(rotation) / 10.0  # 10도 = 패널티 1.0

        penalties.append(scale_penalty + rotation_penalty)

    avg_penalty = np.mean(penalties)
    # 0~1 범위로 변환 (penalty 0 → score 1, penalty 2+ → score 0)
    score = max(0.0, 1.0 - avg_penalty / 2.0)
    return score


def evaluate_result(panorama, inlier_counts, global_H=None, image_shapes=None):
    """
    평가 지표 (리밸런싱):
      1) RANSAC inlier 합계 (정규화, 25%)  -- 기존 50%에서 축소
      2) Crop 후 유효 픽셀 비율 + fill penalty (20%)
      3) 선명도: 유효 픽셀의 Laplacian 분산 (35%)  -- 기존 30%에서 강화
      4) Geometric consistency (20%)  -- 신규 추가
    """
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    fill_ratio = np.mean(gray > 0)
    total_inliers = sum(inlier_counts)
    inlier_norm = total_inliers / max(1, len(inlier_counts) * 100)

    # 선명도: 유효 픽셀 영역의 Laplacian 분산
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    valid_mask = gray > 0
    sharpness = float(lap[valid_mask].var()) if valid_mask.sum() > 0 else 0.0
    sharpness_norm = min(sharpness / 1000.0, 1.0)

    # Geometric consistency
    geometric = 1.0
    if global_H is not None and image_shapes is not None:
        geometric = compute_geometric_consistency(global_H, image_shapes)

    # fill 98% 미만이면 패널티 (black border 이 많은 결과 억제)
    fill_penalty = 1.0 if fill_ratio >= 0.98 else fill_ratio / 0.98

    # 리밸런싱된 점수: sharpness 강화, inlier 축소, geometric 추가
    score = (0.25 * inlier_norm
             + 0.20 * fill_ratio * fill_penalty
             + 0.35 * sharpness_norm
             + 0.20 * geometric)
    return score, total_inliers, fill_ratio, sharpness_norm, geometric


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
#  main -- 전체 파이프라인 오케스트레이션
# =====================================================================
def main():
    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  파노라마 스티칭 (Panorama Stitching) -- v4 품질 개선")
    print("=" * 60)

    # ---- A. 이미지 로드 ----
    images, file_paths = load_images()
    num_images = len(images)
    img_h, img_w = images[0].shape[:2]

    # ---- 고해상도 입력 자동 downscale (메모리/속도 최적화) ----
    if img_w > MAX_INPUT_WIDTH:
        scale_input = MAX_INPUT_WIDTH / img_w
        images = [cv2.resize(img, None, fx=scale_input, fy=scale_input,
                             interpolation=cv2.INTER_AREA) for img in images]
        img_h, img_w = images[0].shape[:2]
        print(f"  [A] 고해상도 자동 다운스케일 → {img_w}x{img_h}")
    else:
        print(f"  [A] 원본 해상도 유지: {img_w}x{img_h} (MAX_INPUT_WIDTH={MAX_INPUT_WIDTH})")

    # ---- EXIF focal length 자동 감지 ----
    exif_focal = detect_exif_focal_lengths(file_paths)
    if exif_focal is not None:
        # EXIF에서 focal length를 성공적으로 추출 → 단일 후보만 사용
        focal_candidates = [round(exif_focal)]
        print(f"  [EXIF] 단일 focal length 사용: {focal_candidates[0]}mm")
    else:
        # EXIF 없음 → fallback 후보
        focal_candidates = FOCAL_MM_CANDIDATES
        print(f"  [EXIF] EXIF 없음 → fallback 후보: {focal_candidates}")

    results = {}  # f_mm → (panorama, score, inliers, fill, sharpness, geometric)

    for f_idx, f_mm in enumerate(focal_candidates):
        print(f"\n{'─' * 60}")
        print(f"  [{f_idx+1}/{len(focal_candidates)}] Focal length: {f_mm}mm")
        print(f"{'─' * 60}")

        # ---- B. Cylindrical Warping ----
        f_px = img_w * f_mm / SENSOR_WIDTH_MM
        print(f"  [B] f_px = {f_px:.1f}  ({f_mm}mm → {img_w}px 센서)")

        # 원본 이미지를 warp → 합성에 사용 (원본 화질 유지)
        warped_images = []
        for img in images:
            w_img = cylindrical_warp(img, f_px)
            w_img = apply_vignette_correction(w_img, f_px)
            warped_images.append(w_img)

        # CLAHE를 warped 이미지에 적용 → SIFT 검출 전용 (합성엔 미사용)
        warped_clahe = [apply_clahe(w) for w in warped_images]

        print(f"  [B] Cylindrical warp + vignette 보정 완료 ({num_images}장)")
        debug_show(f"Cylindrical Warp (f={f_mm}mm) - Image 1", warped_images[0])

        # ---- C. Pairwise SIFT 매칭 (P02 재사용) ----
        print(f"  [C] SIFT 매칭 시작...")

        # SIFT 검출 -- CLAHE 적용 이미지로 특징점 품질 향상
        sift_results = []
        for i, w_img in enumerate(warped_clahe):
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
            print(f"  [경고] 일부 쌍에서 매칭/변환 실패 -- 결과 품질 저하 가능")

        # ---- E. 변환 체이닝 + 캔버스 계산 ----
        image_shapes = [img.shape for img in warped_images]
        global_H, canvas_size = chain_transforms(pairwise_H, num_images, image_shapes)

        if not is_canvas_within_memory_budget(canvas_size):
            canvas_h, canvas_w = canvas_size
            print(
                f"  [F] 합성 건너뜀: {canvas_w}x{canvas_h} "
                f"({canvas_w * canvas_h:,} px) 캔버스가 메모리 예산 "
                f"{MAX_CANVAS_PIXELS:,} px를 초과"
            )
            continue

        # ---- F. 합성 (backward warp + histogram matching + gain compensation + multi-band blending) ----
        print(f"  [F] 합성 시작...")
        panorama = composite_panorama(warped_images, global_H, canvas_size)
        debug_show(f"Composite (f={f_mm}mm)", panorama)

        # ---- G. Crop ----
        panorama = crop_black_borders(panorama)
        debug_show(f"Cropped (f={f_mm}mm)", panorama)

        # ---- G-1. 최종 후처리: Bilateral + Unsharp + Saturation boost ----
        panorama = apply_final_postprocess(panorama)

        # ---- 평가 (geometric consistency 포함) ----
        score, total_inliers, fill_ratio, sharpness_norm, geometric = evaluate_result(
            panorama, inlier_counts, global_H, image_shapes
        )

        # ---- 저장 ----
        out_path = os.path.join(base_dir, f"result_f{f_mm}.jpg")
        cv2.imwrite(out_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 97])
        print(f"  [H] 저장: {os.path.basename(out_path)}  ({panorama.shape[1]}x{panorama.shape[0]})")
        print(f"       Score={score:.4f}  Inliers={total_inliers}  Fill={fill_ratio:.2%}  Sharp={sharpness_norm:.3f}  Geo={geometric:.3f}")

        results[f_mm] = (panorama, score, total_inliers, fill_ratio, sharpness_norm, geometric)

    # ---- Best 선택 + result_panorama.jpg 저장 ----
    print(f"\n{'=' * 60}")
    print(f"  결과 요약")
    print(f"{'=' * 60}")
    print(f"  {'f_mm':>6}  {'Score':>8}  {'Inliers':>8}  {'Fill':>8}  {'Sharp':>7}  {'Geo':>5}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*5}")

    best_mm = None
    best_score = -1
    for f_mm, (pano, score, inliers, fill, sharp, geo) in results.items():
        if score > best_score:
            best_score = score
            best_mm = f_mm
        print(f"  {f_mm:>4}mm  {score:>8.4f}  {inliers:>8}  {fill:>7.2%}  {sharp:>7.3f}  {geo:>5.3f}")

    if best_mm is not None:
        best_pano = results[best_mm][0]
        best_path = os.path.join(base_dir, "result_panorama.jpg")
        cv2.imwrite(best_path, best_pano, [cv2.IMWRITE_JPEG_QUALITY, 98])
        print(f"\n  ★ Best: f={best_mm}mm (score={best_score:.4f})")
        print(f"  ★ 저장: result_panorama.jpg ({best_pano.shape[1]}x{best_pano.shape[0]})")
        print(f"  ★ JPEG Quality: 98")

    # ===== 보고서 기입용 요약 (표1 + 분석1) =====
    print(f"\n{'='*60}")
    print("  [보고서 기입용] 표 1 - 초점거리별 결과")
    print(f"{'='*60}")
    print(f"  {'f(mm)':>6} {'f_px':>9} {'Score':>9} {'Inliers':>9} {'Fill(%)':>9} {'Sharp':>7} {'Geo':>5}")
    for f_mm in focal_candidates:
        if f_mm not in results:
            continue
        f_px = img_w * f_mm / SENSOR_WIDTH_MM
        _, score, inliers, fill, sharp, geo = results[f_mm]
        print(f"  {f_mm:>6} {f_px:>9.1f} {score:>9.4f} {inliers:>9} {fill*100:>8.2f} {sharp:>7.3f} {geo:>5.3f}")
    if best_mm is not None:
        print(f"\n  분석(1) Best 초점거리 = {best_mm}mm,  Inlier 합계 = {results[best_mm][2]}")
    print(f"{'='*60}")


    # ---- results.json 저장 (update_docx.py에서 읽어 사용) ----
    results_data = {}
    for f_mm in focal_candidates:
        if f_mm not in results:
            continue
        f_px = img_w * f_mm / SENSOR_WIDTH_MM
        _, score, inliers, fill, sharp, geo = results[f_mm]
        results_data[str(f_mm)] = {
            "f_px": round(f_px, 1),
            "score": round(score, 4),
            "inliers": inliers,
            "fill": round(fill * 100, 2),
            "sharp": round(sharp, 3),
            "geo": round(geo, 3),
            "best": f_mm == best_mm
        }
    results_data["best_mm"] = best_mm
    json_path = os.path.join(base_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results_data, jf, indent=2, ensure_ascii=False)
    print(f"  [결과] results.json 저장 완료")

    elapsed = time.time() - start_time
    print(f"\n  총 실행 시간: {elapsed:.1f}초")
    print("=" * 60)


if __name__ == "__main__":
    main()
