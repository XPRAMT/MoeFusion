import cv2
import numpy as np
import os
import utils

class ImageStitcher:
    def __init__(self, debug=False):
        self.debug = debug
        self.detector = cv2.SIFT_create()

    def _log(self, msg):
        if self.debug:
            utils.toMainGUI.put([2,f"[stitch] {msg}"])

    def _read_and_gray(self, path):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._log(f"[讀取圖片] {path} | 大小: {img.shape}")
        return gray, img
    
    def show_image_window(self,title, image, max_scale=0.9):
        """顯示圖片，確保視窗大小不會超過螢幕"""
        # 查詢螢幕解析度
        screen_res = (1920, 1080)  # 預設值，若用 tkinter 可抓真實解析度
        try:
            import tkinter as tk
            root = tk.Tk()
            screen_res = (root.winfo_screenwidth(), root.winfo_screenheight())
            root.destroy()
        except:
            pass

        screen_w, screen_h = screen_res
        max_w, max_h = int(screen_w * max_scale), int(screen_h * max_scale)

        h, w = image.shape[:2]

        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        else:
            resized = image

        cv2.imshow(title, resized)

    def _match_features(self, img1, img2):
        '計算重疊特徵點'
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)
        self._log(f"[特徵點] 當前拼接圖: {len(kp1)} 個點, 待比較圖: {len(kp2)} 個點")

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        self._log(f"[比對結果] 好的配對數量: {len(good)}")
        return kp1, kp2, good

    def _find_homography(self, kp1, kp2, matches):
        if len(matches) < 4:
            self._log("[單應矩陣] 配對數量不足，無法計算")
            return None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self._log("[單應矩陣] 已成功估算")
        return H

    def _warp_and_merge(self, img1, img2, H, n):
        '''
        img1 是基底，img2 是被變形的圖。輸出為 RGBA，填充透明像素。
        '''
        # 轉為 4 通道
        if img1.shape[2] == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
        if img2.shape[2] == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # 計算 canvas 尺寸
        corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners_img2, H)
        all_corners = np.concatenate((transformed_corners,
                                    np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)), axis=0)

        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        translation = [-xmin, -ymin]

        T = np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]],
            [0, 0, 1]
        ])

        canvas_size = (xmax - xmin, ymax - ymin)

        # warp img2 到 canvas
        warped_img2 = cv2.warpPerspective(img2, T @ H, canvas_size, flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        warped_mask2 = (warped_img2[..., 3] > 0).astype(np.uint8)

        # 將 img1 貼到 canvas
        canvas_img1 = np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)
        canvas_img1[translation[1]:translation[1]+h1, translation[0]:translation[0]+w1] = img1
        mask_img1 = (canvas_img1[..., 3] > 0).astype(np.uint8)

        # 區域分類
        overlap = (warped_mask2 > 0) & (mask_img1 > 0)
        only_img1 = (mask_img1 > 0) & (~overlap)
        only_img2 = (warped_mask2 > 0) & (~overlap)

        result = np.zeros_like(canvas_img1)

        result[only_img1] = canvas_img1[only_img1]
        result[only_img2] = warped_img2[only_img2]

        if np.any(overlap):
            self._log("[融合] 對重疊區域進行 feather blending")

            # 計算 feather blending 的 alpha
            dist1 = cv2.distanceTransform(mask_img1.astype(np.uint8) * 255, cv2.DIST_L2, 5)
            dist2 = cv2.distanceTransform(warped_mask2.astype(np.uint8) * 255, cv2.DIST_L2, 5)

            total = dist1 + dist2 + 1e-8
            alpha1 = dist1 / total
            alpha2 = dist2 / total

            for c in range(4):
                result[..., c][overlap] = (
                    canvas_img1[..., c][overlap] * alpha1[overlap] +
                    warped_img2[..., c][overlap] * alpha2[overlap]
                ).astype(np.uint8)

        self._log(f"[影像融合] 合成結果大小: {result.shape}")

        if self.debug:
            os.makedirs("debug", exist_ok=True)
            rgb_result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(f"debug/result_{n}.jpg", rgb_result)
        return result


    def _auto_crop(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self._log("[裁切] 找不到有效區域")
            return img

        x, y, w, h = cv2.boundingRect(contours[0])
        cropped = img[y:y+h, x:x+w]
        self._log(f"[裁切] 區域: x={x}, y={y}, w={w}, h={h}")
        return cropped

    def stitch(self, image_paths):
        images = []
        gray_images = []

        self._log(f"[開始] 共讀取 {len(image_paths)} 張圖片")
        for path in image_paths:
            gray, color = self._read_and_gray(path)
            gray_images.append(gray)
            images.append(color)

        count = 1
        while len(images) > 1:
            self._log(f"\n[步驟 {count}] 搜尋最佳拼接組合")
            best_i, best_j = -1, -1
            max_matches = 100
            best_H = None

            # 比較每一對圖片
            for i in range(len(images)):
                for j in range(i+1, len(images)):
                    kp1, kp2, good = self._match_features(gray_images[i], gray_images[j])
                    H = self._find_homography(kp1, kp2, good)

                    if H is not None and len(good) > max_matches:
                        max_matches = len(good)
                        best_i, best_j = i, j
                        best_H = H

            if best_H is None:
                self._log("[中止] 沒有找到合適的圖片組合")
                break

            self._log(f"[拼接] 選擇圖片 {best_i} 與圖片 {best_j}（配對點數: {max_matches}）")
            img1 = images[best_i]
            img2 = images[best_j]
            

            stitched = self._warp_and_merge(img2, img1, best_H,count)
            stitched = self._auto_crop(stitched)

            # 將新圖加入列表，移除原圖
            del_idx = sorted([best_i, best_j], reverse=True)
            for idx in del_idx:
                del images[idx]
                del gray_images[idx]

            images.append(stitched)
            gray_images.append(cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY))
            count += 1

        self._log("[完成] 所有圖片拼接完成")
        
        return cv2.cvtColor(images[0], cv2.COLOR_BGRA2BGR)