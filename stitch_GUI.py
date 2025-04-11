import os,time,io
import cv2
import numpy as np
import utils
from waifu2x_vulkan import waifu2x
from PyQt6 import QtWidgets, QtCore
from waifu2x_Composite import waifu2xComposite

class StitchWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.thread = None
        self.Stitcher = None
        self.waifu2x_instance = None

        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        params_group = QtWidgets.QGroupBox("Stitch Parameters")
        params_layout = QtWidgets.QVBoxLayout(params_group)

        settinglayout = QtWidgets.QVBoxLayout()
        settinglayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        # 相似度
        self.similarLabel = QtWidgets.QLabel("")
        self.similarSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.similarSlider.setRange(1, 100)
        self.similarSlider.setValue(1)
        self.similarSlider.valueChanged.connect(self._updateSettings)
        settinglayout.addWidget(self.similarLabel)
        settinglayout.addWidget(self.similarSlider)
        # 重疊大小
        self.OverlapSizeLabel = QtWidgets.QLabel("")
        self.OverlapSizeSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.OverlapSizeSlider.setRange(1, 100)
        self.OverlapSizeSlider.setValue(1)
        self.OverlapSizeSlider.valueChanged.connect(self._updateSettings)
        settinglayout.addWidget(self.OverlapSizeLabel)
        settinglayout.addWidget(self.OverlapSizeSlider)
        # 使用 waifu2x
        self.UseWaifu2xLabel = QtWidgets.QLabel("")
        self.UseWaifu2xSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.UseWaifu2xSlider.setRange(10, 100)
        self.UseWaifu2xSlider.setValue(10)
        self.UseWaifu2xSlider.valueChanged.connect(self._updateSettings)
        settinglayout.addWidget(self.UseWaifu2xLabel)
        settinglayout.addWidget(self.UseWaifu2xSlider)
        # # #
        params_layout.addLayout(settinglayout)
        layout.addWidget(params_group)
        # 儲存、讀取
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save Parameter")
        self.btn_save.clicked.connect(self.save_parameters)
        btn_layout.addWidget(self.btn_save)
        self.btn_load = QtWidgets.QPushButton("Load Parameter")
        self.btn_load.clicked.connect(self.load_parameters)
        btn_layout.addWidget(self.btn_load)
        layout.addLayout(btn_layout)
        self._updateSettings()
        self.load_parameters()

    def _updateSettings(self):
        self.similarLabel.setText(f'Feature similar: {self.similarSlider.value()/100:.0%}')
        self.OverlapSizeLabel.setText(f'Overlap size: {self.OverlapSizeSlider.value()/100:.0%}')
        waifu2xValue = self.UseWaifu2xSlider.value()/10
        if waifu2xValue > 1:
            self.UseWaifu2xLabel.setText(f'Use waifu2x: scale factor > {waifu2xValue:.1f}')
        else:
            self.UseWaifu2xLabel.setText(f'Use waifu2x: off')

    def run_processing(self):
        # 收集待處理圖片，並更新狀態
        fromFolder = False
        self.image_queue = []
        for file_path, status_code in utils.image_queue.items():
            if status_code == 0:
                self.image_queue.append(file_path)
                utils.image_queue[file_path] = 1
            if file_path in utils.fromeFolder:
                fromFolder = True

        utils.toMainGUI.put([2, '更新table'])

        if not self.image_queue:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select images!")
            return

        utils.isRunning = True
        utils.toMainGUI.put([1, utils.isRunning])

        # 建立 ImageStitcher 並移至新的線程中
        self.Stitcher = ImageStitcher()
        self.Stitcher.image_queue = self.image_queue
        self.Stitcher.fromFolder = fromFolder
        self.Stitcher.similarity = self.similarSlider.value()/100
        self.Stitcher.OverlapSize = self.OverlapSizeSlider.value()/100
        self.Stitcher.waifu2x = self.waifu2x_instance
        self.Stitcher.useWaifu2x = self.UseWaifu2xSlider.value()/10
        
        self.thread = QtCore.QThread()
        self.Stitcher.moveToThread(self.thread)
        # 線程啟動後，透過 lambda 呼叫 Stitcher.run()
        self.thread.started.connect(self.Stitcher.run)
        # Stitcher 完成後發射 finished 信號，由 handle_result 處理結果
        self.Stitcher.finished.connect(self.handle_result)
        # 線程結束後清除資源
        self.Stitcher.finished.connect(self.thread.quit)
        self.Stitcher.finished.connect(self.Stitcher.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @QtCore.pyqtSlot(object)
    def handle_result(self, result):
        for path in self.image_queue:
            if path in result:
                utils.image_queue[path] = 2 #完成
            else:
                utils.image_queue[path] = 3 #錯誤 
                
        utils.isRunning = False
        utils.toMainGUI.put([0, ""])
        utils.toMainGUI.put([2, '更新table'])
        utils.toMainGUI.put([1, utils.isRunning])

    def stop_processing(self):
        if self.Stitcher:
            self.Stitcher._stop = True  # 通知拼接運算中斷
        utils.toMainGUI.put([0, "Stop requested."])

    def save_parameters(self):
        params = {
            "similarity": self.similarSlider.value()/100,
            "OverlapSize": self.OverlapSizeSlider.value()/100,
            "UseWaifu2x" : self.UseWaifu2xSlider.value()/10
        }
        config_data = utils.config_file()
        config_data["stitch"] = params
        utils.config_file(config_data)
        utils.toMainGUI.put([0,"[parameters] Stitch saved."])

    def load_parameters(self):
        config_data = utils.config_file()
        params = config_data.get("stitch", {})
        if params:
            self.similarSlider.setValue(int(params.get("similarity", 0.5)*100))
            self.OverlapSizeSlider.setValue(int(params.get("OverlapSize", 0.5)*100))
            self.UseWaifu2xSlider.setValue(int(params.get("UseWaifu2x", 1)*10))
            utils.toMainGUI.put([0,"[parameters] Stitch loaded."])
        else:
            utils.toMainGUI.put([0,"[parameters] Not found stitch."])

###################################################################

class ImageStitcher(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)  # 處理完成後傳遞結果

    def __init__(self):
        super().__init__()
        # 外部傳入
        self._stop = False
        self.image_queue = None
        self.fromFolder = False
        self.similarity = 0
        self.OverlapSize = 0
        self.waifu2x = None
        self.useWaifu2x = 1
        # 內部
        self.detector = cv2.SIFT_create(nfeatures=2000)
        self.best = None
        self.stitchedImg = []

    def _log(self, msg):
        if utils.debug:
            utils.toMainGUI.put([0, f"[stitch]{msg}"])

    def run(self):
        result = self.stitch()
        self.finished.emit(result)
    
    def _read(self, path):
        try:
            # 以二進位方式讀取檔案，適用於中文路徑
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is None:
                self._log(f"[讀取失敗] {os.path.basename(path)} 無法解碼")
                return None
            imgName = os.path.basename(path)
            self._log(f"[讀取圖片] 大小: {img.shape} | {imgName}")
            # 轉換為 BGRA 
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            return img
        except Exception as e:
            self._log(f"[讀取圖片例外] {imgName} | {e}")
            return None
        
    def _gray(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)
        
    def _find_good(self, imgB, imgT):
        desT = imgT['des']
        desB = imgB['des']
        kpT = imgT['kp']
        
        # 依據 kpT 所有特徵點的位置，計算最小與最大值，以便劃分網格
        pts = np.array([kp.pt for kp in kpT])
        min_x = np.min(pts[:, 0])
        max_x = np.max(pts[:, 0])
        min_y = np.min(pts[:, 1])
        max_y = np.max(pts[:, 1])
        
        cell_size = int(np.sqrt((max_x - min_x)*(max_y - min_y)/200))

        # 當描述子不存在時直接回傳 None
        if desT is None or desB is None:
            return None
        
        # 1. 使用 BFMatcher 的 knnMatch 進行初步匹配（k=2）
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desT, desB, k=2)
        
        # 2. 根據比值測試 (Lowe's ratio test 變體) 篩選初步匹配
        candidate_matches = []
        for m, n in matches:
            # 當 m 與 n 的距離差距夠大，表示 m 應該是更可靠的匹配
            if m.distance < (1 - self.similarity) * n.distance:
                candidate_matches.append(m)
        
        # 3. 根據 kpT 中匹配點的空間位置進行網格篩選，保證匹配點分佈均勻
        grid_dict = {}  # 用來記錄每個網格中當前最佳（距離最小）的匹配
        for m in candidate_matches:
            # 取得頂圖對應關鍵點的位置 (x, y)
            x, y = kpT[m.queryIdx].pt
            # 將位置轉換為相對於最小值的坐標，再計算出所屬的網格索引（行、列）
            grid_x = int((x - min_x) // cell_size)
            grid_y = int((y - min_y) // cell_size)
            grid_key = (grid_x, grid_y)
            
            # 如果該格子尚無匹配點，或者當前匹配距離更短，則保留當前匹配
            if grid_key not in grid_dict or m.distance < grid_dict[grid_key].distance:
                grid_dict[grid_key] = m
        
        good = list(grid_dict.values())
        self._log(f"[匹配點] 區塊大小:{cell_size}² ,特徵點: {len(candidate_matches)} ➜ {len(good)}")
        
        return good

    def _find_homography(self, kpB, kpT, matches):
        '找變換矩陣 base, top'
        if len(matches) < 4:
            self._log("[變換矩陣] 匹配點不足，無法計算")
            return None,None
        
        src_pts = np.float32([kpT[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kpB[m.trainIdx].pt for m in matches])

        # 使用 cv2.estimateAffinePartial2D 計算仿射矩陣，僅估算旋轉、縮放和平移
        #affine_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        affine_matrix, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)

        #H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) #透視
        
        if affine_matrix is None:
            self._log("[變換矩陣] 無法計算仿射轉換矩陣")
            return None,None

        # 根據 H 計算縮放因子
        scale_factor = np.sqrt(np.abs(np.linalg.det(affine_matrix[:, :2])))
        if scale_factor>0:
            self._log(f"[變換矩陣] 縮放比例: {scale_factor:.3f}")
        else:
            self._log(f"[變換矩陣] 縮放比例: {scale_factor:.3f} 錯誤")
        return affine_matrix,scale_factor

    def _resize(self,img,scale_factor):
        'img_dict, scale_factor'
        if 10 > self.useWaifu2x > 1 and scale_factor > self.useWaifu2x:
            self._log(f"[放大] {img['name']} 透過waifu2放大 {scale_factor:.2f} 倍")
            Waifu2x = waifu2xComposite()
            Waifu2x.waifu2x = self.waifu2x
            Waifu2x.inputImg = img['raw']
            Waifu2x.scale = scale_factor
            Waifu2x.isReturnImg = True
            img = Waifu2x.run()
        elif 20 > scale_factor > 0:
            self._log(f"[放大] {img['name']} 放大 {scale_factor:.2f} 倍")
            img = cv2.resize(img['raw'], None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        else:
            self._log(f"[放大] 失敗,縮放率:{scale_factor:.2f}")
            return None

        return img

    def _warp_and_mergeX(self, img1, img2, H):
        # 如果圖像只有三個通道（BGR），則轉換為四通道（BGRA），以便後續處理 alpha 通道
        if img1.shape[2] == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
        if img2.shape[2] == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
        # 取得 img1 與 img2 的高度和寬度
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        # 定義 img2 的四個角點，形成一個 (4,1,2) 的陣列
        corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        # 使用單應矩陣 H 將 img2 的角點做透視變換
        transformed_corners = cv2.perspectiveTransform(corners_img2, H)
        # 將變換後的角點與 img1 的角點結合，以便計算整體畫布的尺寸
        all_corners = np.concatenate((
            transformed_corners,
            np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        ), axis=0)
        
        # 找出所有角點的最小與最大座標，並微調小數點誤差後轉換為整數
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        # 計算平移量，確保結果圖像中所有像素座標都是正數
        translation = [-xmin, -ymin]
        # 建立 3x3 的平移矩陣 T
        T = np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]],
            [0, 0, 1]
        ])
        # 計算輸出畫布的大小 (寬, 高)
        canvas_size = (xmax - xmin, ymax - ymin)
        # 將 img2 透視變換 (結合 T 和 H)，映射到新的畫布上
        # BORDER_CONSTANT 並用透明值作為邊界填充
        warped_img = cv2.warpPerspective(img2, T @ H, canvas_size, flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        # 為 warped_img2 建立 alpha 掩膜，判斷哪些像素是有效的 (alpha 大於 0)
        warped_mask = (warped_img[..., 3] > 0).astype(np.uint8)
        # 建立與輸出畫布大小相同的空白圖像，並作為 img1 的畫布
        canvas_img1 = np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)
        # 將 img1 放入這個畫布中，位置依據 translation 調整
        canvas_img1[translation[1]:translation[1]+h1, translation[0]:translation[0]+w1] = img1
        # 建立 img1 的 alpha 掩膜，判斷哪些區域為有效 (alpha 大於 0)
        img1_mask = (canvas_img1[..., 3] > 0).astype(np.uint8)
        # 計算兩個掩膜的重疊區域，即同時屬於 img1 和 warped_img 的區域
        overlap_mask = (warped_mask > 0) & (img1_mask > 0)
        # 計算只有 warped_img 有效的區域（但不在重疊區域中）
        only_warped_mask = (warped_mask > 0) & (~overlap_mask)
        # 計算只有 img1 有效的區域（但不在重疊區域中）
        only_img1_mask = (img1_mask > 0) & (~overlap_mask)
        # 創建與畫布大小相同的空白結果圖像
        result = canvas_img1.copy()
        # 將只有 warped_img2 的區域直接複製到結果圖中
        result[only_warped_mask] = warped_img[only_warped_mask]
        # 如果存在重疊區域，則對該區域進行 feather blending（羽化融合）
        if np.any(overlap_mask):
            self._log("[拼接] 對重疊區域進行羽化融合")
            # 計算 img1 掩膜的距離轉換：每個像素到最近背景像素（0）之距離
            dist1 = cv2.distanceTransform(img1_mask.astype(np.uint8) * 255, cv2.DIST_L2, 5)
            dist2 = cv2.distanceTransform(warped_mask.astype(np.uint8) * 255, cv2.DIST_L2, 5)
            if np.any(only_img1_mask ):
                self._log("[拼接] 不完全重疊")
                total = dist1 + dist2 + 1e-8
                alpha1 = dist1 / total 
                alpha2 = 1-alpha1
            else:
                self._log("[拼接] 完全重疊")
                max_dist1 = np.max(dist1)
                alpha1 = (dist1 * 2) / max_dist1
                alpha1 = np.clip(alpha1, 0, 1)
                alpha2 = 1-alpha1
            # 對重疊區域中的每一個通道 (BGRA 共 4 通道) 進行加權混合
            for c in range(4):
                result[..., c][overlap_mask] = (
                    canvas_img1[..., c][overlap_mask] * alpha1[overlap_mask] +
                    warped_img[..., c][overlap_mask] * alpha2[overlap_mask]
                ).astype(np.uint8)
        
        # 記錄最終合成後結果圖片的尺寸
        self._log(f"[拼接] 合成結果大小: {result.shape}")
        # 返回最終拼接結果
        return result

    def _warp_and_merge(self, imgB, imgT, affine_matrix, scale_T):
        
        if scale_T < 1:
            self._log(f'[拼接] 放大底圖 {1 / scale_T:.2f} 倍')
            imgB = self._resize(imgB, 1 / scale_T)
            imgT = imgT['raw']
            affine_matrix = affine_matrix.copy() / scale_T # 調整仿射矩陣
        else:
            self._log(f'[拼接] 放大頂圖 {scale_T:.2f} 倍')
            imgT = self._resize(imgT, scale_T)
            imgB = imgB['raw']
            affine_matrix[:, :2] = affine_matrix.copy()[:, :2] / scale_T
        
        if imgB is None or imgT is None:
            return None
        # 計算兩張圖像的尺寸與角點位置
        hB, wB = imgB.shape[:2]
        hT, wT = imgT.shape[:2]
        # 取得 top 圖像 (imgT) 的四個角點，並以浮點數型態儲存
        corners_imgT = np.float32([[0, 0], [wT, 0], [wT, hT], [0, hT]]).reshape(-1, 1, 2)
        # 以 affine_matrix 對 top 圖像角點進行幾何變換 (旋轉、平移)
        transformed_corners = cv2.transform(corners_imgT, affine_matrix)
        # 取得 base 圖像 (imgB) 的四個角點，預設位於原點位置
        corners_imgB = np.float32([[0, 0], [wB, 0], [wB, hB], [0, hB]]).reshape(-1, 1, 2)
        # 合併兩組角點以計算整體輸出畫布的範圍
        all_corners = np.concatenate((transformed_corners, corners_imgB), axis=0)
        # 取得 x、y 最小與最大值，並略微擴充 (0.5 像素)
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        self._log(f"[拼接] 整體畫布範圍：xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
        # 建立平移矩陣並調整 affine_matrix 使所有像素位於正坐標系統中
        # 計算平移量：使最小座標為 (0, 0)
        translation = [-xmin, -ymin]
        # 建立 2x3 平移矩陣
        T = np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]]
        ], dtype=np.float32)
        # 將 affine_matrix 轉換為 3x3 同時合併平移矩陣：
        # 先把 2x3 affine_matrix 轉換為 3x3 矩陣 (補上 [0, 0, 1])
        M_hom = np.vstack([affine_matrix, [0, 0, 1]])
        # 同理建立 3x3 的平移矩陣
        T_hom = np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        # 合併後得到新的變換矩陣
        new_M = T_hom @ M_hom  # 3x3 矩陣
        # 取回 2x3 的 affine_matrix
        affine_matrix_new = new_M[0:2, :]
        # 計算最終畫布尺寸
        canvas_size = (xmax - xmin, ymax - ymin)
        # 使用 cv2.warpAffine() 進行頂圖的幾何變換 (旋轉、平移)
        warped_imgT = cv2.warpAffine(imgT, affine_matrix_new, canvas_size, flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        # 為 warped 建立 alpha 掩膜，定位圖像中有效（不透明）區域
        warped_mask = (warped_imgT[..., 3] > 0).astype(np.uint8)
        # 建立與畫布同大小的空白圖，將 base 圖像置於正確位置 (依據平移量)
        canvas_imgB = np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)
        canvas_imgB[translation[1]:translation[1] + hB, translation[0]:translation[0] + wB] = imgB
        # 為 base 圖像建立 alpha 掩膜
        base_mask = (canvas_imgB[..., 3] > 0).astype(np.uint8)
        # 區分非重疊與重疊區域
        overlap_mask = (warped_mask > 0) & (base_mask > 0)
        only_warped_mask = (warped_mask > 0) & (~overlap_mask)
        # 將 base 圖像內容複製至結果畫布
        result = canvas_imgB.copy()
        result[only_warped_mask] = warped_imgT[only_warped_mask]

        if np.any(overlap_mask):
            # 對 base 掩膜與 warped top 掩膜分別計算距離轉換，得到各像素離邊界的距離
            dist_base = cv2.distanceTransform(base_mask * 255, cv2.DIST_L2, 5)
            dist_warp = cv2.distanceTransform(warped_mask * 255, cv2.DIST_L2, 5)
            # 判斷是否為不完全重疊 (存在僅在 base 上有效的區域)
            if np.any(only_warped_mask):
                self._log(f"[羽化] 不完全重疊,羽化區域:{self.OverlapSize:.0%}")
                sub = dist_base + dist_warp + 1e-8  
            else:
                self._log(f"[羽化] 完全重疊,羽化區域:{self.OverlapSize:.0%}")
                sub = np.max(dist_warp)
            alpha_warp = np.clip((dist_warp/self.OverlapSize)/sub, 0, 1) 
            alpha_base = 1 - alpha_warp
            for c in range(4):
                result[..., c][overlap_mask] = (
                    canvas_imgB[..., c][overlap_mask] * alpha_base[overlap_mask] +
                    warped_imgT[..., c][overlap_mask] * alpha_warp[overlap_mask]
                ).astype(np.uint8)

        self._log(f"[拼接] 處理完成,尺寸:{result.shape}")
        return result

    def _auto_crop(self, img):
        '自動裁切並顯示輪廓與裁切框'
        # 轉換為灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 二值化 (所有像素 > 1 的都變成 255)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        # 找到外部輪廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self._log("[裁切] 找不到有效區域")
            return img

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # 裁切圖片
        cropped = img[y:y+h, x:x+w]
        orig_h, orig_w = img.shape[:2]
        self._log(f"[裁切] {orig_w}x{orig_h} -> {w}x{h}")
        return cropped

    def stitch(self):
        self._stop = False  # 開始前重置停止旗標

        imgs = []
        self._log(f"[開始] 共讀取 {len(self.image_queue)} 張圖片")

        for path in self.image_queue:
            if self._stop:
                self._log("stitching stopped during image loading.")
                return False
            img = self._read(path)

            if img is not None:
                imgs.append({
                    'raw':img,
                    'gray':self._gray(img),
                    'name':os.path.splitext(os.path.basename(path))[0],
                    'kp':None,
                    'des':None,
                    'path':path
                })

        self._log("-"*30)
        count = 1
        while len(imgs) > 1:
            self._log(f"[搜尋最佳拼接組合] {count}")
            if self._stop:
                self._log("中止")
                break
            best_i, best_j = -1, -1
            best_H = None

            for img in imgs:
                kp, des = self.detector.detectAndCompute(img['gray'], None)
                img['kp'] = kp
                img['des'] = des
                self._log(f"[尋找特徵] {len(kp)} 點 | {img['name']}")

            maxMatcheNum = 0
            best_matches = []
            for i in range(len(imgs)):
                for j in range(i+1, len(imgs)):
                    good = self._find_good(imgs[j],imgs[i])
                    if good is None:
                        self._log("[特徵匹配] 無法取得描述子，跳過匹配")
                    elif len(good) >= 4:
                        symbol = '>='
                        if len(good) > maxMatcheNum:
                            best_matches = good
                            maxMatcheNum = len(good)
                            best_i, best_j = i, j
                    else:
                        symbol = '<'
                    self._log(f"[特徵匹配] {imgs[j]['name']} | {imgs[i]['name']} | matchs: {len(good)} {symbol} 4")

            if best_matches is None:
                self._log(f"[特徵匹配] 沒有找到匹配的特徵")
                break
            imgT = imgs[best_i]
            imgB = imgs[best_j]
            best_H,scale = self._find_homography(imgB['kp'],imgT['kp'], best_matches)
            if best_H is None or not (scale > 0):
                break

            hB, wB = imgB['raw'].shape[:2]
            hT, wT = imgT['raw'].shape[:2]
            pixelB = hB*wB
            pixelT = hT*wT
            if scale < 1:
                pixelB = pixelB*(1/scale)**2
            else:
                pixelT = pixelT*scale**2

            if pixelT > pixelB:
                imgT = imgs[best_j]
                imgB = imgs[best_i]

                good = self._find_good(imgB,imgT)
                best_H, scale = self._find_homography(imgB['kp'], imgT['kp'], good)
                if best_H is None or not (scale > 0):
                    break
    
            self._log(f"[拼接] 底圖: {imgB['name']} | 上圖: {imgT['name']}")
            self._log(f"[拼接] matchs:{maxMatcheNum}")
            stitched = self._warp_and_merge(imgB, imgT, best_H ,scale)
            if stitched is None:
                break
            stitched = self._auto_crop(stitched)
            # 移除已拼接圖片
            del_idx = sorted([best_i, best_j], reverse=True)
            for idx in del_idx:
                del imgs[idx]
            # 添加拼接圖片
            self.best = {
                'raw':stitched,
                'gray':self._gray(stitched),
                'name':f"{imgB['name']} + {imgT['name']}",
                'kp':None,
                'des':None,
                'path':''
            }
            imgs.append(self.best)
            self.stitchedImg.append(imgB['path'])
            self.stitchedImg.append(imgT['path'])

            count += 1
            self._log("-"*30)

        if self.best:
            inputPath = self.image_queue[0]
            current_folder = os.path.dirname(inputPath)
            basename = self.best['name']

            if utils.outputFolder != "":
                folderPath = utils.outputFolder
            elif self.fromFolder:
                folderPath = os.path.dirname(current_folder)
                basename = os.path.basename(current_folder)
            else:
                folderPath = current_folder
            
            basename = f'{basename} stitch_{self.similarity:.0%}'

            utils.save_img(self.best['raw'],folderPath,basename)

            return self.stitchedImg
        return []
    
