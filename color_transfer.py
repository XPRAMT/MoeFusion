from PyQt6 import QtWidgets,QtCore,QtGui
import os
import cv2
import numpy as np
import utils

def getSystemRatio():
    '取得系統縮放比例'
    screen = QtGui.QGuiApplication.primaryScreen()
    return screen.devicePixelRatio()

class TransferWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.thread = None
        self.transfer = None

        self.reference_path = ''
        self.input_image_paths = []
        self.image_queue = []

        self.init_ui()
        
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        #選擇參考圖片
        btn_select_ref = QtWidgets.QPushButton("選擇參考圖片")
        btn_select_ref.clicked.connect(self.select_reference_image)
        layout.addWidget(btn_select_ref)

        # 顯示參考圖片
        self.label_ref = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        #self.label_ref.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.label_ref)
        
        # 儲存、讀取
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save Parameter")
        self.btn_save.clicked.connect(self.save_parameters)
        btn_layout.addWidget(self.btn_save)
        self.btn_load = QtWidgets.QPushButton("Load Parameter")
        self.btn_load.clicked.connect(self.load_parameters)
        btn_layout.addWidget(self.btn_load)
        layout.addLayout(btn_layout)

        self.load_parameters()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self._setCover()
        
    def select_reference_image(self):
        self.reference_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "選擇參考圖片", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        self._setCover()

    def _setCover(self):
        if self.reference_path:
            # 建立 QPixmap 並從檔案讀取圖片
            pixmap = QtGui.QPixmap(self.reference_path)
            # 如果需要縮放圖片以適應 label 的大小
            # 保持圖片比例並使用平滑縮放
            SystemRatio = getSystemRatio()
            pixmap.setDevicePixelRatio(SystemRatio)
            scaled_pixmap = pixmap.scaled(
                self.label_ref.size()*SystemRatio, 
                QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            self.label_ref.setPixmap(scaled_pixmap)
        else:
            # 如果沒有選擇檔案，清除 label 上的圖片
            self.reference_path = None
            self.label_ref.clear()
        
    def run_processing(self):
        if not self.reference_path:
            utils.toMainGUI.put([0, "請先選擇參考圖片"])
            return
        
        # 收集待處理圖片，並更新狀態
        self.image_queue = []
        for file_path, status_code in utils.image_queue.items():
            if status_code == 0:
                self.image_queue.append(file_path)

        if not self.image_queue:
            utils.toMainGUI.put([0, "請先選擇輸入圖片"])
            return

        utils.isRunning = True
        utils.toMainGUI.put([1, utils.isRunning])

        # 建立 ImageStitcher 並移至新的線程中
        self.transfer = ColorTransfer()
        self.transfer.targetPath = self.reference_path
        self.transfer.image_queue = self.image_queue

        self.thread = QtCore.QThread()
        self.transfer.moveToThread(self.thread)
        # 線程啟動後，透過 lambda 呼叫 Stitcher.run()
        self.thread.started.connect(self.transfer.run)
        # Stitcher 完成後發射 finished 信號，由 handle_result 處理結果
        self.transfer.finished.connect(self.handle_result)
        # 線程結束後清除資源
        self.transfer.finished.connect(self.thread.quit)
        self.transfer.finished.connect(self.transfer.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @QtCore.pyqtSlot(object)
    def handle_result(self, result):
        result
        utils.isRunning = False
        utils.toMainGUI.put([1, utils.isRunning])

    def stop_processing(self):
        if self.transfer:
            self.transfer._stop = True  # 通知拼接運算中斷
        utils.toMainGUI.put([0, "Stop requested."])

    def save_parameters(self):
        params = {
            'reference':self.reference_path
        }
        config_data = utils.config_file()
        config_data["transfer"] = params
        utils.config_file(config_data)
        utils.toMainGUI.put([0,"[parameters] Color transfer saved."])

    def load_parameters(self):
        config_data = utils.config_file()
        params = config_data.get("transfer", {})
        if params:
            self.reference_path = params.get("reference", '')
            self._setCover()
            utils.toMainGUI.put([0,"[parameters] Color transfer loaded."])
        else:
            utils.toMainGUI.put([0,"[parameters] Not found  Color transfer."])

###################################################################

class ColorTransfer(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)  # 處理完成後傳遞結果

    def __init__(self):
        super().__init__()
        self._stop = False  # 中斷旗標
        self.image_queue = None
        self.targetPath = ''
        self.ref_img = None

    def _log(self, msg):
        if utils.debug:
            utils.toMainGUI.put([0, f"[transfer]{msg}"])

    def run(self):

        if self.targetPath:
            self.ref_img = utils.read_img(self.targetPath,True)
            if self.ref_img is None:
                return
        
        # 處理每一張輸入圖片
        for img_path in self.image_queue:
            self._log(f'[處理圖片] {os.path.basename(img_path)}')
            utils.image_queue[img_path] = 1 #處理中
            utils.toMainGUI.put([2, '更新table'])
            state_code = 2
            img = utils.read_img(img_path)
            if img is not None:
                Exif = img.info.get("exif")
                arr = np.array(img, dtype=np.uint8)
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
                result_img = self.match_color(arr,self.ref_img)

                current_folder = os.path.dirname(img_path)
                basename = os.path.splitext(os.path.basename(img_path))[0]
                print(basename)

                if utils.outputFolder != "":
                    folderPath = utils.outputFolder
                elif img_path in utils.fromeFolder: #資料夾開啟
                    folderPath = os.path.dirname(current_folder)
                    folderPath = os.path.join(folderPath, 'output')
                else:
                    folderPath = current_folder
                
                if not utils.save_img(result_img,folderPath,f'{basename}_fix',Exif):
                    state_code = 3
            else:
                state_code = 3

            utils.image_queue[img_path] = state_code
            utils.toMainGUI.put([2, '更新table'])
            self._log('-'*30)

        self.finished.emit(True)

    def match_color(self, source, target):
        """
        (source, target)

        參考論文: Reinhard et al. "Color Transfer between Images" (色彩轉換)
        
        處理流程:
        1. 將圖片轉換至 LAB 色彩空間
        2. 分別計算參考圖與輸入圖各通道的平均值與標準差
        3. 調整輸入圖的每個通道，使其統計數據符合參考圖
        4. 轉回 BGR 色彩空間並返回結果
        """
        # 轉換到 LAB 色彩空間
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        
        # 分離通道
        l_source, a_source, b_source = cv2.split(source_lab)
        l_target, a_target, b_target = cv2.split(target_lab)
        
        # 計算每個通道的平均值與標準差
        l_mean_src, l_std_src = cv2.meanStdDev(l_source)
        a_mean_src, a_std_src = cv2.meanStdDev(a_source)
        b_mean_src, b_std_src = cv2.meanStdDev(b_source)
        
        l_mean_tar, l_std_tar = cv2.meanStdDev(l_target)
        a_mean_tar, a_std_tar = cv2.meanStdDev(a_target)
        b_mean_tar, b_std_tar = cv2.meanStdDev(b_target)
        
        # 轉換成 float 型態
        l_mean_src, a_mean_src, b_mean_src = float(l_mean_src), float(a_mean_src), float(b_mean_src)
        l_std_src, a_std_src, b_std_src = float(l_std_src), float(a_std_src), float(b_std_src)
        l_mean_tar, a_mean_tar, b_mean_tar = float(l_mean_tar), float(a_mean_tar), float(b_mean_tar)
        l_std_tar, a_std_tar, b_std_tar = float(l_std_tar), float(a_std_tar), float(b_std_tar)
        
        # 單通道調整函式
        def transfer_channel(source , mean_tar, std_tar, mean_src, std_src):
            if std_tar < 1e-8:  # 避免除以零
                std_tar = 1e-8
            # 調整公式
            matched_src = (source - mean_src) * (std_tar/std_src) + mean_tar
            return np.clip(matched_src, 0, 255).astype(np.uint8)
        
        l_result = transfer_channel(l_source, l_mean_tar, l_std_tar, l_mean_src, l_std_src)
        a_result = transfer_channel(a_source, a_mean_tar, a_std_tar, a_mean_src, a_std_src)
        b_result = transfer_channel(b_source, b_mean_tar, b_std_tar, b_mean_src, b_std_src)
        
        # 合併通道並轉換回 BGR 色彩空間
        result_lab = cv2.merge([l_result, a_result, b_result])
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        return result

