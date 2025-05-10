import os,time,io
import cv2
import numpy as np
import utils
from waifu2x_vulkan import waifu2x
from PyQt6 import QtWidgets, QtCore
from waifu2x_Composite import waifu2xComposite
from PIL import Image, ImageFilter,ImageEnhance

class BaseOperateWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.thread = None

        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        params_group = QtWidgets.QGroupBox("Base Operate Parameters")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        params_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        # 縮放
        self.resizeLabel = QtWidgets.QLabel("")
        self.resizeSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.resizeSlider.setRange(0, 50)
        self.resizeSlider.setValue(0)
        self.resizeSlider.valueChanged.connect(self._updateSettings)
        params_layout.addWidget(self.resizeLabel)
        params_layout.addWidget(self.resizeSlider)
        # 亮度
        self.brightLabel = QtWidgets.QLabel("")
        self.brightSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.brightSlider.setRange(0, 50)
        self.brightSlider.setValue(0)
        self.brightSlider.valueChanged.connect(self._updateSettings)
        params_layout.addWidget(self.brightLabel)
        params_layout.addWidget(self.brightSlider)

        # # #
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
        resizeValue = self.resizeSlider.value()/10
        self.resizeLabel.setText(f'Resize: {resizeValue:.1f}')
        brightValue = self.brightSlider.value()/10
        self.brightLabel.setText(f'Brightness: {brightValue:.1f}')

    def run_processing(self):
        # 收集待處理圖片，並更新狀態
        self.image_queue = []
        for file_path, status_code in utils.image_queue.items():
            if status_code == 0:
                self.image_queue.append(file_path)
                utils.image_queue[file_path] = 1

        utils.toMainGUI.put([2, '更新table'])

        if not self.image_queue:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select images!")
            return

        utils.isRunning = True
        utils.toMainGUI.put([1, utils.isRunning])

        self.BaseOperate = BaseOperate()
        self.BaseOperate.image_queue = self.image_queue
        self.BaseOperate.Scale = self.resizeSlider.value()/10
        self.BaseOperate.brightness = self.brightSlider.value()/10
        
        self.thread = QtCore.QThread()
        self.BaseOperate.moveToThread(self.thread)
        # 線程啟動後，透過 lambda 呼叫 Stitcher.run()
        self.thread.started.connect(self.BaseOperate.run)
        # Stitcher 完成後發射 finished 信號，由 handle_result 處理結果
        self.BaseOperate.finished.connect(self.handle_result)
        # 線程結束後清除資源
        self.BaseOperate.finished.connect(self.thread.quit)
        self.BaseOperate.finished.connect(self.BaseOperate.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @QtCore.pyqtSlot(object)
    def handle_result(self, result):
        utils.isRunning = False
        utils.toMainGUI.put([0, ""])
        utils.toMainGUI.put([2, '更新table'])
        utils.toMainGUI.put([1, utils.isRunning])

    def stop_processing(self):
        if self.BaseOperate:
            self.BaseOperate._stop = True  # 通知拼接運算中斷
        utils.toMainGUI.put([0, "Stop requested."])

    def save_parameters(self):
        params = {
            "resize": self.resizeSlider.value()/10,
            "brightness": self.brightSlider.value()/10
        }
        config_data = utils.config_file()
        config_data["BaseOperate"] = params
        utils.config_file(config_data)
        utils.toMainGUI.put([0,"[parameters] Stitch saved."])

    def load_parameters(self):
        config_data = utils.config_file()
        params = config_data.get("BaseOperate", {})
        if params:
            self.resizeSlider.setValue(int(params.get("resize", 1)*10))
            self.brightSlider.setValue(int(params.get("brightness", 1)*10))
            utils.toMainGUI.put([0,"[parameters] Base operate loaded."])
        else:
            utils.toMainGUI.put([0,"[parameters] Not found base operate."])

###################################################################

class BaseOperate(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)  # 處理完成後傳遞結果

    def __init__(self):
        super().__init__()
        self._stop = False  # 中斷旗標
        self.image_queue = None
        self.Scale = 1
        self.brightness = 1

    def _log(self, msg):
        if utils.debug:
            utils.toMainGUI.put([0, f"[Base]{msg}"])

    def run(self):
        
        # 處理每一張輸入圖片
        for img_path in self.image_queue:
            if self._stop:
                break
            self._log(f'[處理圖片] {os.path.basename(img_path)}')
            utils.image_queue[img_path] = 1 #處理中
            utils.toMainGUI.put([2, '更新table'])
            state_code = 2
            img = utils.read_img(img_path)
            if img is not None:
                Exif = img.info.get("exif")
                current_folder = os.path.dirname(img_path)
                basename = os.path.splitext(os.path.basename(img_path))[0]
                print(basename)
                fixName = basename
                
                if self.Scale != 1:
                    img = self._resize(img)
                    fixName += f'_Scale{self.Scale}'
                if self.brightness != 1:
                    img = self._brightness(img)
                    fixName += f'_Bright{self.brightness}'

                if utils.outputFolder != "":
                    folderPath = utils.outputFolder
                elif img_path in utils.fromeFolder: #資料夾開啟
                    folderPath = os.path.dirname(current_folder)
                    folderPath = os.path.join(folderPath, 'output')
                else:
                    folderPath = current_folder
                
                
                if not utils.save_img(img,folderPath,fixName,Exif):
                    state_code = 3
            else:
                state_code = 3

            utils.image_queue[img_path] = state_code
            utils.toMainGUI.put([2, '更新table'])
            self._log('-'*30)

        self.finished.emit(True)

    def _resize(self,img):
        if img is not None and self.Scale != 1 and 0 < self.Scale < 6:
            # 取得原始大小
            w, h = img.size
            # 計算新的尺寸，並轉成整數
            new_w = max(1, int(w * self.Scale))
            new_h = max(1, int(h * self.Scale))
            
            # 使用 Lanczos（高品質）做縮放
            resized = img.resize((new_w, new_h), Image.LANCZOS)

            return resized
        else:
            return None
        
    def _brightness(self,img):
        if img is not None and self.brightness !=1 and 0 < self.brightness < 6:
            
            enhancer = ImageEnhance.Brightness(img)
            brightness_factor = self.brightness  # 例如增加到 150% 的亮度
            img_enhanced = enhancer.enhance(brightness_factor)

            return img_enhanced
        else:
            return None
