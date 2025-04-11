from PyQt6 import QtWidgets, QtCore
import utils
from waifu2x_Composite import waifu2xComposite

# waifu2x_Composite 功能的 Widget，包含參數設定、save、load 與 run
class Waifu2xCompositeWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.stop = True
        self.worker = None
        self.waifu2x_instance = None

        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 功能參數區塊
        params_group = QtWidgets.QGroupBox("Waifu2x Composite Parameters")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        params_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop )

        # Scale Factor
        scale_layout = QtWidgets.QHBoxLayout()
        scale_layout.addWidget(QtWidgets.QLabel("Scale Factor:"))
        self.spin_scale = QtWidgets.QSpinBox()
        self.spin_scale.setRange(1, 8)
        self.spin_scale.setValue(2)
        scale_layout.addWidget(self.spin_scale)
        params_layout.addLayout(scale_layout)

        # Top Layer Model
        model_layout = QtWidgets.QHBoxLayout()
        model_layout.addWidget(QtWidgets.QLabel("Top Layer Model:"))
        self.combo_model = QtWidgets.QComboBox()
        self.combo_model.addItems(["CUNET_NO_NOISE", "CUNET_NOISE1"])
        model_layout.addWidget(self.combo_model)
        params_layout.addLayout(model_layout)

        # Gaussian Blur factor
        blur_layout = QtWidgets.QHBoxLayout()
        blur_layout.addWidget(QtWidgets.QLabel("Gaussian Blur factor:"))
        self.spin_blur = QtWidgets.QDoubleSpinBox()
        self.spin_blur.setRange(0.1, 2)
        self.spin_blur.setSingleStep(0.1)
        self.spin_blur.setValue(1.0)
        blur_layout.addWidget(self.spin_blur)
        params_layout.addLayout(blur_layout)

        # Gamma 輸入
        gamma_layout = QtWidgets.QHBoxLayout()
        gamma_layout.addWidget(QtWidgets.QLabel("Mask Gamma:"))
        self.spin_gamma = QtWidgets.QDoubleSpinBox()
        self.spin_gamma.setRange(0.1, 2.0)
        self.spin_gamma.setSingleStep(0.1)
        self.spin_gamma.setValue(1.0)
        gamma_layout.addWidget(self.spin_gamma)
        params_layout.addLayout(gamma_layout)

        # Input Range
        range_layout = QtWidgets.QHBoxLayout()
        range_layout.addWidget(QtWidgets.QLabel("Input Range Low:"))
        self.spin_low = QtWidgets.QSpinBox()
        self.spin_low.setRange(0, 255)
        self.spin_low.setValue(10)
        range_layout.addWidget(self.spin_low)
        range_layout.addWidget(QtWidgets.QLabel("High:"))
        self.spin_high = QtWidgets.QSpinBox()
        self.spin_high.setRange(0, 255)
        self.spin_high.setValue(240)
        range_layout.addWidget(self.spin_high)
        params_layout.addLayout(range_layout)
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
        self.load_parameters()
    
    def save_parameters(self):
        params = {
            "scale": self.spin_scale.value(),
            "top_model": self.combo_model.currentText(),
            "blur_factor": self.spin_blur.value(),
            "gamma": self.spin_gamma.value(),
            "range_low": self.spin_low.value(),
            "range_high": self.spin_high.value()
        }
        config_data = utils.config_file()
        config_data["waifu2x"] = params
        utils.config_file(config_data)
        utils.toMainGUI.put([0,"[parameters] waifu2x saved."])

    def load_parameters(self):
        config_data = utils.config_file()
        params = config_data.get("waifu2x", {})
        if params:
            self.spin_scale.setValue(params.get("scale", 2))
            self.combo_model.setCurrentText(params.get("top_model", "CUNET_NO_NOISE"))
            self.spin_blur.setValue(params.get("blur_factor", 1.0))
            self.spin_gamma.setValue(params.get("gamma", 1.0))
            self.spin_low.setValue(params.get("range_low", 10))
            self.spin_high.setValue(params.get("range_high", 240))
            utils.toMainGUI.put([0,"[parameters] waifu2x loaded."])
        else:
            utils.toMainGUI.put([0,"[parameters] Not found waifu2x."])

    def run_processing(self):
        if not utils.image_queue:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select images!")
            return
        self.stop = False
        utils.isRunning = True
        utils.toMainGUI.put([1,utils.isRunning])

        self.process_next_image()

    def stop_processing(self):
        self.stop = True

    def process_next_image(self):
        all_processed = True
        for img_path, status_code in utils.image_queue.items():
            if self.stop:
                break

            if status_code == 0:
                all_processed = False
                utils.image_queue[img_path] = 1
                utils.toMainGUI.put([2,'更新table'])
                # 建立並啟動 waifu2x_Composite 的 Worker
                self.worker = waifu2xComposite()
                self.worker.image_path = img_path
                self.worker.scale = self.spin_scale.value()
                self.worker.top_model = self.combo_model.currentText()
                self.worker.gamma = self.spin_gamma.value()
                self.worker.input_low = self.spin_low.value()
                self.worker.input_high = self.spin_high.value()
                self.worker.blur_factor = self.spin_blur.value()
                self.worker.waifu2x = self.waifu2x_instance

                self.worker.finished.connect(lambda path, success: self.on_finished(path, success))
                self.worker.start()
                utils.toMainGUI.put([2,'更新table'])
                break
            
        if all_processed:
            utils.isRunning = False
            utils.toMainGUI.put([0,"All images processed."])
            utils.toMainGUI.put([1,utils.isRunning])
            
    def on_finished(self, path, success):
        if success:
            utils.image_queue[path] = 2
        else:
            utils.image_queue[path] = 3
        utils.toMainGUI.put([0,"-"*30])
        self.worker.quit()
        self.worker.wait()
        self.worker = None
        utils.toMainGUI.put([2,'更新table'])
        self.process_next_image()