import sys,os,ctypes
from PyQt6 import QtWidgets, QtGui, QtCore
from waifu2x_vulkan import waifu2x
import utils
import waifu2x_Composite_GUI
import stitch
import color_transfer
import base_operate
import send2trash
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('xpramt.moefusion')

class HandleReturnMessages(QtCore.QThread):
    ReturnMeg = QtCore.pyqtSignal(object,object)
    def run(self):
        while True:
            # 等待狀態更新
            state,parameter = utils.toMainGUI.get()
            self.ReturnMeg.emit(state,parameter)

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MoeFusion v1.0")
        self.setWindowIcon(QtGui.QIcon("C:\APP\@develop\MoeFusion\icon.ico"))
        # 啟用拖曳
        self.setAcceptDrops(True)
        self.current_widget = None
        self.waifu2x_instance = self.initWaifu2x()
        self.initUI()
        # 啟動狀態更新worker
        self.worker = None
        self.worker = HandleReturnMessages()
        self.worker.ReturnMeg.connect(self.update)
        self.worker.start()

    def update(self,state,parameter):
        match state:
            case 0:  # 顯示log
                self.text_log.append(parameter)
                print(parameter)
            case 1:  # 開始按鈕
                if parameter:
                    self.btn_StartStop.setText('⏹️') #Text['Stop'])
                else:
                    self.btn_StartStop.setText('▶️') #Text['Start'])
                    self.btn_StartStop.setEnabled(True)
            case 2:  # 更新table
                self.update_table()

    def initWaifu2x(self):
        '初始化 waifu2x'
        sts = waifu2x.init()
        isCpuModel = sts < 0
        infos = waifu2x.getGpuInfo()
        if infos and len(infos) == 1 and "LLVM" in infos[0]:
            isCpuModel = True
        cpuNum = waifu2x.getCpuCoreNum()
        if isCpuModel:
            waifu2x.initSet(-1, cpuNum)
        else:
            waifu2x.initSet(0)
        waifu2x.setDebug(True)
        return waifu2x

    def initUI(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        # 左側功能區
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        # 共用輸出設定
        setting_group = QtWidgets.QGroupBox('settings')
        setting_layout = QtWidgets.QVBoxLayout(setting_group)
        # 功能選擇下拉選單
        SelectFunc_layout = QtWidgets.QHBoxLayout()
        combo_functionLabel = QtWidgets.QLabel("Select Function: ")
        SelectFunc_layout.addWidget(combo_functionLabel)
        self.combo_function = QtWidgets.QComboBox()
        self.combo_function.addItems(["waifu2x","stitch","transfer","base"])
        self.combo_function.currentIndexChanged.connect(self.change_function)
        SelectFunc_layout.addWidget(self.combo_function)
        setting_layout.addLayout(SelectFunc_layout)
        
        # 輸出格式設定
        Format_layout = QtWidgets.QHBoxLayout()
        Format_layout.addWidget(QtWidgets.QLabel("Output Format:"))
        self.combo_format = QtWidgets.QComboBox()
        self.combo_format.addItems(["AVIF","HEIF","JPG","PNG"])
        self.combo_format.currentIndexChanged.connect(self.change_parameter)
        Format_layout.addWidget(self.combo_format)
        self.lbl_quality = QtWidgets.QLabel("Quality:")
        Format_layout.addWidget(self.lbl_quality)
        self.spin_quality = QtWidgets.QSpinBox()
        self.spin_quality.setRange(0, 100)
        self.spin_quality.valueChanged.connect(self.change_parameter)
        Format_layout.addWidget(self.spin_quality)
        setting_layout.addLayout(Format_layout)
        # 輸出路徑設定
        outputPath_layout = QtWidgets.QHBoxLayout()
        self.outputLineEdit = QtWidgets.QLineEdit()
        self.outputLineEdit.setPlaceholderText("請選擇輸出位置")
        self.outputLineEdit.textChanged.connect(self.change_parameter)
        self.selectOutputButton = QtWidgets.QPushButton("📁")
        self.selectOutputButton.setFixedWidth(40)
        self.selectOutputButton.clicked.connect(self.select_output_folder)
        outputPath_layout.addWidget(self.outputLineEdit)
        outputPath_layout.addWidget(self.selectOutputButton)
        setting_layout.addLayout(outputPath_layout)
        # Debug 模式
        self.chk_debug = QtWidgets.QCheckBox("Debug mode")
        self.chk_debug.checkStateChanged.connect(self.change_parameter)
        setting_layout.addWidget(self.chk_debug)
        # 添加到左側功能區
        left_layout.addWidget(setting_group) 
        # 儲存、讀取
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save Parameter")
        self.btn_save.clicked.connect(self.save_parameters)
        btn_layout.addWidget(self.btn_save)
        self.btn_load = QtWidgets.QPushButton("Load Parameter")
        self.btn_load.clicked.connect(self.load_parameters)
        btn_layout.addWidget(self.btn_load)
        left_layout.addLayout(btn_layout)
        # 按鍵
        btn_layout2 = QtWidgets.QHBoxLayout()
        self.btn_folder = QtWidgets.QPushButton("📂")
        self.btn_folder.clicked.connect(self.open_folder)
        btn_layout2.addWidget(self.btn_folder)
        self.btn_select = QtWidgets.QPushButton("🖼️")
        self.btn_select.clicked.connect(self.select_file)
        btn_layout2.addWidget(self.btn_select)
        self.btn_reset = QtWidgets.QPushButton("🔄")
        self.btn_reset.clicked.connect(self.reset_images)
        btn_layout2.addWidget(self.btn_reset)
        self.btn_clear = QtWidgets.QPushButton("❌")
        self.btn_clear.clicked.connect(self.clear_images)
        btn_layout2.addWidget(self.btn_clear)
        self.btn_delete = QtWidgets.QPushButton("🗑️")
        self.btn_delete.clicked.connect(self.delete_images)
        btn_layout2.addWidget(self.btn_delete)
        left_layout.addLayout(btn_layout2)
        # 檔案清單
        self.table_files = QtWidgets.QTableWidget()
        self.table_files.setColumnCount(2)
        self.table_files.setHorizontalHeaderLabels(["File Name", "Status"])
        self.table_files.setColumnWidth(1, 60) 
        self.table_files.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table_files.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_files.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        # 設定右鍵菜單
        self.table_files.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_files.customContextMenuRequested.connect(self.show_context_menu)
        left_layout.addWidget(self.table_files)
        
        # 添加左側到主介面
        main_layout.addLayout(left_layout)

        # 右側功能區
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        #right_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMaximumSize)
        # 利用 QStackedWidget 放置各功能的 Widget
        self.stacked_widget = QtWidgets.QStackedWidget()
        self.stacked_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        # 注意：在建立功能 Widget 時，明確指定 parent 與額外參數
        waifu2x_widget = waifu2x_Composite_GUI.Waifu2xCompositeWidget(parent=self)
        waifu2x_widget.waifu2x_instance = self.waifu2x_instance
        stitch_widget = stitch.StitchWidget(parent=self)
        stitch_widget.waifu2x_instance = self.waifu2x_instance
        transfer_widget = color_transfer.TransferWidget(parent=self)
        base_widget = base_operate.BaseOperateWidget(parent=self)
        self.stacked_widget.addWidget(waifu2x_widget)
        self.stacked_widget.addWidget(stitch_widget)
        self.stacked_widget.addWidget(transfer_widget)
        self.stacked_widget.addWidget(base_widget)
        self.current_widget = self.stacked_widget.currentWidget()
        right_layout.addWidget(self.stacked_widget)
        # 開始按鈕
        self.btn_StartStop = QtWidgets.QPushButton("▶️")
        self.btn_StartStop.clicked.connect(self.start_stop)
        right_layout.addWidget(self.btn_StartStop)

        # 日誌區
        self.text_log = QtWidgets.QTextEdit()
        right_layout.addWidget(self.text_log)

        # 添加右側到主介面
        main_layout.addLayout(right_layout)

        self.load_parameters()

    def start_stop(self):
        if not utils.isRunning:
            self.current_widget.run_processing()
        else:
            self.btn_StartStop.setEnabled(False)
            self.btn_StartStop.setText('⌛')
            self.current_widget.stop_processing()
            
    def change_function(self, index):
        self.stacked_widget.setCurrentIndex(index)
        self.current_widget = self.stacked_widget.currentWidget()
        self.stacked_widget.setMinimumWidth(300)
        self.stacked_widget.setMaximumHeight(self.current_widget.sizeHint().height())

    def change_parameter(self):
        utils.format = self.combo_format.currentText()
        if utils.format != "PNG":
            self.lbl_quality.show()
            self.spin_quality.show()
        else:
            self.lbl_quality.hide()
            self.spin_quality.hide()
            
        utils.quality = self.spin_quality.value()
        utils.debug = self.chk_debug.isChecked()
        utils.outputFolder = self.outputLineEdit.text()

    def load_parameters(self):
        config_data = utils.config_file()
        params = config_data.get("general", {})
        if params:
            self.combo_format.setCurrentText(params.get("output_format", "jpg"))
            self.spin_quality.setValue(params.get("quality", 100))
            self.chk_debug.setChecked(params.get("debug_enabled", False))
            self.combo_function.setCurrentIndex(params.get("function", 0))
            self.outputLineEdit.setText(params.get("outputFolder", ""))
            utils.toMainGUI.put([0,"[parameters] Main loaded."])
        else:
            utils.toMainGUI.put([0,"[parameters] Not found main."])

    def save_parameters(self):
        params = {
            "output_format": self.combo_format.currentText(),
            "quality": self.spin_quality.value(),
            "debug_enabled": self.chk_debug.isChecked(),
            "function": self.combo_function.currentIndex(),
            "outputFolder": self.outputLineEdit.text()
        }
        config_data = utils.config_file()
        config_data["general"] = params
        utils.config_file(config_data)
        utils.toMainGUI.put([0,"[parameters] Main saved."])

    def dragEnterEvent(self, event):
        '拖曳事件：允許拖入圖片檔案'
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = os.path.normpath(url.toLocalFile())
            if os.path.isfile(file_path) and file_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp" ,".heic" ,".hif")):
                if file_path not in utils.image_queue:
                    utils.image_queue[file_path] = 0
        self.update_table()

    def select_output_folder(self):
        '開啟 QT 資料夾選擇介面'
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇輸出資料夾", "")
        # 將選擇的路徑填入單行文字視窗
        self.outputLineEdit.setText(folder)

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if folder:
            valid_extensions = (".jpg", ".jpeg", ".png", ".webp" ,".heic" ,".hif")
            for file_name in os.listdir(folder):
                if file_name.lower().endswith(valid_extensions):
                    full_path = os.path.normpath(os.path.join(folder, file_name))
                    if full_path not in utils.image_queue:
                        utils.image_queue[full_path] = 0
                        utils.fromeFolder[full_path] = 0
            self.update_table()

    def select_file(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Image Files (*.jpg *.jpeg *.png *.webp *.heic *.hif)"
        )
        if files:
            for full_path_raw in files:
                full_path = os.path.normpath(full_path_raw)
                if full_path not in utils.image_queue:
                    utils.image_queue[full_path] = 0
            self.update_table()

    def show_context_menu(self, position):
        '右鍵菜單：點右鍵時提供 Reset 與 Clear 選項，可對多行執行操作'
        # 取得全域選中的行（返回 QTableWidgetSelectionRange 或 QModelIndex 列表）
        selected_rows = [index.row() for index in self.table_files.selectionModel().selectedRows()]

        # 若沒有多選，則退化為點擊單行
        if not selected_rows:
            index = self.table_files.indexAt(position)
            if index.isValid():
                selected_rows = [index.row()]

        menu = QtWidgets.QMenu()
        # 使用 lambda 傳入所選行列表
        reset_action = menu.addAction("🔄")
        reset_action.triggered.connect(lambda: self.reset_rows(selected_rows))
        clear_action = menu.addAction("❌")
        clear_action.triggered.connect(lambda: self.clear_rows(selected_rows))
        delete_action = menu.addAction("🗑️")
        delete_action.triggered.connect(lambda: self.delete_rows(selected_rows))
        
        menu.setMaximumWidth(80)
        menu.exec(self.table_files.viewport().mapToGlobal(position))


    def reset_rows(self, rows):
        for row in rows:
            file_item = self.table_files.item(row, 0)
            file_path = file_item.data(QtCore.Qt.ItemDataRole.UserRole)
            if file_path in utils.image_queue:
                utils.image_queue[file_path] = 0
        self.update_table()

    def clear_rows(self, rows):
        for row in rows:
            file_item = self.table_files.item(row, 0)
            file_path = file_item.data(QtCore.Qt.ItemDataRole.UserRole)
            if file_path in utils.image_queue:
                del utils.image_queue[file_path]
            if file_path in utils.fromeFolder:
                del utils.fromeFolder[file_path]
        self.update_table()

    def delete_rows(self, rows):
        for row in rows:
            file_item = self.table_files.item(row, 0)
            file_path = file_item.data(QtCore.Qt.ItemDataRole.UserRole)
            try:
                normalized_path = os.path.normpath(file_path)
                send2trash.send2trash(normalized_path)  # 移動檔案到回收桶
                if file_path in utils.image_queue:
                    del utils.image_queue[file_path]
                if file_path in utils.fromeFolder:
                    del utils.fromeFolder[file_path]
                self.update_table()
            except Exception as e:
                print(f"無法刪除檔案 {file_path}: {e}")  # 捕捉例外並輸出錯誤（可替換為日誌記錄）
            
    def reset_images(self):
        # 將所有圖片的狀態重設為 "0"（等待）
        for file_path in utils.image_queue:
            utils.image_queue[file_path] = 0
        # 更新表格顯示新的狀態
        self.update_table()

    def clear_images(self):
        utils.image_queue = {}
        utils.fromeFolder = {}
        self.update_table()

    def delete_images(self):
        # 遍歷圖片佇列，將每個圖片檔案移動到系統資源回收桶
        for file_path in list(utils.image_queue.keys()):  # 使用 list() 避免運行時修改字典
            #try:
                normalized_path = os.path.normpath(file_path)
                send2trash.send2trash(normalized_path)  # 移動檔案到回收桶
                # 清空圖片佇列和來源資料夾字典，重置狀態
                utils.image_queue = {}
                utils.fromeFolder = {}
                # 更新表格顯示新的狀態
                self.update_table()
            #except Exception as e:
            #    print(f"無法刪除檔案 {file_path}: {e}")  # 捕捉例外並輸出錯誤（可替換為日誌記錄）
        
    def update_table(self):
        # 先清空表格內容
        self.table_files.setRowCount(0)
        # 定義狀態碼對應的文字
        status_mapping = {
            0 : "等待",
            1 : "處理中",
            2 : "完成",
            3 : "錯誤"
        }
        # 遍歷 utils.image_queue 的每筆資料
        for file_path, status_code in utils.image_queue.items():
            row = self.table_files.rowCount()
            self.table_files.insertRow(row)
            
            # 第1欄：檔名（使用 os.path.basename 取得檔名）
            file_item = QtWidgets.QTableWidgetItem(os.path.basename(file_path))
            file_item.setData(QtCore.Qt.ItemDataRole.UserRole, file_path)  # 儲存完整路徑
            self.table_files.setItem(row, 0, file_item)
            # 第2欄：狀態，根據狀態碼取得對應文字
            status_text = status_mapping.get(status_code, "未知")
            status_item = QtWidgets.QTableWidgetItem(status_text)
            status_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table_files.setItem(row, 1, status_item)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(0, 0, 0))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(20, 20, 20))
    app.setPalette(dark_palette)
    default_font = QtGui.QFont('Microsoft JhengHei', 12)
    app.setFont(default_font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
