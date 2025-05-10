from concurrent.futures import ThreadPoolExecutor
import os, time, io
import numpy as np
import cv2
import utils
from PyQt6 import QtCore
from PIL import Image, ImageFilter
from waifu2x_vulkan import waifu2x
from pillow_heif import register_heif_opener
register_heif_opener()
    
class waifu2xComposite(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str, bool)  # 完成訊息，包含檔案路徑和成功/失敗標誌
    
    def __init__(self):
        super().__init__()
        try:
            self.waifu2x = None
            self.isReturnImg = False
            # 輸入圖片
            self.inputImg = None
            self.image_path = ''
            # 從config讀取參數
            config_data = utils.config_file()
            params = {}
            params = config_data.get("waifu2x", {})
            self.scale = params.get("scale", 2)
            #"MODEL_CUNET_NO_NOISE" 或 "MODEL_CUNET_NOISE1"
            self.top_model = params.get("top_model", "CUNET_NO_NOISE")
            self.gamma = params.get("gamma", 1.0)
            self.blur_factor = params.get("blur_factor", 1.0)
            self.input_low = params.get("range_low", 10)
            self.input_high = params.get("range_high", 240)
            # 內部產生
            self.base_name = ''
            self.executor = ThreadPoolExecutor(max_workers=8) 
            self._log(' 初始化完成')
        except:
            self._log(' 初始化失敗')

    def _log(self, msg):
        'msg'
        if utils.debug:
            utils.toMainGUI.put([0, f"[waifu2x]{msg}"])

    def debug_save(self, img, step_order, step_name):
        """
        若 debug_enabled 為 True,將 PIL 影像存至 debug 資料夾。
        檔名格式：
        - 若圖像模式為 RGBA,存成 PNG
        - 其他模式存成 JPEG
        """
        if not utils.debug or self.inputImg is not None:
            return

        debug_folder = "debug"
        os.makedirs(debug_folder, exist_ok=True)

        if img.mode != "RGBA":
            out_name = os.path.join(debug_folder, f"{self.base_name}_{step_order}.{step_name}.jpg")
            self.executor.submit(img.save, out_name, quality=100)
        else:
            out_name = os.path.join(debug_folder, f"{self.base_name}_{step_order}.{step_name}.png")
            self.executor.submit(img.save, out_name)

    def adjust_gamma(self,image, gamma=1.0, input_range=(0, 255)):
        black, white = input_range
        # 保證 black 和 white 值的合法性
        black = max(0, min(white - 1, black))
        white = min(255, max(black + 1, white))
        # 將 image 轉換成 float 避免 uint8 算術運算中的 underflow/overflow 問題
        image_float = image.astype(np.float32)
        image_normalized = np.clip((image_float - black) / (white - black) * 255, 0, 255).astype(np.uint8)
        inv_gamma = 1.0 / gamma
        # 建立 gamma 對應的查找表
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image_normalized, table)

    def process_layer(self,image_data, Model, task_id):
        """
        通用函式：處理圖層(底圖或上層圖）
        - image_data: 原始影像的二進制資料
        - model: waifu2x 模型
        - task_id: 任務 ID(底圖用 1,上層圖用 2)
        """
        if  Model == 3: # 底圖
            model = waifu2x.MODEL_CUNET_NOISE3
            task = 'base'
        else: # 上層圖
            task = 'top'
            if Model == 0:
                model = waifu2x.MODEL_CUNET_NO_NOISE
            else:
                model = waifu2x.MODEL_CUNET_NOISE1

        debug_filepath = os.path.join("debug", f"{self.base_name}_0.{task}_{self.scale}x_{Model}")
        
        path_jpg = debug_filepath + '.jpg'
        path_png = debug_filepath + '.png'
        if os.path.exists(path_jpg):
            self._log(f" 使用已存在圖片 {path_jpg}")
            img = Image.open(path_jpg)
            return img
        elif os.path.exists(path_png):
            self._log(f" 使用已存在圖片 {path_png}")
            img = Image.open(path_png)
            return img
        else:
            ret = self.waifu2x.add(image_data, model, task_id, self.scale, tileSize=128)
            if ret <= 0:
                raise Exception(f"無法為 {model} 模型新增任務 ")
            layer_bytes = None
            while True:
                time.sleep(1)
                info = self.waifu2x.load(0)
                if info:
                    data, fmt, out_id, tick = info
                    layer_bytes = data
                    break
            img = Image.open(io.BytesIO(layer_bytes))
            
            self.debug_save(img.copy(), 0, f"{task}_{self.scale}x_{Model}")

            return img

    def run(self):
        '返回CV2圖片'
        self.input_range = (self.input_low, self.input_high )
        if self.image_path:
            self.base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        self._log(f" Scale: {self.scale}, Top Model: {self.top_model}, Gamma: {self.gamma}, Input Range: {self.input_range}, Blur Radius: {self.blur_factor}")
        try:
            # 讀取原始影像 轉為bytes
            if self.image_path:
                self._log(f" 讀取圖片 {self.base_name}")
                with open(self.image_path, "rb") as f:
                    img_bytes = f.read()
            elif self.inputImg is not None:        
                success, buffer = cv2.imencode(".jpg",self.inputImg,[cv2.IMWRITE_JPEG_QUALITY, 100])
                if success:
                    img_bytes = buffer.tobytes()
            else:
                return None
            
            # 1. 生成底圖：使用 MODEL_CUNET_NOISE3
            self._log(" 生成底圖")
            base_img = self.process_layer(img_bytes,3,1)
            self._log(" 生成上圖")
            # 2. 生成上層圖：根據選擇模型
            if self.top_model == "CUNET_NO_NOISE":
                top_model = 0
            else:
                top_model = 1
            top_img = self.process_layer(img_bytes,top_model,2)

            def sobel(img):
                '''
                Sobel邊緣檢測
                output: np array
                '''
                img_np = np.array(img).astype(np.float32)
                sobelx = cv2.Sobel(img_np, cv2.CV_32F, 1, 0, ksize=3)
                sobely = cv2.Sobel(img_np, cv2.CV_32F, 0, 1, ksize=3)
                sobel = cv2.magnitude(sobelx, sobely)
                sobel = np.uint8(255 * cv2.normalize(sobel, None, 0, 1, cv2.NORM_MINMAX))
                inverted = 255 - sobel
                return inverted
            
            # 製作 mask：從上層圖複製產生
            self._log(f" 生成遮罩")
            # 轉灰階
            mask_img = top_img.copy().convert("L")  
            self.debug_save(mask_img,1, "mask_gray")
            # 邊緣偵測
            sobel_inverted0 = sobel(mask_img)
            self.debug_save(Image.fromarray(sobel_inverted0), 3.1, "mask_sobel_inverted0")
            # 模糊
            width, height = top_img.size
            radiu = np.sqrt(width*height)/1000
            radius = round(radiu * self.blur_factor, 1)
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius))
            self.debug_save(mask_img, 2, "mask_blur")
            self._log(f' radius: {radiu:.1f}*{self.blur_factor:.1f} = {radius}')
            # 邊緣偵測
            sobel_inverted = sobel(mask_img)
            self.debug_save(Image.fromarray(sobel_inverted), 3.2, "mask_sobel_inverted")
            # 灰階調整
            adjusted_mask_np = self.adjust_gamma(sobel_inverted, gamma=self.gamma, input_range=self.input_range)
            self.debug_save(Image.fromarray(adjusted_mask_np), 4, "mask_adjusted")
            final_mask = Image.fromarray(adjusted_mask_np).convert("L")

            # 根據 mask 調整上層圖透明度
            self._log(f" 上圖根據遮罩透明化")
            if top_img.mode == "RGBA":
                r, g, b , a= top_img.split() # 分離通道
                # 將原始 alpha 與 final_mask 轉換為 numpy 陣列
                orig_alpha_np = np.array(a, dtype=np.uint8)
                final_mask_np = np.array(final_mask, dtype=np.uint8)
                # 逐像素取最小值
                combined_alpha_np = np.minimum(orig_alpha_np, final_mask_np)
                # 再轉回 PIL 影像（單通道）
                final_mask = Image.fromarray(combined_alpha_np, mode="L")
            else:
                r, g, b = top_img.split() # 分離通道
            top_img = Image.merge("RGBA", (r, g, b, final_mask))
            self.debug_save(top_img, 5, "top_transparent")
            
            # 疊加上層圖到底圖上
            self._log(f" 疊加上圖到底圖")
            base_img = base_img.convert("RGBA")
            if top_img.size != base_img.size:
                top_img = top_img.resize(base_img.size, Image.LANCZOS)
            result_img = Image.alpha_composite(base_img, top_img)
            self.debug_save(result_img, 6, "composite")

            self._log(f" 完成 {self.base_name}")

            if self.isReturnImg:
                return cv2.cvtColor(np.array(result_img), cv2.COLOR_RGBA2BGRA)
            else:
                # 儲存輸出影像
                inputPath = self.image_path
                current_folder = os.path.dirname(inputPath)
                basename = os.path.splitext(os.path.basename(inputPath))[0]
                if utils.outputFolder != "":
                    folderPath = utils.outputFolder
                elif inputPath in utils.fromeFolder:
                    folderPath = os.path.dirname(current_folder)
                    folderPath = os.path.join(folderPath, 'waifu2x_output')
                else:
                    folderPath = current_folder
                
                basename = basename + f"_{self.scale}x,model {top_model},blur {self.blur_factor:.1f},range {self.input_range}"

                utils.save_img(result_img,folderPath,basename)

                self.finished.emit(self.image_path, True)  # 成功

        except Exception as e:
            self._log(f"[Error] {e}")
            self.finished.emit(self.image_path, False)  # 失敗