import json,os,queue
import cv2
import numpy as np
from PIL import Image, ImageFilter
import pillow_heif
pillow_heif.register_heif_opener()

image_queue = {}
fromeFolder = {}
isRunning = False
toMainGUI = queue.Queue()

format = 'jpg'
quality = 100
debug = False
outputFolder = ""

# 操作 config.json
def config_file(save_config=None):
    file_name = 'config.json'
    def Save(data):
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    # 確保檔案存在
    with open(file_name, 'a') as json_file:
        pass
    if save_config is None:
        try:
            with open(file_name, 'r') as json_file:
                save_config = json.load(json_file)
        except json.decoder.JSONDecodeError:
            save_config = {}
            Save(save_config)
        return save_config
    else:
        Save(save_config)

def save_img(img,folderPath,basename):
    '圖片,路徑'
    try:
        # 判斷如果 img 為 numpy 陣列，則認為它來自 cv2
        if isinstance(img, np.ndarray):
            # 檢查是否為彩色圖像（3 通道）；OpenCV 讀取的彩色圖預設是 BGR 順序
            if len(img.shape) == 3 and img.shape[2] == 3:
                # 將 BGR 格式轉換為 RGB 格式
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 如果是 4 通道圖像（BGRA），同理轉換為 RGBA
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            # 將 NumPy 陣列轉成 PIL 影像物件
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            # 如果傳入的影像既不是 numpy 陣列也不是 PIL 影像，就丟出例外
            toMainGUI.put([0, f"[儲存圖片] 沒有圖片"])
            return False
        
        if not os.path.exists(folderPath):
            os.makedirs(folderPath,exist_ok=True)
        
        path = os.path.join(folderPath, basename)
        # 根據全域變數 format 保存檔案
        fmt = format.lower()
        if fmt == "heic":
            # 若要保存 HEIC 格式，需 pillow-heif 模組支援（pip install pillow-heif）
            save_path = path + ".heic"
            # PIL 在保存 HEIC 時可能需要指定 format 與 quality 參數
            img.save(save_path, format="HEIF", quality=quality , bit_depth=10)
        elif fmt == "png":
            save_path = path + ".png"
            img.save(save_path, format="PNG")
        else:
            # 預設使用 JPEG 格式
            save_path = path + ".jpg"
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(save_path, format="JPEG", quality=quality)
        
        toMainGUI.put([0, f"[儲存圖片] 已保存: {save_path}"])
        return True
    except:
        return False