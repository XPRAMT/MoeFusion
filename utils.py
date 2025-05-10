import json,os,queue
import cv2
import numpy as np
from PIL import Image, ImageFilter
import pillow_avif
import pillow_heif
pillow_heif.register_heif_opener()
import piexif

image_queue = {}
fromeFolder = {}
isRunning = False
toMainGUI = queue.Queue()

format = 'JPG'
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

def read_img(path, ReCv2=False):
    """
    讀取圖片，預設返回 PIL Image\n
    若 cv2=True，回傳 OpenCV BGR numpy.ndarray\n
    失敗時回傳 None
    """
    try:
        img = Image.open(path)
        toMainGUI.put([0, f"[讀取圖片] 讀取 {os.path.basename(path)} 成功 | {img.format} {img.mode}"])
    except Exception as e:
        toMainGUI.put([2, f"[讀取圖片] 讀取 {os.path.basename(path)} 失敗\n{e}"])
        return None
    
    img.convert('RGBA')
    # 如果要求 cv2 形式，先確保是 RGB，再轉 BGR numpy array
    if ReCv2:
        arr = np.array(img, dtype=np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        return arr
    else:
        # 否則直接回傳 PIL Image
        return img
        
def save_img(img,folderPath,basename,Exif=None):
        save_path = None
        if Exif is None:
            Exif = b""
    #try:
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
        fmt = format.upper()
        if fmt == "HEIF" or fmt == "AVIF":
            exif_dict = piexif.load(Exif) if Exif else {"Exif": {}}
            exif_dict["Exif"][piexif.ExifIFD.ColorSpace] = 65535  # Uncalibrated
            Exif = piexif.dump(exif_dict)
            with open("heic_fix.icc", "rb") as f:
                icc_bytes = b""#f.read()
            save_path = path + f"_{fmt}.heic"
            # PIL 在保存 HEIC 時可能需要指定 format 與 quality 參數
            img.save(save_path, format=fmt, quality=quality , exif=Exif,icc_profile=icc_bytes)
        elif fmt == "PNG":
            save_path = path + ".png"
            img.save(save_path, format="PNG")
        elif fmt == "JPG":
            # 預設使用 JPEG 格式
            save_path = path + ".jpg"
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(save_path, format="JPEG", quality=quality,exif=Exif)

        if save_path is not None:
            toMainGUI.put([0, f"[儲存圖片] 已保存: {save_path}"])
            return True
        else:
            toMainGUI.put([0, f"[儲存圖片] 失敗: {save_path}"])
            return False
    #except Exception as e:
    #    toMainGUI.put([0, f"[儲存圖片] 失敗\n{e}"])
    #    return False