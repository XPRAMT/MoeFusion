import json,os,queue
import cv2
import numpy as np
from PIL import Image, ImageFilter
import pillow_avif
import pillow_heif
pillow_heif.register_heif_opener()
import exiftool
import piexif
import subprocess
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
        
def save_img(img,folderPath,basename,RawImg=None):
        save_path = None
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
        if fmt in ("HEIF", "AVIF"):
            fmt = "heic" if fmt == "HEIF" else "avif"
            tmp_path = "_tmp.png"
            save_path = path + "." + fmt
            toMainGUI.put([0, f"[儲存圖片] 保存暫存檔:{tmp_path}"])
            img.save(tmp_path, format="PNG", quality_mode="lossless")
            toMainGUI.put([0, f"[儲存圖片] 轉為{fmt}"])
            w, h = img.size
            thumb_w = min(w // 4, 512)
            cmd = [
                "libheif/heif-enc.exe",
                tmp_path,
                "-o", save_path,
                "-q", str(quality),
                "-t", str(thumb_w),
                "--matrix_coefficients", "1",
                "--colour_primaries", "1",
                "--transfer_characteristic", "13",
                "--full_range_flag", "1"
            ]
            print("執行命令:", cmd)
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.run(cmd, check=True,startupinfo=startupinfo)
            os.remove(tmp_path)
        elif fmt == "PNG":
            save_path = path + ".png"
            img.save(save_path, format="PNG")
        elif fmt == "JPG":
            # 預設使用 JPEG 格式
            save_path = path + ".jpg"
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(save_path, format="JPEG", quality=quality)

        if save_path is not None:
            if RawImg is not None:
                toMainGUI.put([0, f"[儲存圖片] 寫入EXIF"])
                with exiftool.ExifTool(encoding='utf-8') as et:
                    et.execute(
                        b"-overwrite_original",
                        b"-TagsFromFile",
                        RawImg.encode("utf-8"),
                        save_path.encode("utf-8")
                    )
            toMainGUI.put([0, f"[儲存圖片] 已保存: {save_path}"])
            return True
        else:
            toMainGUI.put([0, f"[儲存圖片] 失敗: {save_path}"])
            return False
    #except Exception as e:
    #    toMainGUI.put([0, f"[儲存圖片] 失敗\n{e}"])
    #    return False