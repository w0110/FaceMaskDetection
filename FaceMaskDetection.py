import dlib
import cv2
import imutils
import os.path
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
# import playsound

''' 建立中文字輸出 '''
def cv2_Chinese_Text(img,text,left,top,textColor,fontSize):    
    # 影像轉成 PIL影像格式
    if (isinstance(img,np.ndarray)):   
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)              # 建立PIL繪圖物件
    fontText = ImageFont.truetype(          # 建立字型 - 新細明體
                "C:\Windows\Fonts\mingliu.ttc",     # 新細明體
                fontSize,                   # 字型大小
                encoding="utf-8")           # 編碼方式
    draw.text((left,top),text,textColor,font=fontText)  # 繪製中文字
    # 將PIL影像格式轉成OpenCV影像格式
    return cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

while True:
    print("1:圖片讀取模式\n2:動態鏡頭讀取模式\n0:結束程式")

    mode = 0
    mode = input("請選擇模式：")
    print()
    if mode == '1':
        print("1.圖片讀取模式")
        
        name = input("請輸入檔案名稱(須包括副檔名 如.jpg .png .webp):")
        # 檢查檔案是否存在
        while 1:
            if not os.path.isfile(name):
                print("Wrong input!! 請重新輸入")
                name = input("請輸入檔案名稱(須包括副檔名 如.jpg .png .webp):")
            else:
                break
        
        # 讀取照片圖檔
        img = cv2.imread(name)

        near_threshold = 0.975

        # 縮小圖片
        img = imutils.resize(img, width=400)

        # Dlib 的人臉偵測器
        detector = dlib.get_frontal_face_detector()

        # 偵測人臉
        face_rects = detector(img, 0)

        # 取出所有偵測的結果
        for i, d in enumerate(face_rects):
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            # 以方框標示偵測的人臉
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

        # 抓人臉 切成三等份 上1/3、下2/3
        faceLong = y2-y1
        upface = img[y1:y1 + int(1 / 3 * faceLong), x1:x2]
        downface = img[y1 + int(1 / 3 * faceLong):y2, x1:x2]

        hist1 = cv2.calcHist([upface], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([downface], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

        # 平移縮放
        cv2.normalize(hist1, hist1, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1.0, cv2.NORM_MINMAX)

        near = cv2.compareHist(hist1, hist2, 0)
        # print(f'near = {near}')
        
        if near < near_threshold:
            # 畫框框與寫字
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)            
            cv2.putText(img, "Wear Mask", (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            img = cv2_Chinese_Text(img, "歡迎通行", x1-10, y1-50, (0, 255, 0), 30)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(img, "No Mask", (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            img = cv2_Chinese_Text(img, "禁止通行", x1, y1-50, (255, 0, 0), 30)

        cv2.imshow("Face Detection", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif mode == '2':
        n = 20
        print("2.動態鏡頭讀取模式")
        print("鏡頭開啟中...")
        
        detector = dlib.get_frontal_face_detector()

        color = ('b', 'g', 'r')
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
        
        # 無聊的小進度條 ㄎㄎ
        for i in range(n+1):
            print(f'\r[{"█"*i}{" "*(n-i)}] {i*100/n}%', end='')
            time.sleep(0.1)
            
        near_threshold = 0.01
        print("\n按'q'結束讀取\n")
        while cap.isOpened():
            ret, frame = cap.read()
            face_rects, scores, idx = detector.run(frame, 0)

            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()

                faceLong = y2 - y1
                upface = frame[y1:y1 + int(1 / 3 * faceLong), x1:x2]
                downface = frame[y1 + int(1 / 3 * faceLong):y2, x1:x2]

                hist1 = cv2.calcHist([upface], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
                hist2 = cv2.calcHist([downface], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

                # 平移縮放
                cv2.normalize(hist1, hist1, 0, 1.0, cv2.NORM_MINMAX)
                cv2.normalize(hist2, hist2, 0, 1.0, cv2.NORM_MINMAX)

                near = cv2.compareHist(hist1, hist2, 0)
                # print(near)
                
                if near < near_threshold:
                    # 框框 & 寫字
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, "Welcome", (x1+30, y1-50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, "Wear Mask", (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4, cv2.LINE_AA)
                    cv2.putText(frame, "NOT ALLOW", (x1+10, y1-50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, "No Mask", (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
                    # playsound.playsound('alert.mp3')
                    
            cv2.imshow("Face Detection", frame)

            # 按'q'關閉鏡頭
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    elif mode == '0':
        print("結束程式!")
        print("歡迎再次使用")
        break
    else:
        print("錯誤輸入，請重新輸入")
        continue