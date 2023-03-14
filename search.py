# 导入所需的库
import cv2
import numpy as np
from deepface import DeepFace
import sqlite3
import os # 新增

# 创建数据库连接
conn = sqlite3.connect("face_db.db")
c = conn.cursor()

# 创建表格，如果不存在的话
c.execute("CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY, name TEXT, vector BLOB)")

# 定义人脸检测器和识别器
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_recognizer = DeepFace.build_model("Facenet")

# 定义阈值和摄像头编号
threshold = 0.4
camera_id = 0

# 创建摄像头对象，并设置分辨率为640x480像素
cap = cv2.VideoCapture(camera_id)
cap.set(3, 640)
cap.set(4, 480)

# 循环读取摄像头的每一帧图像
while True:
    # 读取一帧图像，并转换为灰度图像
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸并返回坐标和大小
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # 对每个检测到的人脸进行处理
    for (x,y,w,h) in faces:
        # 截取人脸区域并调整大小为160x160像素
        face_img = img[y:y+h,x:x+w]
        face_img = cv2.resize(face_img, (160, 160))

        # 将人脸图像转换为向量，并归一化为0-1之间的值
        face_vector = face_recognizer.predict(np.expand_dims(face_img, axis=0))[0]
        face_vector = (face_vector - face_vector.min()) / (face_vector.max() - face_vector.min())

        # 查询数据库中是否有相似的人脸向量，如果有则返回姓名，否则返回None
        c.execute("SELECT name FROM faces WHERE vector MATCH ?", (face_vector,))
        result = c.fetchone()
        if result:
            name = result[0]
        else:
            name = None

        # 在原始图像上绘制人脸框和姓名标签，并显示结果
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    
    # 显示摄像头画面，并等待用户按键输入    
    cv2.imshow("Camera",img)
    key = cv2.waitKey(1)

    # 如果用户按下Q键，则退出循环并关闭摄像头和窗口 
    if key == ord('q'):
        break
    
    # 如果用户按下U键，则弹出文件选择对话框，让用户选择要上传的照片文件 
    elif key == ord('u'):
        
        # 弹出文件选择对话框，并获取选择的文件路径 
        file_path = filedialog.askopenfilename(title="Select a photo", filetypes=[("Image files", "*.jpg *.png")])

        # 如果文件路径不为空，则读取图片并转换为灰度图像 
        if file_path:
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img,cv.COLOR_BGR2GRAY)

            # 检测人脸并返回坐标和大小 
            faces= face_detector.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)

            # 对每个检测到的人脸进行处理 
            for (x,y,w,h) in faces:

                # 截取人脸区域并调整大小为160x160像素 
                face_img=img[y:y+h,x:x+w]
                face_img=cv.resize(face_img,(160.160))

                # 将人脸图像转换为向量，并归一化为0-1之间的值
                  face_vector = face_recognizer.predict(np.expand_dims(face_img, axis=0))[0]
                face_vector = (face_vector - face_vector.min()) / (face_vector.max() - face_vector.min())

                # 查询数据库中是否有相似的人脸向量，如果有则返回姓名，否则返回None
                c.execute("SELECT name FROM faces WHERE vector MATCH ?", (face_vector,))
                result = c.fetchone()
                if result:
                    name = result[0]
                else:
                    name = None

                # 在原始图像上绘制人脸框和姓名标签，并显示结果
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            
            # 显示上传的图片，并等待用户按任意键关闭窗口 
            cv2.imshow("Photo",img)
            cv2.waitKey(0)
            cv2.destroyWindow("Photo")

# 关闭数据库连接和窗口    
conn.close()
cap.release()
cv2.destroyAllWindows()
