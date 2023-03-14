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

# 定义目录路径和阈值
dir_path = "photos/"
threshold = 0.4

# 定义已识别和未成功文件夹的路径 # 新增
recognized_path = "recognized/"
unsuccessful_path = "unsuccessful/"

# 遍历目录下的所有图片文件
for file in os.listdir(dir_path):
    # 读取图片并转换为灰度图像
    img = cv2.imread(dir_path + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸并返回坐标和大小
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # 判断是否有人脸被检测到 # 新增
    if len(faces) > 0:
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

            # 如果没有找到相似的人脸向量，则询问用户输入姓名，并将新的人脸向量和姓名插入数据库中
            if not name:
                name = input(f"Please enter the name of the person in {file}: ")
                c.execute("INSERT INTO faces (name, vector) VALUES (?, ?)", (name, face_vector))
                conn.commit()

            # 在原始图像上绘制人脸框和姓名标签，并显示结果
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        
        cv2.imshow(file,img)
        cv2.waitKey(0)

        # 将文件移动到已识别文件夹中 # 新增 
        os.rename(dir_path + file, recognized_path + file)

    else:
        # 将文件移动到未成功文件夹中 # 新增 
        os.rename(dir_path + file, unsuccessful_path + file)

# 关闭数据库连接和窗口    
conn.close()
cv2.destroyAllWindows()
