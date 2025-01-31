import cv2
import threading
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout, QLabel, QPushButton, QWidget,QApplication, QCheckBox
from model_n_judge_4 import Model_n_Judge 
from PyQt5.QtCore import Qt
import winsound
import torch
    
#

class MainWindow(QMainWindow):
    def __init__(self, model):
        super().__init__()
        
        self.running = False
        self.sound = False
        self.model_n_judge = Model_n_Judge(model)
        self.list = []
        
        # 인터페이스의 구성 요소들
        
        self.setWindowTitle('GUI Sample') # 윈도우 창 이름 설정 

        self.app = QApplication([])
        self.label = QLabel()
        self.easy_mode = QPushButton("Easy Mode")
        self.mid_mode = QPushButton("Mid Mode")
        self.hard_mode = QPushButton("Hard Mode")
        self.btn_go_to_capture_correct_pose = QPushButton("Capture your correct pose image")
        self.btn_now_capture = QPushButton("Capture now")
        self.btn_start = QPushButton("Camera On")
        self.btn_stop = QPushButton("Camera Off")
        self.checkbox = QCheckBox('Alarm On/Off')
        self.checkbox.move(20,20)
        self.checkbox.toggle()

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.label)
        self.vbox.addWidget(self.easy_mode)
        self.vbox.addWidget(self.mid_mode)
        self.vbox.addWidget(self.hard_mode)
        self.vbox.addWidget(self.btn_go_to_capture_correct_pose)
        self.vbox.addWidget(self.btn_now_capture)
        self.vbox.addWidget(self.btn_start)
        self.vbox.addWidget(self.btn_stop)
        self.vbox.addWidget(self.checkbox)

        self.widget = QWidget()
        self.widget.setLayout(self.vbox)
        
        self.setCentralWidget(self.widget)
        
        #
        
        self.btn_go_to_capture_correct_pose.setEnabled(True)
        self.btn_now_capture.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)

        # event에 대한 함수 연결 

        self.app.aboutToQuit.connect(self.onExit)
        self.easy_mode.clicked.connect(self.start)
        self.mid_mode.clicked.connect(self.start)
        self.hard_mode.clicked.connect(self.start)
        self.btn_go_to_capture_correct_pose.clicked.connect(self.go_to_capture_correct_pose)
        self.btn_now_capture.clicked.connect(self.now_capture)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.checkbox.stateChanged.connect(self.alarm_state)
        
    # 각각의 함수 정의

    def slot_toggle(self, state):
        self.setStyleSheet("background-color: %s" % ({True: "green", False: "red"}[state]))
        self.setText({True: "ON", False: "OFF"}[state])
        
    def go_to_capture_correct_pose(self):
        self.running = True
        
        th = threading.Thread(target=self.wait_capture)
        th.start()
        
        #
        
        self.btn_go_to_capture_correct_pose.setEnabled(False)
        self.btn_now_capture.setEnabled(True)
        
    def wait_capture(self):
        cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 높이
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 너비
        cap.set(cv2.CAP_PROP_FOURCC, 0x32595559) # 코텍
        cap.set(cv2.CAP_PROP_FPS, 20) # 프레임 조절
                
        self.resize_label(cap)
        
        if not (cap.isOpened()):
            print("Could not open video device")
        
        while self.running: 
            # time.sleep(3)
            
            ret, img2 = cap.read()
            
            if ret:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) 
                
                boxes, keypoints = self.model_n_judge.run_model(img2)
                
                print("Here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(keypoints.xyn.shape)
                if keypoints.xyn.shape == torch.Size([1, 10, 2]):
                    self.model_n_judge.correct_pose = keypoints
                    print("Generated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                
                self.show_bbox_and_keypoints(img2, boxes, keypoints)
                
                self.show_pixmap(img2)
            
            else:
                QMessageBox.about(self, "Error", "Cannot read frame.")
                print("cannot read frame.")
                
                break
            
        cap.release()
         
        print("Your current pose is successfully set to the correct pose.")
        print(self.model_n_judge.correct_pose.xyn)
        
    def now_capture(self):
        self.running = False
        
        print('captured..')
        
        #
        
        self.btn_go_to_capture_correct_pose.setEnabled(True)
        self.btn_now_capture.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    ########### start ####################3
    def start(self):
        self.running = True
        
        th = threading.Thread(target=self.run)
        th.start()
        
        print("started..")
        
        #
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)



    def run(self):
        cap = cv2.VideoCapture(0)
                
        self.resize_label(cap)
        
        if not (cap.isOpened()):
            print("Could not open video device")
        
        while self.running:
            ret, img = cap.read()
            
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                
                boxes, keypoints = self.model_n_judge.run_model(img)
                
                self.show_bbox_and_keypoints(img, boxes, keypoints)
    
                self.show_pixmap(img)
                
                # 자세 판단
                
                ############### 이 부분  ########################
                if self.easy_mode:
                    if keypoints.xyn.shape == torch.Size([1, 10, 2]): # pose estimation 결과가 적절하면 (한 개의 bounding box)
                        how_bad = self.model_n_judge.calculate(keypoints)
                        
                        self.set_easy_mode(how_bad)
                        print("easy_test...")


                elif self.mid_mode:
                    if keypoints.xyn.shape == torch.Size([1, 10, 2]): 
                        how_bad = self.model_n_judge.calculate(keypoints)
                        
                        self.set_mid_mode(how_bad)
                        print("mid_test...")
                        


                elif self.hard_mode:
                    if keypoints.xyn.shape == torch.Size([1, 10, 2]): 
                        how_bad = self.model_n_judge.calculate(keypoints)
                        
                        self.set_hard_mode(how_bad)
                        print("hard_test...")

            else:
                QMessageBox.about(self, "Error", "Cannot read frame.")
                print("cannot read frame.")
                
                break
            
        cap.release()
        
        print("Thread end.")

    def stop(self):
        self.running = False
        
        print("stopped..")
        
        #
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def onExit(self):
        print("exit")
        self.stop()

    def alarm_state(self, state):
        if state == Qt.Checked:
            QMessageBox.about(self,'Alarm On', "Alarm On !!")
            winsound.PlaySound("./GUI_Project/alarm.wav", winsound.SND_FILENAME)
            pass
            
        else:
            pass
            # self.setWindowTitle(' ')
            
    def resize_label(self, cap):
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.label.resize(round(width), round(height))

    def show_bbox_and_keypoints(self, image, boxes, keypoints):
        for x1y1x2y2 in boxes.xyxy:
            x1, y1, x2, y2 = x1y1x2y2
            [x1, y1, x2, y2] = [i.item() for i in [x1, y1, x2, y2]]
            
            (x1, y1, x2, y2) = map(round, (x1, y1, x2, y2))
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        for xy in keypoints.xy: # 각 세트마다
            for p in xy: # 각 포인트마다
                px = round(p[0].item())
                py = round(p[1].item())
                cv2.circle(image, (px, py), 3, (0, 255, 0), -1)  
                
    def show_pixmap(self, image):
        h,w,c = image.shape
        qImg = QtGui.QImage(image.data, w, h, w*c, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.label.setPixmap(pixmap)


    ################################## 함수 추가 부분 ##############################33
    def set_easy_mode(self, value):
        if 1.0 <= value <= 1.2:
            self.list.append(value)


        if len(self.list) >= 20:
            print("거북목 입니다 !")
            self.list.clear()



    def set_mid_mode(self, value):
        if 0.8 <= value <= 1.0:
            self.list.append(value)


        if len(self.list) >= 15:
            print("거북목 입니다 !")
            self.list.clear()


    def set_hard_mode(self, value):
        if 0.7 <= value <= 1.2:
            self.list.append(value)


        if len(self.list) >= 10:
            print("거북목 입니다 !")
            self.list.clear()


        


       




    




        
        


        
        
        



        