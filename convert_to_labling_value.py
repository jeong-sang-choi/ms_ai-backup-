import csv
from shutil import copyfile

# 1. "makesense_rect_and_kps" folder 만들기

# 2. MakeSense로 각 image에서 인물에 rectangle 치고, keypoints에 point 찍고 (순서 정확히 10개 찍어야 함), 
# rectangle은 YOLO로, point는 csv로 Export하기 

# 3. 위 결과값인 rectangle (folder), keypoints (csv) -> "makesense_rect_and_kps" folder에 넣어주기

# 4. 결과값을 YOLO format으로 통합하기
# : 모든 image가 images folder로 복사되며, 모든 image 각각의 rectangle과 keypoints 정보가 통합된 하나의 yolo가 labels folder에 담김

def result_to_yolo(ori_image_folder, rect_yolo_folder, kps_csv_path, output_image_folder, output_yolo_folder):
    
    # 우선 모든 image의 keypoints 담은 csv를 open함 

    with open(kps_csv_path, newline='', encoding='UTF-8') as kps_csv:
        
        kps_reader = csv.reader(kps_csv, delimiter=' ', quotechar='|')
        
        # prev_img_name 초기화
        
        prev_img_name = ""
        
        # 각 keypoint마다

        for kp in kps_reader: 
            
            # 현재 keypoint의 정보 추출
            
            kp = kp[0].split(",")
            
            kp_x, kp_y, img_name = float(kp[1]), float(kp[2]), kp[3].split(".")[0]
            
            # 현재 keypoint가 새로운 image에 속한다면 (if 현재 keypoint가 속한 image != 이전 keypoint가 속한 image) 
            
            if img_name != prev_img_name:
                
                # 새로운 image를 copy함: png 혹은 jpg
                
                try:
                        
                    copyfile(f"{ori_image_folder}/{img_name}.png", f"{output_image_folder}/{img_name}.png")
                        
                except:
                        
                    copyfile(f"{ori_image_folder}/{img_name}.jpg", f"{output_image_folder}/{img_name}.jpg")
                    
                # 새로운 image에 대응되는 rect_yolo와 output_yolo 파일 열기
                        
                rect_yolo_path = f"{rect_yolo_folder}/{img_name}.txt"
                output_yolo_path = f"{output_yolo_folder}/{img_name}.txt"
                
                rect_yolo = open(rect_yolo_path, 'r', encoding='UTF-8')
                output_yolo = open(output_yolo_path, 'a', encoding='UTF-8')
                
                # 새로운 rect_yolo의 정보 추출하여 새로운 output_yolo에 작성
                    
                rect = rect_yolo.readline().split(" ")
                
                rect_x, rect_y, rect_w, rect_h = round(float(rect[1]), 6), round(float(rect[2]), 6), round(float(rect[3]), 6), round(float(rect[4]), 6)
                
                output_yolo.write(f"0 {rect_x} {rect_y} {rect_w} {rect_h}")
                
                # 새로운 image의 w, h 추출
                
                img_w, img_h = float(kp[4]), float(kp[5])
                
                # prev_img_name 재설정
                
                prev_img_name = img_name
                
            # 현재 keypoint의 normalized x, y를 output_yolo에 작성
                
            kp_x_norm, kp_y_norm = round(kp_x / img_w, 6), round(kp_y / img_h, 6)
            
            output_yolo.write(f" {kp_x_norm} {kp_y_norm}") 

result_to_yolo(ori_image_folder="org_img", # 원래의 image folder
               rect_yolo_folder="makesense_rect_and_kps/rect_file", # rectangle yolo가 있는 folder
               kps_csv_path="makesense_rect_and_kps/keypoint.csv", # keypoints csv의 path
               output_image_folder="datasets/images/val", # 모든 image가 복사될 folder
               output_yolo_folder="datasets/labels/val") # 모든 yolo 출력이 담길 folder

