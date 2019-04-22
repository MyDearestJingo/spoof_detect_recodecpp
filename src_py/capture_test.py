import time
import sys
import pathlib
import logging
import cv2
from tools.detect import MtcnnDetector
from sklearn.externals import joblib
from classifier.features import *
import argparse as ap
import time

# 控制台与日志输出
logger = logging.getLogger("app")
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
console_handler.formatter = formatter  # 也可以直接给formatter赋值

def crop_images(img, bboxs):
    num_face = bboxs.shape[0]
    h, w = img.shape[:2]
    croped_bboxs = []
    cropped = []
    for i in range(num_face):
        b_w = bboxs[i, 2] - bboxs[i, 0]
        b_h = bboxs[i, 3] - bboxs[i, 1]
        l_x = max(bboxs[i, 0] - 0.4 * b_w, 0)
        l_y = max(bboxs[i, 1] - 0.4 * b_h, 0)
        r_x = min(bboxs[i, 2] + 0.4 * b_w, w)
        # r_y = min(bboxs[i, 3] + 0.4 * b_h, h)
        r_y = bboxs[i, 3]
        cb_0 = (0.4 * b_w) if bboxs[i, 0] > (0.4 * b_w)  else bboxs[i, 0]
        cb_1 = (0.4 * b_h) if bboxs[i, 1] > (0.4 * b_h)  else bboxs[i, 1]
        cb_2 = (1.4 * b_w) if bboxs[i, 0] > (0.4 * b_w)  else bboxs[i, 0] + b_w
        cb_3 = cb_2 + b_h
        cropped.append(img[int(l_y):int(r_y), int(l_x):int(r_x)])
        croped_bboxs.append(int(cb_0))
        croped_bboxs.append(int(cb_1))
        croped_bboxs.append(int(cb_2))
        croped_bboxs.append(int(cb_3))
    return cropped , croped_bboxs

def save_capture(img , real, pred, label):
    now = time.time()
    img_name = str(now)+ '.jpg'
    if label == True:
        path="E:/Projects/face-spoof-detect190321/capture/pred_real/"
        if real == False:
            path = "E:/Projects/face-spoof-detect190321/capture/pred_fake/"
    else:
        path="E:/Projects/face-spoof-detect190321/capture/unlabeled_real/"
        if real == False:
            path = "E:/Projects/face-spoof-detect190321/capture/unlabeled_fake/"
    cv2.imwrite(path + img_name , img)

def spoof_detect(face, crop_bboxs,model_path):
    img = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_hist, nonface_hist = get_context_feature(img, crop_bboxs)
    # mu, sigma, lighting_feature = get_lighting_feature(img, crop_bboxs)
    lbp, lbp_hist = get_lbp_feature(img)
    fd = np.concatenate((face_hist, nonface_hist, lbp_hist), axis=0) # 三个特征向量整合
    feature = fd.reshape(1, -1)
    clf = joblib.load(model_path) # clf: classifier
    pred = clf.predict(feature)
    score= clf.decision_function(feature)
    return pred, score

if __name__ == '__main__':
    # 下面四行做何用？
    parser = ap.ArgumentParser()
    parser.add_argument('-m', "--model_name", help="model name", default="re_train_model_V4.m")
    args = vars(parser.parse_args())
    model = args["model_name"]

    # MTCNN人脸检测器初始化
    mtcnn_detector = MtcnnDetector(min_face_size=24, use_cuda=False) # mtcnn_detector的数据格式需要解析
    logger.info("Init the MtcnnDetector.")
    project_root = pathlib.Path()
    model_path = project_root / "model" / str(model)

    # 从摄像头读取图片，目前已不需要
    cap= cv2.VideoCapture(0)
    cap.set(3, 640)  # 设置分辨率
    cap.set(4, 480)
    cap.set(15,-6.0)
    while True:
        ret,frame= cap.read(0) # 帧提取，ret无用
        start = time.time()
        bboxs, landmarks = mtcnn_detector.detect_face(frame) # landmarks无用，bbox人脸区域坐标矩阵
        detect_time= time.time()- start
        if bboxs.shape[0] > 0:
            face , crop_bboxs = crop_images(frame, bboxs) # ？按照bbox裁切原图
            # cv2.putText(frame, str("%.3f"%(detect_time)) + 's', (0,20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            for face_index in range(bboxs.shape[0]):
                time_start = time.time()
                pred, score= spoof_detect(face[face_index], crop_bboxs, model_path) # 同一帧里可能不只一张人脸
                # img = cv2.cvtColor(face[face_index], cv2.COLOR_BGR2GRAY)
                # face_hist, nonface_hist = get_context_feature(img , crop_bboxs)
                # mu, sigma, lighting_feature = get_lighting_feature(img , crop_bboxs)
                # lbp, lbp_hist = get_lbp_feature(img)
                # fd = np.concatenate((face_hist, nonface_hist, lighting_feature, lbp_hist), axis=0)
                # feature = fd.reshape(1, -1)
                # pred = spoof_classifier.predict(feature)
                # score = spoof_classifier.decision_function(feature)
                spoof_detect_time = ("%.3f" % (time.time() - time_start))
                display_start= time.time()
                if pred == 1:
                    save_capture(frame, True , pred, label= False)
                    cv2.rectangle(frame, (int(bboxs[face_index][0]), int(bboxs[face_index][1])),
                                  (int(bboxs[face_index][2]), int(bboxs[face_index][3])), (0, 255, 0))
                    cv2.putText(frame, str(spoof_detect_time)+ 's', (int(bboxs[face_index][0]), int(bboxs[face_index][1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    save_capture(frame, True , pred, label= True)
                if pred == -1:
                    save_capture(frame, False, pred, label=False)
                    cv2.rectangle(frame, (int(bboxs[face_index][0]), int(bboxs[face_index][1])),
                                  (int(bboxs[face_index][2]), int(bboxs[face_index][3])), (0, 0, 255))
                    cv2.putText(frame, str(spoof_detect_time)+ 's', (int(bboxs[face_index][0]), int(bboxs[face_index][1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    save_capture(frame, False, pred, label=True)
        cv2.imshow('test',frame)
        cv2.waitKey(1)
    print("Detect Finished!")