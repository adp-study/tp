import cv2
import numpy as np

input_path = 'E:/Dataset/Disc_png/Dataset/Anonymized_0006/0006.png'
label_path = 'E:/Dataset/Disc_png/Label/Anonymized_0006/0006.png'

test_image = cv2.imread(input_path)
label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

#test_image = np.expand_dims(test_image, axis=0)
#test_image = np.expand_dims(test_image, axis=3)
#_, pred_image = cv2.threshold(label, 0.5, 1.0, cv2.THRESH_BINARY)

pred_image = label

pred_image = np.expand_dims(pred_image, axis=0)
pred_image = np.expand_dims(pred_image, axis=3)

G = np.zeros([1, 448, 448, 1])
B = np.zeros([1, 448, 448, 1])
R = pred_image
#
pred_image = np.concatenate((B, G, R), axis=3)
#
pred_image = np.squeeze(pred_image)
print(pred_image.shape)
print(test_image.shape)
# tR = test_image
# tG = test_image
# tB = test_image
# print(tR.shape)
# test_image = np.concatenate((tB, tG, tR), axis=3)
# test_image = np.squeeze(test_image)
test_image = test_image.astype(float)
# pred_image = pred_image * 255
# cv2.imwrite(pred_test_img_fullpath, pred_image)
#
# # 원본이미지에 예측결과를 마스킹해줍니다. 마스킹 비율을 결정하는 파라메터가 w이고 각 이미지의 적용비율은 p로 결정합니다.
# # w와 p를 바꿔가면서 저장하며 가시성 좋은 값을 찾으면 됩니다.
w = 40
p = 0.0001
result = cv2.addWeighted(pred_image, float(100 - w) * p, test_image, float(w) * p, 0) * 255
print(np.max(result))
# cv2.imshow('res', result)
# cv2.waitKey()
#
cv2.imwrite('E:/6.png', result)