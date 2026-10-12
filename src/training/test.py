# import cv2
#
# img = cv2.imread("", cv2.IMREAD_UNCHANGED)
#
# print(img.shape)
# print(img.dtype)
#
#
# cv2.imshow("original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# for i in range(len(img)):
#     for j in range(len(img[i])):
#         if not img[i][j] == 0:
#             img[i][j] = 65535-img[i][j]
#
#
# cv2.imwrite("", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#
# print(img.shape)
# print(img.dtype)
#
# cv2.imshow("inverted", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

a = 100

b = -1

for i in range(a):
    if i%b==0 and i!=0 and b != -1:
        print(i)