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

a = "[[-0.1,0.1],[-0.2,0.2],[-0.15,]]"
a = a.replace("[", "")
a = a.replace("]", "")

a = a.split(",")

if not len(a) == 6:
    print("ERROR WITH ARRAY")
    quit()

b = [[float(a[0]), float(a[1])], [float(a[2]), float(a[3])], [float(a[4]), float(a[5])]]


print(b)