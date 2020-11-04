import cv2
import imutils
import glob

#caminho = "/home/medeiros/Downloads/SolarPanelSoilingImageDataset/Solar_Panel_Soiling_Image_dataset/PanelImages/solar_Wed_Jun_21_16__43__8_2017_L_0.000775951704766_I_0.252694117647.jpg"
#caminho = "/home/medeiros/Downloads/SolarPanelSoilingImageDataset/Solar_Panel_Soiling_Image_dataset/PanelImages/solar_Wed_Jun_21_16__44__19_2017_L_0.00355660936574_I_0.251396078431.jpg"
salvar = "/home/medeiros/PFC/images_png64/"

images = glob.glob('/home/medeiros/Downloads/SolarPanelSoilingImageDataset/Solar_Panel_Soiling_Image_dataset/PanelImages/*.jpg', recursive=True)

for i,image in enumerate(images):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    resized = imutils.resize(img, width=64)
    cv2.imwrite(salvar+"image{}.png".format(i), resized)
