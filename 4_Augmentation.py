import imgaug as ia
import cv2
import imageio
from imgaug import augmenters as iaa

image = imageio.v2.imread("Data_Augmentation/5_original.png")
image = cv2.resize(image, (256, 256))

ia.seed(8)

rotate = iaa.Affine(rotate=(-25, 25))
image_aug = rotate(image=image)
status1 = cv2.imwrite('Data_Augmentation/5_aug_1.png', image_aug)

images = [image, image, image, image, image, image, image, image,
          image, image, image, image, image, image, image, image]
images_aug = rotate(images=images)
status2 = cv2.imwrite('Data_Augmentation/5_aug_2.png', ia.draw_grid(images_aug, cols=4, rows=4))

seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.AdditiveGaussianNoise(scale=(10, 60)),
    iaa.Crop(percent=(0, 0.2))])
images_aug = seq(images=images)
status3 = cv2.imwrite('Data_Augmentation/5_aug_3.png', ia.draw_grid(images_aug, cols=4, rows=4))

seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.AdditiveGaussianNoise(scale=(30, 90)),
    iaa.Crop(percent=(0, 0.4))], random_order=True)
images_aug = [seq(image=image) for _ in range(16)]
status4 = cv2.imwrite('Data_Augmentation/5_aug_4.png', ia.draw_grid(images_aug, cols=4, rows=4))

print("\n Image written to file-system : ", (status1, status2, status3, status4))
