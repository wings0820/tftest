import matplotlib.pyplot as plt;
import tensorflow as tf;

image_raw_data_jpg = tf.gfile.FastGFile('/home/mmc/test.jpg', 'r').read()


with tf.Session() as sess:
    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg) #图像解码
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8) #改变图像数据的类型



    plt.figure(1) #图像显示
    plt.imshow(img_data_jpg.eval())

    plt.show()
