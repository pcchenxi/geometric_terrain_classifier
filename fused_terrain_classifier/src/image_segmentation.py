#!/usr/bin/env python
# ros images
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import rospy

import sys, os, os.path, time, cv2, numpy as np

import pickle
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import scipy as sp

############################################################## for image segmentation ####################################
path = '/home/xi/workspace/catkin_rcv/src/train_from_label/label_related' #os.getcwd()

sys.path.append(path + '/segmentation/')
sys.path.append(path + '/segmentation/datasets/')
sys.path.append(path + '/segmentation/models')
sys.path.append(path + '/segmentation/notebooks')
# print (sys.path)
print(path)
import layers
import fcn8s
import util
import cityscapes
from colorize import colorize
from class_mean_iou import class_mean_iou

feature_vision = np.zeros( [1, 256, 512, 34], dtype=np.float32 )
seg_pub = []
prediction_label = []
bridge = CvBridge()
clf = RandomForestClassifier(max_depth=10, n_estimators=5)
feature_norms = []
segment_ready = False

sess = tf.InteractiveSession()
image_shape = [1, 256, 512, 3]
image_op = tf.placeholder(tf.float32, shape=image_shape)

logits_op = fcn8s.inference(image_op)
predictions_op = layers.predictions(logits_op)
predictions_op_prob = tf.nn.softmax(logits_op)

init_op = tf.global_variables_initializer()
sess.run(init_op)

saver = tf.train.Saver()
# saver.restore(sess, checkpoint)
saver.restore(sess, path + '/tf_models/fcn8s_augment_finetune/' + 'fcn8s_augment.checkpoint-30')

prediction_publisher = rospy.Publisher('/prediction_color', Image, queue_size=1)
# pub = rospy.Publisher('prediction_prob', numpy_msg(Floats),queue_size=10)

def callback_prediction(message):
    global segment_ready, feature_vision
    np_arr = np.fromstring(message.data, np.uint8)         
    image = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)    
    # image = bridge.imgmsg_to_cv2(message)

    image = sp.misc.imresize(image, image_shape[1:], interp='bilinear')

    image = image[..., ::-1] # bgr to rgb
    image = (image - image.mean()) / image.std()
    
    feed_dict = {image_op: image[np.newaxis, ...]}
    
    prediction_label = sess.run(predictions_op, feed_dict=feed_dict)
    feature_vision = sess.run(predictions_op_prob, feed_dict=feed_dict)

#     pickle.dump(prediction_prob, open("/home/xi/workspace/labels/prob.p", "wb"))
    prediction_label = colorize(prediction_label, cityscapes.augmented_labels)
    # cv2.imshow("prediction_label", prediction_label)
    # cv2.waitKey(10)

    # prediction_label = prediction_label[..., ::-1] # rgb to bgr
    # prediction_publisher.publish(bridge.cv2_to_imgmsg(prediction_label))
    segment_ready = True

def normolize_data(data, feature_norms):
    # print len(data[0]), feature_norms
    for i in range(len(data[0])):
        feature = np.zeros( [len(data), 1], dtype=np.float32 )
        mean = feature_norms[i][0]
        std = feature_norms[i][1]
        for j in range(len(data)):
            data[j][i] = (data[j][i] - mean)/std

    return data

def normolize_dataset(x):
    feature_norms = []
    for i in range(len(x[0])):
        feature = np.zeros( [len(x), 1], dtype=np.float32 )
        for j in range(len(x)):
            # if i == len(x[j]) or j == len(x):
            #     print 'error', i, len(x[j], j, len(x)
            feature[j] = x[j][i]

        norm = []
        norm.append(np.mean(feature))
        norm.append(np.std(feature))
        feature_norms.append(norm)
    
    return feature_norms

def classify_feature(feature_img):
    global clf, feature_norms, segment_ready, feature_vision, prediction_label
    img_predit = np.zeros([1080,1920], dtype=np.uint8)
    img_check = np.zeros([256,512], dtype=np.uint8)
    rows = []
    cols = []
    x_test = []
    if segment_ready == False:
        return

    for row in xrange(1080):
        for col in xrange(1920):
            x_new = []
            r = float (feature_img[row, col, 3])
            if r == 0:
                continue                
            hd = float (feature_img[row, col, 0])
            slope = float (feature_img[row, col, 1])
            roughness = float (feature_img[row, col, 2])        

            row_seg = int(row*256.0/1080.0)
            col_seg = int(col*512.0/1920.0)

            check = img_check[row_seg, col_seg]
            if check == 1:
                continue
            else:
                img_check[row_seg, col_seg] = 1

            feature_v = []
            max_prob = 0
            max_index = 0
            for v_index in xrange(34):
                f_v = feature_vision[-1][row_seg][col_seg][v_index]
                if f_v > max_prob:
                    max_prob = f_v
                    max_index = v_index
                feature_v.append(f_v)

            x_new.append(hd)
            x_new.append(slope)
            x_new.append(roughness)
            x_new.extend(feature_v)  # vision feature
            x_test.append(x_new)

            rows.append(row)
            cols.append(col)


    print('finish reading feature', time.time())
    x_test = normolize_data(x_test, feature_norms)
    predict = clf.predict(x_test)
    x_test = []
    for i in xrange(len(predict)):
        label = predict[i]
        row = rows[i]
        col = cols[i]
        radius = 15.0/540.0 * row
        img_predit[row, col] = label*50
        cv2.circle(img_predit, (col, row), int(radius), label*50, -1)

    image_message = bridge.cv2_to_imgmsg(img_predit, encoding='8UC1')
    seg_pub.publish(image_message)

def callback(data):
    print('get geometric data', time.time())
    try:
        feature_img = bridge.imgmsg_to_cv2(data)
        print (feature_img.dtype, feature_img.shape)
        classify_feature(feature_img)
        # print "call back"
    except CvBridgeError as e:
        print(e)
    print(time.time())

def get_classifier():
    global clf, feature_norms
    clf = joblib.load(path + '/model_all.pkl') 
    feature_norms = joblib.load(path + '/norm_all.pkl') 
    print (feature_norms)

def main(args):
    global seg_pub
    rospy.init_node('terrain_classifier', anonymous=True)
    image_sub = rospy.Subscriber("/gemoetric_features", Image, callback)
    subscriber = rospy.Subscriber('/kinect2/hd/image_color_rect/compressed', CompressedImage, callback=callback_prediction, queue_size=1, buff_size=52428800 * 2)

    seg_pub = rospy.Publisher('final_segmentation', Image, queue_size=1)

    get_classifier()
    rospy.spin()


if __name__ == '__main__':
	main(sys.argv)
