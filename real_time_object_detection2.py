# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
#import argparse
#import imutils
import time, cv2, imutils, argparse
#import cv2
import matplotlib.pyplot as plt

class PowerPose:
	def __init__(self):

		# construct the argument parse and parse the arguments
		ap = argparse.ArgumentParser()
		ap.add_argument("-p", "--prototxt", required=True,
			help="path to Caffe 'deploy' prototxt file")
		ap.add_argument("-m", "--model", required=True,
			help="path to Caffe pre-trained model")
		ap.add_argument("-c", "--confidence", type=float, default=0.2,
			help="minimum probability to filter weak detections")
		self.args = vars(ap.parse_args())

		# initialize the list of class labels MobileNet SSD was trained to
		# detect, then generate a set of bounding box self.COLORS for each class
		self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]

		self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

		# load our serialized model from disk
		print("[INFO] loading model...")
		self.net = cv2.dnn.readNetFromCaffe(self.args["prototxt"], self.args["model"])

		# initialize the video stream, allow the cammera sensor to warmup,
		# and initialize the FPS counter
		print("[INFO] starting video stream...")
		self.vs = VideoStream(src=0).start()
		time.sleep(2.0)
		self.fps = FPS().start()

		#Begin a timer
		self.t0 = time.time()

		self.area_list_t = []
		self.time_list_t = []
		self.area_list_g = []
		self.time_list_g = []

		self.lastTime = 30

	def begin(self):

		# loop over the frames from the video stream
		while True:
			# grab the frame from the threaded video stream and resize it
			# to have a maximum width of 400 pixels
			frame = self.vs.read()
			frame = imutils.resize(frame, width=400)

			# grab the frame dimensions and convert it to a blob
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
				0.007843, (300, 300), 127.5)

			# pass the blob through the network and obtain the detections and
			# predictions
			self.net.setInput(blob)
			detections = self.net.forward()

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with
				# the prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by ensuring the `confidence` is
				# greater than the minimum confidence
				if confidence > self.args["confidence"]:
					# extract the index of the class label from the
					# `detections`, then compute the (x, y)-coordinates of
					# the bounding box for the object
					idx = int(detections[0, 0, i, 1])
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					area = (startX-endX)*(startY-endY)
					#SHOW BOUNDS if person
					if self.CLASSES[idx] == "person":
						#print(self.CLASSES[idx] + ": " + str(area)
						PowerPose.storeGraphTime(self, area, time.time())
						PowerPose.storeTableTime(self, area, time.time())

					# draw the prediction on the frame
					# label = "{}: {:.2f}%".format(self.CLASSES[idx],
					# 	confidence * 100)

					# REVISED: draw the prediction on the frame
					label = "{}".format(self.CLASSES[idx])
					cv2.rectangle(frame, (startX, startY), (endX, endY),
						self.COLORS[idx], 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

			# show the output frame
			#if self.CLASSES[idx] == "person":
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF



			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				self.lastTime = time.time() - self.t0
				break

			# update the FPS counter
			self.fps.update()

		# stop the timer and display FPS information
		self.fps.stop()
		print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

		# do a bit of cleanup
		cv2.destroyAllWindows()
		self.vs.stop()

	def storeGraphTime(self, area, time):
		time = time - self.t0
		# if time%1 <= 0.05:
			# NEED -- account for threshold value
		print(str(area) +  " @ " +  str(time))
		self.area_list_g.append(area)
		self.time_list_g.append(time)

	def storeTableTime(self, area, time):
		time = time - self.t0
		if time%1 <= 0.05:
			# NEED -- account for threshold value
			print(str(area) +  " @ " +  str(time))
			self.area_list_t.append(area)
			self.time_list_t.append(time)

	def sampleplot(self):

		plt.plot(self.time_list_g, self.area_list_g, 'ro')
		plt.axis([0, self.lastTime, 0, 100000])
		#plt.plot(np.unique(self.time_list_g), np.poly1d(np.polyfit(self.time_list_g, self.area_list_g, 1))(np.unique(self.time_list_g)))
		plt.show()


PP = PowerPose()
PP.begin()
PP.sampleplot()
print(PP.area_list_t)
print(PP.area_list_g)
