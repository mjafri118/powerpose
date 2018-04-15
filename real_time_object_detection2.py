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
		self.area_list_g = np.array(self.area_list_g)
		self.time_list_g = np.array(self.time_list_g)
		plt.plot(self.time_list_g, self.area_list_g, 'ro')
		test_time = np.array([0.11185193061828613, 0.7710678577423096, 0.8473117351531982, 0.9151899814605713, 0.9831826686859131, 1.0497028827667236, 1.1167547702789307, 1.1825079917907715, 1.2489347457885742, 1.3157999515533447, 1.3811838626861572, 1.4475467205047607, 1.5143578052520752, 1.5807507038116455, 1.6475656032562256, 1.7138137817382812, 1.7826149463653564, 1.850909948348999, 1.9160637855529785, 1.9838347434997559, 2.050286054611206, 2.121074914932251, 2.1870007514953613, 2.2526588439941406, 2.318483829498291, 2.3834967613220215, 2.447227716445923, 2.509294033050537, 2.571802854537964, 2.639021873474121, 2.7027640342712402, 2.767507791519165, 2.833077907562256, 2.8963608741760254, 2.9578208923339844, 3.0250799655914307, 3.0916378498077393, 3.157658815383911, 3.2218258380889893, 3.2974278926849365, 3.3860487937927246, 3.475496768951416, 3.55279278755188, 3.6332499980926514, 3.712634801864624, 3.793052911758423, 3.85459566116333, 3.922102928161621, 3.990288734436035, 4.055332660675049, 4.120389699935913, 4.184771776199341, 4.249773025512695, 4.3139870166778564, 4.377272605895996, 4.438794851303101, 4.505769968032837, 4.57016396522522, 4.635387897491455, 4.700315713882446, 4.7688586711883545, 4.835255861282349, 4.903131723403931, 4.968633651733398, 5.036087989807129, 5.10408878326416, 5.169422626495361, 5.237217903137207, 5.306164979934692, 5.37099289894104, 5.436870813369751, 5.50104284286499, 5.5722479820251465, 5.641594648361206, 5.7099609375, 5.775867700576782, 5.840990781784058, 5.91161584854126, 6.001928806304932, 6.074123859405518, 6.139208793640137, 6.205784797668457, 6.2728660106658936, 6.338212728500366, 6.403555631637573, 6.469053745269775, 6.539074897766113, 6.606305837631226, 6.673075914382935, 6.7444727420806885, 6.808649778366089, 6.872586965560913, 6.936152696609497, 7.005079030990601, 7.071382999420166, 7.13706374168396, 7.201919794082642, 7.277371883392334, 7.344583749771118, 7.409810781478882, 7.474670648574829, 7.5400660037994385, 7.6045098304748535, 7.670475959777832, 7.736814737319946, 7.803768634796143, 7.868291616439819, 7.932514905929565, 8.00222373008728, 8.067925930023193, 8.13113284111023, 8.194920778274536, 8.262458801269531, 8.444324731826782, 8.566378831863403, 8.630224704742432, 8.698303937911987, 8.76473593711853, 8.83032488822937, 8.898416757583618, 8.960525751113892, 9.029415845870972, 9.101062774658203, 9.164305686950684, 9.225965023040771, 9.295961856842041, 9.360223770141602, 9.427267074584961, 9.495770692825317, 9.561792850494385, 9.627869844436646, 9.694147825241089, 9.757414817810059, 9.827139616012573, 9.894629716873169, 9.96239185333252, 10.031437635421753, 10.096321821212769, 10.163268804550171, 10.231209754943848, 10.301522731781006, 10.37812876701355, 10.445665836334229, 10.511476993560791, 10.580993890762329, 10.647580862045288, 10.712413787841797, 10.777914762496948, 10.879143714904785, 11.04175877571106, 11.116204023361206, 11.186778783798218, 11.254578828811646, 11.322606801986694, 11.395437955856323, 11.468061923980713])



		test_area = np.array([64350, 64960, 65408, 65184, 64512, 64960, 64960, 65184, 65408, 65632, 65184, 65408,
 65785, 66304, 66600, 66375, 63450, 59175, 53058, 47008, 44460, 40906, 42262, 43130,
 42036, 40860, 41314, 41541, 41314, 40906, 41087, 41132, 41087, 41314, 41314, 41314,
 41995, 44265, 46575, 51959, 56448, 62048, 67050, 70200, 75151, 75374, 73590, 74036,
 73024, 74928, 74482, 75151, 74705, 74705, 72900, 68400, 65712, 64125, 60975, 56896,
 47925, 44070, 43456, 43904, 42560, 42525, 41625, 41175, 41132, 40725, 40768, 40809,
 40544, 40906, 40906, 40950, 40950, 40950, 40950, 40906, 40725, 40725, 40906, 40906,
 40906, 41132, 40906, 40680, 40275, 40002, 39150, 39200, 39424, 40725, 40544, 40768,
 40992, 40586, 40626, 41070, 41216, 41664, 42112, 46368, 54225, 59175, 64288, 64125,
 67348, 70336, 76489, 76043, 74036, 69216, 64288, 69750, 66150, 65475, 63450, 63000,
 63450, 63675, 65250, 62550, 63450, 64125, 65856, 63675, 61020, 55370, 52432, 50850,
 48600, 49050, 50400, 51754, 51754, 52884, 53336, 54240, 55144, 55842, 56296, 56500,
 56896, 57344, 57120, 62376, 63900, 69300, 73350, 74025, 75150, 75825, 76275, 76500])

		plt.axis([0, self.lastTime, 0, 100000])
		plt.plot(test_time, test_area , 'bo')
		plt.savefig('foo.png')
		#plt.plot(np.unique(self.time_list_g), np.poly1d(np.polyfit(self.time_list_g, self.area_list_g, 1))(np.unique(self.time_list_g)))
		plt.show()


PP = PowerPose()
PP.begin()
PP.sampleplot()
print(PP.time_list_g)
print(PP.area_list_g)
