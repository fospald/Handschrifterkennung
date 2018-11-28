# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:05:24 2018

@author: Theresa
"""

import sys, random, os
from PyQt5.QtWidgets import (QApplication, QWidget, QDialog,
		QPushButton, QGridLayout, QHBoxLayout, QVBoxLayout, QFrame, QLabel, QTextBrowser)
from PyQt5.QtGui import (QPainter, QPen, QPainterPath, QImage, QTransform, QFont, QColor, QKeySequence)
from PyQt5.QtCore import (pyqtSlot, pyqtSignal, Qt, QTimer, QRect)
import numpy as np

from TensorFlowMNIST_Classifier import TensorFlowMNIST_Classifier
	

class NumberDrawWidget(QFrame):

	numberChanged = pyqtSignal()
	
	def __init__(self):
		self.recreatePath = True
		self.path = None
		self.timer = QTimer()
		self.timer.timeout.connect(self.timeout)
		super().__init__()
		self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
		self.setCursor(Qt.CrossCursor)

		font = QFont(self.font())
		font.setPixelSize(self.width()*0.25)
		self.path = QPainterPath()
		#self.path.addText(0.5*self.width(), 1.2*self.height(), font, "4")

	def timeout(self):
		self.recreatePath = True
		self.numberChanged.emit()
		self.timer.stop()
		self.update()

	def resetTimer(self):
		self.timer.start(1000)
		
	#mousePressEvent: Mausklick
	def mousePressEvent(self, e):
		if self.recreatePath:
			self.recreatePath = False
			self.path = QPainterPath()
		self.path.moveTo(e.pos())
		self.update()
		self.timer.stop()

	#mouseMoveEvent: Mausbewegung
	def mouseMoveEvent(self, e):

		if int(e.buttons()) == 0:
			return
		self.path.lineTo(e.pos())
		self.update()
	
	def mouseReleaseEvent(self, e):	
		self.resetTimer()

	#paintEvent: es wird gezeichnet
	def paintEvent(self, e):
		
		super().paintEvent(e)
		
		qp = QPainter()
		qp.begin(self)
		
		fr = self.frameRect()
		fr.adjust(1, 1, -1, -1)
		qp.fillRect(fr, QColor(255, 255, 255))

		def bzsez():
			g = 200
			pen = QPen(QColor(255, 0, 0, 255-g))
			qp.setPen(pen)
			
			font = QFont(qp.font())
			font.setPixelSize(self.width()/10)
			qp.setFont(font)
			qp.drawText(fr, Qt.AlignCenter, "Bitte zeichnen Sie\neine Ziffer von 0 bis 9!")

		bzsez()

		if not self.path is None:
			if self.recreatePath:
				pen = QPen(Qt.gray)
			else:
				pen = QPen(Qt.black)
			pen.setWidth(self.width()/32.0)
			pen.setJoinStyle(Qt.RoundJoin)
			pen.setCapStyle(Qt.RoundCap)
			qp.setPen(pen)
			qp.drawPath(self.path)

		if self.recreatePath:
			bzsez()

		qp.end()


class ResizeFrame(QFrame):

	def __init__(self):
		super().__init__()
		self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
		self.resize = True

	def resizeEvent(self, e):
		w = int(1.0*self.height())
		if self.resize:
			self.setMaximumWidth(w)
			self.setMinimumWidth(w)


class NumberVectorWidget(ResizeFrame):

	def __init__(self, image_shape=(16, 16)):
		super().__init__()
		self.image = QImage(*image_shape, QImage.Format_Grayscale8)
		self.image.fill(0xff)
		self.border = 5.00
		
	def updateImage(self, path):
		
		self.image.fill(0xff)
		
		br = path.boundingRect()
		br.adjust(-1, -1, 1, 1)
		
		if br.width()*br.height() == 0:
			self.update()
			return
		
		sx = self.image.width()/float(br.width())
		sy = self.image.height()/float(br.height())
		s = min(sx, sy)

		t = QTransform()

		t.translate(self.border, self.border)
		t.scale(1.0 - 2*self.border/self.image.width(), 1.0 - 2*self.border/self.image.height())
		t.translate(0.5*(sx-s)*br.width(), 0.5*(sy-s)*br.height())

		t.scale(s, s)

		t.translate(-br.x(), -br.y())


		qp = QPainter()
		
		qp.begin(self.image)
		qp.setRenderHint(QPainter.Antialiasing, True)
		
		pen = QPen(Qt.black)
		pen.setWidth(5.0/min(sx, sy))
		pen.setJoinStyle(Qt.RoundJoin)
		pen.setCapStyle(Qt.RoundCap)
		qp.setTransform(t)
		qp.setPen(pen)
		
		qp.drawPath(path)
		
		qp.end()
		
		self.update()
		

	#paintEvent: es wird gezeichnet
	def paintEvent(self, e):
		
		super().paintEvent(e)
		
		qp = QPainter()
		qp.begin(self)
		
		fr = self.frameRect()
		fr.adjust(1, 1, -1, -1)
		qp.drawImage(QRect(fr.x(), fr.y(), fr.width(), fr.height()), self.image, QRect(0, 0, self.image.width(), self.image.height()))
		
		qp.setCompositionMode(QPainter.RasterOp_SourceXorDestination);
		qp.setPen(QColor(0xff, 0xff, 0xff));

		for i in range(1, self.image.width()+1):
			x = fr.x() + i*fr.width()/self.image.width()
			qp.drawLine(x, fr.y(), x, fr.y() + fr.height())
		for i in range(1, self.image.height()+1):
			y = fr.y() + i*fr.height()/self.image.height()
			qp.drawLine(fr.x(), y, fr.x() + fr.width(), y)


		qp.end()



class DetectedNumberWidget(ResizeFrame):

	def __init__(self):
		super().__init__()
		self.number = None

	def paintEvent(self, e):
		
		super().paintEvent(e)

		qp = QPainter()
		qp.begin(self)
		
		font = QFont(qp.font())
		font.setPixelSize(self.width()*1.0)
		qp.setFont(font)

		#pen = QPen(Qt.black)
		#pen.setWidth(10)
		#qp.setPen(pen)
		
		fr = self.frameRect()
		fr.adjust(1, 1, -1, -1)

		#qp.fillRect(fr, QColor(255, 255, 255))

		if self.number is None:
			number = "-"
		else:
			number = str(self.number)

		qp.drawText(fr, Qt.AlignCenter, number)
		
		qp.end()
 

class DetectedProbabilitiesWidget(ResizeFrame):

	def __init__(self):
		super().__init__()
		self.probabilities = None
		self.resize = False

	def paintEvent(self, e):
		
		super().paintEvent(e)
		
		if self.probabilities is None:
			return

		w = self.width()
		h = self.height()
		pmax = max(self.probabilities)

		qp = QPainter()
		qp.begin(self)
		#qp.setRenderHint(QPainter.Antialiasing, True)

		#pen = QPen(Qt.black)
		#qp.setPen(pen)

		s = 20

		#qp.fillRect(1, 1, w-2, h-s, QColor(255, 255, 255))
		#qp.fillRect(1, 1, w, h, QColor(255, 255, 255))


		for i in range(10):
			r = w/15
			x = (i+0.5)*w/11 + 0.5*((w/10)-r)
			y = self.probabilities[i]/pmax*(h - 2*s)

			qp.drawLine(x+0.5*r, h-s-3, x+0.5*r, h-s)
			p = int(round(self.probabilities[i]*100))
			if p > 0:
				qp.drawText(x-r, h-y - 2*s, 3*r, s, Qt.AlignCenter, str(p))
				color = QColor(0, 200, 0, 127) if self.probabilities[i] == pmax else QColor(255, 0, 0, 127)
				qp.fillRect(x, h-y-s, r, y+1, color)
			qp.drawText(x-r+1, h-s, 3*r, s, Qt.AlignCenter, str(i))

		qp.drawLine(3, h-s, w-4, h-s)

		qp.end()

class Title(QLabel):

	def __init__(self, title):
		
		super().__init__(title)

		font = QFont(self.font())
		font.setBold(True)
		font.setPixelSize(15)
		self.setFont(font)

		self.setMaximumHeight(22)
		

class HelpWidget(QWidget):
	
	def __init__(self, parent):
		
		super().__init__(parent, Qt.Window)
	
		self.setMinimumWidth(900)
		self.setMinimumHeight(700)
		self.setWindowTitle(parent.windowTitle() + ' - Hilfe')

		with open("help.html", "rt") as f:
			html = f.read()

		b = QTextBrowser()
		b.setOpenExternalLinks(True)
		b.setHtml(html)

		layout = QVBoxLayout()
		layout.addWidget(b)

		self.setLayout(layout)



class MainWidget(QWidget):
	
	def __init__(self):
		
		super().__init__()
		
		train_dir = "mnist_convnet_model"
		self.classifier = TensorFlowMNIST_Classifier(train_dir)
		if not os.path.isdir(train_dir):
			self.classifier.train(20000)

		self.ndw = NumberDrawWidget()
		self.nvw = NumberVectorWidget(self.classifier.image_shape())
		self.dnw = DetectedNumberWidget()
		self.dpw = DetectedProbabilitiesWidget()
		
		self.ndw.numberChanged.connect(self.updateNumber)
		
		self.fullscreenButton = QPushButton("Vollbild")
		self.fullscreenButton.setCheckable(True)
		self.fullscreenButton.toggled.connect(self.fullscreenChanged)

		self.helpButton = QPushButton("Hilfe")
		self.helpButton.clicked.connect(self.showHelp)
		self.helpButton.setShortcut(QKeySequence.HelpContents);

		layout = QGridLayout()

		vbox = QVBoxLayout()
		hbox = QHBoxLayout()
		hbox.addWidget(Title("Benutzereingabe"))
		hbox.addStretch()
		hbox.addWidget(self.helpButton)
		hbox.addWidget(self.fullscreenButton)
		vbox.addLayout(hbox)
		vbox.addWidget(self.ndw)
		layout.addLayout(vbox, 0, 0)

		vbox = QVBoxLayout()
		vbox.addWidget(Title("Gerasterte Ziffer"))
		vbox.addWidget(self.nvw)
		vbox.addWidget(Title("P(Ziffer) (%)"))
		vbox.addWidget(self.dpw)
		vbox.addWidget(Title("Erkannte Ziffer"))
		vbox.addWidget(self.dnw)
		layout.addLayout(vbox, 0, 1)

		layout.addLayout(vbox, 0, 1)
		#layout.setColumnStretch(0, 2)
		#layout.setColumnStretch(1, 1)
		self.setLayout(layout)
		
		self.setMinimumWidth(900)
		self.setMinimumHeight(700)
		self.setWindowTitle('Handschrifterkennung')

		
		self.updateNumber()

		self.show()
	
	def showHelp(self, checked):
		hw = HelpWidget(self)
		hw.show()

	def fullscreenChanged(self, checked):
		self.setWindowState(self.windowState() ^ Qt.WindowFullScreen)
		self.show()

	"""
	def detectNumberStub(self, image):

		img = np.zeros([self.nvw.image.width(), self.nvw.image.height()])
		for i in range(self.nvw.image.height()):
			for j in range(self.nvw.image.width()):
				img[i,j] = (image.pixel(i, j) & 0xff)/255.0
		img /= np.sum(img)

		cx = (self.nvw.image.width()-1)*0.5
		cy = (self.nvw.image.height()-1)*0.5

		moments = np.zeros(8)

		for i in range(self.nvw.image.height()):
			for j in range(self.nvw.image.width()):
				dx = j - cx
				dy = i - cy
				moments += img[i,j]*np.array([
					dx, dy, dx*dx, dy*dy, dx*dy, dx*dx*dy, dy*dy*dx, dx*dx*dy*dy,

				])
		
		moments /= self.nvw.image.height()*self.nvw.image.width()

		print("np.%s," % repr(moments))

		train_moments = [
np.array([ -2.71753257e-04,   1.82478553e-05,   8.05689470e-02,   8.98118068e-02,   1.38005545e-03,   2.93691297e-03,  -1.02584777e-02,   2.04175352e+00]),
np.array([  4.81323961e-04,  -4.66405929e-04,   8.48766473e-02,   9.16351180e-02,  -1.88014037e-04,  -9.22007991e-03,   3.38415231e-04,   1.95394800e+00]),
np.array([ -7.18247850e-04,   1.13902810e-04,   7.53860623e-02,   9.44035643e-02,  -6.94522657e-04,   1.37308050e-02,  -8.87323132e-03,   1.92594727e+00]),
np.array([ -5.62474041e-04,  -8.43170258e-04,   8.06523662e-02,   9.50552225e-02,  -1.16968567e-03,  -5.12316283e-03,  -1.60721242e-02,   2.02329539e+00]),
np.array([  1.21197780e-04,   8.82283385e-04,   9.18003196e-02,   8.72346807e-02,  -4.66108435e-03,   1.55593618e-02,   2.07653623e-02,   1.99465709e+00]),
np.array([ -2.38617250e-05,   7.99982013e-05,   7.77577118e-02,   9.50124565e-02,  -3.09396487e-03,   4.93625941e-03,  -5.32225353e-03,   1.99991212e+00]),
np.array([ -8.07040962e-04,   6.56766411e-04,   8.25645396e-02,   9.52343868e-02,  -1.63392113e-03,   5.08905137e-03,  -8.05127368e-03,   2.07151381e+00]),
np.array([  1.53310426e-03,  -8.41183619e-04,   8.06848494e-02,   9.35057612e-02,   1.67215939e-03,  -5.14189905e-03,   2.24527105e-02,   1.95711246e+00]),
np.array([  1.53054802e-04,  -1.50109204e-04,   8.28906862e-02,   9.99143800e-02,  -8.15659231e-05,  -4.52607544e-03,  -4.44857909e-03,   2.13827880e+00]),
np.array([  3.28861030e-04,  -5.71444401e-04,   8.44269446e-02,   9.65669462e-02,   1.56398568e-03,  -2.55995098e-03,   4.83468272e-03,   2.12933916e+00]),
		]

		p = np.zeros(len(train_moments))
		for i in range(len(train_moments)):
			tmoments = train_moments[i]
			d = (moments - tmoments)/train_moments[5]
			dist = 1.0/(0.01 + np.linalg.norm(d))
			p[i] += dist
		p /= np.sum(p)

		return p
	"""


	def updateNumber(self):
		
		self.nvw.updateImage(self.ndw.path)

		# Konvertierung Bild zu Vektor der Länge 256 (Werte aus [-1, 1])
		vec = []
		for j in range(self.nvw.image.width()):
			for i in range(self.nvw.image.height()):
				vec.append((self.nvw.image.pixel(i, j) & 0xff)*self.classifier.image_scaling() + self.classifier.image_offset())

		# call Matlab to detect get number probabilities from vec
		# TODO
		#p = self.detectNumberStub(self.nvw.image)
		#p = np.random.rand(10)
		#p /= np.sum(p)

		p = self.classifier.predict(np.reshape(vec, (1, len(vec))))

		self.dpw.probabilities = p
		self.dnw.number = np.argmax(self.dpw.probabilities)

		self.dnw.update()
		self.dpw.update()

		
if __name__ == '__main__':

	np.set_printoptions(linewidth=1000)

	app = QApplication(sys.argv)
	ex = MainWidget()
	#ex.showMaximized()
	ex.show()
	sys.exit(app.exec_())
