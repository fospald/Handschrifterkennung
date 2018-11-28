# Handschrifterkennung

Handwriting recognition of numbers with TensorFlow.

## Requirements

* [Python 3](https://www.python.org/)
* [scipy](https://www.scipy.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [PyQt5](https://www.riverbankcomputing.com/software/pyqt/download5)

## Running

```
  python3 Handschrifterkennung.py
  python3 Handschrifterkennung_v2.py
```

## Algorithm

The handwriting recognition algorithm is based on the mathematical method of Support Vector Machines (SVM). The basis of this procedure is a set of test data. The SVMs process looks for a hyperplane that separates training objects. This creates two different classes. If a new object is added, it can be assigned to a class. In the case of digit recognition, the amount of training data is a series of examples of written numbers. For these numbers it is known to which class they belong, that is to say which number they should represent. Generally speaking, the program is trained with these test data. The goal is that the trained program gradually recognizes a pattern and then automatically assigns the number written by the user to the correct class. The better the program is trained, the lower the probability of error.

## GUI

The user interface for digit recognition consists of four windows and two buttons. The large window for user input is located on the left side of the surface. On the right side, the pixmap, the bar graph of the number probabilities and the recognized number are arranged one above the other. At the top of the resulting user interface are the full screen and the Help button.
When used for the first time, it will automatically be trained with 20000 test data. For each subsequent use this is already done, after which it is dispensed with. The operation of the application is kept simple. As a user, if you draw a number in the input field, it is automatically transformed into a pixmap when you have finished the input. It is assumed that this is the case if one second has not been drawn. This pixmap is read into the handwriting recognition program in the background and further processed. Subsequently, the number probabilities are passed to our program. These processes are hidden from you as a user. So after you type the number, it's drawn in a 16x16 pixmap that you see. The probabilities returned by the handwriting recognition program are displayed in a bar graph in percent, the bar of the most probable number is greened and the number is output as the recognized digit in the corresponding window. If the button full screen is pressed, the full screen mode is activated for the user interface. When opening the help button, this separate window will be opened, which will help you to use it.

When the user interface is opened, the 5 is output as a recognized number. In addition, the individual probabilities of the numbers give a total of 101 percent. This seems inappropriate and confusing at first. After all, no number was drawn in the input window at this time. However, this is the result of digit recognition applied to a blank picture. The sum of the probabilities of 101 percent is due to rounding errors. You can see that in this case the individual numbers are almost equally probable. The program most likely assumes the figure 5 for an empty picture.

## MNIST Training Set

As described above, the evaluation of the user input is based on training data. These are from the US-American font usage. Since the usual spelling of some digits differs from that of the training data, it may happen that the program fools with these numbers more frequently and the input does not recognize correctly. If you as a user have this problem, you can orient yourself with the input of the training data below.

![MNIST Examples](MnistExamples.png)

See also: [MNIST Homepage](http://yann.lecun.com/exdb/mnist/)


