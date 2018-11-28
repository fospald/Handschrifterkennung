# Handschrifterkennung

Handschrifterkennung von Zahlen mit TensorFlow

## Installation

To install (TensorFlow)[https://www.tensorflow.org/install/install_linux]
```
  sudo apt-get install python3-pip python3-dev python-virtualenv
  virtualenv --system-site-packages -p python3 ~/Work/Software/tensorflow
  source ~/Work/Software/tensorflow/bin/activate
  easy_install -U pip
  pip3 install --upgrade tensorflow
```

To test
  
```
  ipython3 
  import tensorflow as tf
  hello = tf.constant('Hello, TensorFlow!')
  sess = tf.Session()
  print(sess.run(hello))
```

To run

```
  ipython3 Handschrifterkennung.py
  ipython3 Handschrifterkennung_v2.py
```

