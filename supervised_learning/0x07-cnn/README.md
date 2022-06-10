# 0x07. Convolutional Neural Networks
## Details
      By Alexa Orrico, Software Engineer at Holberton School          Weight: 6              Ongoing second chance project - started Jun 6, 2022 , must end by Jun 13, 2022           - you're done with 0% of tasks.      Manual QA review must be done          (request it when you are done with the project)              An auto review will be launched at the deadline      #### In a nutshell…
* Manual QA review:          Pending      
* Auto QA review:          0.0/46 mandatory      
* Altogether:        waiting on some reviews    
 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/9/c9d2bd7153ac51f24e52.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220610%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220610T222105Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=80105e8dbf3de88ec44e1a2d315a898fa16bae949720d30ff653093e181e3e3d) 

## Resources
Read or watch :
* [Convolutional neural network](https://intranet.hbtn.io/rltoken/KOQSWajVz2BF6QuIM0ZyXg) 

* [Convolutional Neural Networks (CNNs) explained](https://intranet.hbtn.io/rltoken/YsCwFCpCZn5dIJM95qc2Dg) 

* [The best explanation of Convolutional Neural Networks on the Internet!](https://intranet.hbtn.io/rltoken/lOzKGVsnF4czDq1iVG5CJw) 
 (It’s pretty good but I wouldn’t call it the best…)
* [Machine Learning is Fun! Part 3: Deep Learning and Convolutional Neural Networks](https://intranet.hbtn.io/rltoken/S99iSsHQKr6d73WbYH5uEw) 

* [Convolutional Neural Networks: The Biologically-Inspired Model](https://intranet.hbtn.io/rltoken/XrV_YYGzqFEIZ6QIvDG7pQ) 

* [Back Propagation in Convolutional Neural Networks — Intuition and Code](https://intranet.hbtn.io/rltoken/B3AT0iVrA9nTo47ngD3ZDg) 

* [Backpropagation in a convolutional layer](https://intranet.hbtn.io/rltoken/umbBcqS1ijEixvL5aRfFWw) 

* [Convolutional Neural Network – Backward Propagation of the Pooling Layers](https://intranet.hbtn.io/rltoken/I2rEHVaSYQ3TbjmwyiCLrg) 

* [Pooling Layer](https://intranet.hbtn.io/rltoken/X2wFWCk1U97QfUUMV7hcwA) 

* [deeplearning.ai](https://intranet.hbtn.io/rltoken/BE_hLHcLBPNMkWJFFGaHNw) 
 videos (Note: I suggest watching these videos at 1.5x - 2x speed):* [Why Convolutions](https://intranet.hbtn.io/rltoken/pbJeODUGde5jWyVRzbYZWA) 

* [One Layer of a Convolutional Net](https://intranet.hbtn.io/rltoken/hQJA3cgC2mGBk4OfQkRqzg) 

* [Simple Convolutional Network Example](https://intranet.hbtn.io/rltoken/8AU4cPmX3jn8wkd0f0h-bg) 

* [CNN Example](https://intranet.hbtn.io/rltoken/JXJg5MMzZ4JEbM8Wv7oemw) 


* [Gradient-Based Learning Applied to Document Recognition (LeNet-5)](https://intranet.hbtn.io/rltoken/R84em95wWIF6PEEu4WG7HA) 

References :
* [tf.layers.Conv2D](https://intranet.hbtn.io/rltoken/P8iTl5HSNm0y5LQq2L4vcw) 

* [tf.keras.layers.Conv2D](https://intranet.hbtn.io/rltoken/eMO39JERmcFTwvkZ62NsHw) 

* [tf.layers.AveragePooling2D](https://intranet.hbtn.io/rltoken/kGx3e8VHoLOqh3vaCB6Rgg) 

* [tf.keras.layers.AveragePooling2D](https://intranet.hbtn.io/rltoken/QqUwCTDLzPdYlXWn5TexWw) 

* [tf.layers.MaxPooling2D](https://intranet.hbtn.io/rltoken/DSxDA_INxLRROjU0KRyePQ) 

* [tf.keras.layers.MaxPooling2D](https://intranet.hbtn.io/rltoken/CUi7PB0evfPFa0vMBGBVKQ) 

* [tf.layers.Flatten](https://intranet.hbtn.io/rltoken/C9htHwq74q1LMTS70x0xLg) 

* [tf.keras.layers.Flatten](https://intranet.hbtn.io/rltoken/7GmWhGCTk94KpMlDvp3RiQ) 

## Learning Objectives
At the end of this project, you are expected to be able to  [explain to anyone](https://intranet.hbtn.io/rltoken/esQCD89z9cq_MGjLqHFU1A) 
 ,  without the help of Google :
### General
* What is a convolutional layer?
* What is a pooling layer?
* Forward propagation over convolutional and pooling layers
* Back propagation over convolutional and pooling layers
* How to build a CNN using Tensorflow and Keras
## Requirements
### General
* Allowed editors:  ` vi ` ,  ` vim ` ,  ` emacs ` 
* All your files will be interpreted/compiled on Ubuntu 20.04 LTS using  ` python3 `  (version 3.8)
* Your files will be executed with  ` numpy `  (version 1.19.2) and  ` tensorflow `  (version 2.6)
* All your files should end with a new line
* The first line of all your files should be exactly  ` #!/usr/bin/env python3 ` 
* A  ` README.md `  file, at the root of the folder of the project, is mandatory
* Your code should use the  ` pycodestyle `  style (version 2.6)
* All your modules should have documentation ( ` python3 -c 'print(__import__("my_module").__doc__)' ` )
* All your classes should have documentation ( ` python3 -c 'print(__import__("my_module").MyClass.__doc__)' ` )
* All your functions (inside and outside a class) should have documentation ( ` python3 -c 'print(__import__("my_module").my_function.__doc__)' `  and  ` python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)' ` )
* Unless otherwise noted, you are not allowed to import any module
* All your files must be executable
* The length of your files will be tested using  ` wc ` 
## Tasks
### 0. Convolutional Forward Prop
          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write a function   ` def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)): `   that performs forward propagation over a convolutional layer of a neural network:
*  ` A_prev `  is a  ` numpy.ndarray `  of shape  ` (m, h_prev, w_prev, c_prev) `  containing the output of the previous layer*  ` m `  is the number of examples
*  ` h_prev `  is the height of the previous layer
*  ` w_prev `  is the width of the previous layer
*  ` c_prev `  is the number of channels in the previous layer

*  ` W `  is a  ` numpy.ndarray `  of shape  ` (kh, kw, c_prev, c_new) `  containing the kernels for the convolution*  ` kh `  is the filter height
*  ` kw `  is the filter width
*  ` c_prev `  is the number of channels in the previous layer
*  ` c_new `  is the number of channels in the output

*  ` b `  is a  ` numpy.ndarray `  of shape  ` (1, 1, 1, c_new) `  containing the biases applied to the convolution
*  ` activation `  is an activation function applied to the convolution
*  ` padding `  is a string that is either  ` same `  or  ` valid ` , indicating the type of padding used
*  ` stride `  is a tuple of  ` (sh, sw) `  containing the strides for the convolution*  ` sh `  is the stride for the height
*  ` sw `  is the stride for the width

* you may  ` import numpy as np ` 
* Returns: the output of the convolutional layer
```bash
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$ cat 0-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
conv_forward = __import__('0-conv_forward').conv_forward

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    m, h, w = X_train.shape
    X_train_c = X_train.reshape((-1, h, w, 1))

    W = np.random.randn(3, 3, 1, 2)
    b = np.random.randn(1, 1, 1, 2)

    def relu(Z):
        return np.maximum(Z, 0)

    plt.imshow(X_train[0])
    plt.show()
    A = conv_forward(X_train_c, W, b, relu, padding='valid')
    print(A.shape)
    plt.imshow(A[0, :, :, 0])
    plt.show()
    plt.imshow(A[0, :, :, 1])
    plt.show()
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$ ./0-main.py

```
 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/254c9d28e187cc72b2bf.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220610%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220610T222105Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c4123521f187eb1d4983d99938d40da960de58cb6ab9aa641721e4cc0dc9f537) 

 ` (50000, 26, 26, 2)
 `  ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/3/34cbf0e0145715bcca99.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220610%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220610T222105Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=91a3a9349af7acb5fcece4afa936e42ec68881ccac3f921ffe77969c55437ba4) 

 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/3/bff2983b614651fad107.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220610%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220610T222105Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2edfc1eff5745f958fc49c4d4f8c805e48027f104b3cee2662d1d50fecee5919) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x07-cnn ` 
* File:  ` 0-conv_forward.py ` 
 Self-paced manual review  Panel footer - Controls 
### 1. Pooling Forward Prop
          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write a function   ` def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'): `   that performs forward propagation over a pooling layer of a neural network:
*  ` A_prev `  is a  ` numpy.ndarray `  of shape  ` (m, h_prev, w_prev, c_prev) `  containing the output of the previous layer*  ` m `  is the number of examples
*  ` h_prev `  is the height of the previous layer
*  ` w_prev `  is the width of the previous layer
*  ` c_prev `  is the number of channels in the previous layer

*  ` kernel_shape `  is a tuple of  ` (kh, kw) `  containing the size of the kernel for the pooling*  ` kh `  is the kernel height
*  ` kw `  is the kernel width

*  ` stride `  is a tuple of  ` (sh, sw) `  containing the strides for the pooling*  ` sh `  is the stride for the height
*  ` sw `  is the stride for the width

*  ` mode `  is a string containing either  ` max `  or  ` avg ` , indicating whether to perform maximum or average pooling, respectively
* you may  ` import numpy as np ` 
* Returns: the output of the pooling layer
```bash
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$ cat 1-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
pool_forward = __import__('1-pool_forward').pool_forward

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    m, h, w = X_train.shape
    X_train_a = X_train.reshape((-1, h, w, 1))
    X_train_b = 1 - X_train_a
    X_train_c = np.concatenate((X_train_a, X_train_b), axis=3)

    print(X_train_c.shape)
    plt.imshow(X_train_c[0, :, :, 0])
    plt.show()
    plt.imshow(X_train_c[0, :, :, 1])
    plt.show()
    A = pool_forward(X_train_c, (2, 2), stride=(2, 2))
    print(A.shape)
    plt.imshow(A[0, :, :, 0])
    plt.show()
    plt.imshow(A[0, :, :, 1])
    plt.show()
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$ ./1-main.py 
(50000, 28, 28, 2)

```
 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/254c9d28e187cc72b2bf.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220610%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220610T222105Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c4123521f187eb1d4983d99938d40da960de58cb6ab9aa641721e4cc0dc9f537) 

 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/01a9bf81144713cd8e65.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220610%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220610T222105Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e60d16257833b75e0bb0f40c76411a13e31c856cc9dc409ec6800a319ceeba7f) 

 ` (50000, 14, 14, 2)
 `  ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/a961a7533f70bb98a695.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220610%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220610T222105Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=30d8e6532575c80656066eaf1c4238b6e7e4ecfe96d0a15689fe2c93885a32e0) 

 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/81ed31f0d740fb5067da.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220610%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220610T222105Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=60c57e997eeacbea74688790aa259ed81207cdde7d2a2fb08b01fdca1127af8a) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x07-cnn ` 
* File:  ` 1-pool_forward.py ` 
 Self-paced manual review  Panel footer - Controls 
### 2. Convolutional Back Prop
          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write a function   ` def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)): `   that performs back propagation over a convolutional layer of a neural network:
*  ` dZ `  is a  ` numpy.ndarray `  of shape  ` (m, h_new, w_new, c_new) `  containing the partial derivatives with respect to the unactivated output of the convolutional layer*  ` m `  is the number of examples
*  ` h_new `  is the height of the output
*  ` w_new `  is the width of the output
*  ` c_new `  is the number of channels in the output

*  ` A_prev `  is a  ` numpy.ndarray `  of shape  ` (m, h_prev, w_prev, c_prev) `  containing the output of the previous layer*  ` h_prev `  is the height of the previous layer
*  ` w_prev `  is the width of the previous layer
*  ` c_prev `  is the number of channels in the previous layer

*  ` W `  is a  ` numpy.ndarray `  of shape  ` (kh, kw, c_prev, c_new) `  containing the kernels for the convolution*  ` kh `  is the filter height
*  ` kw `  is the filter width

*  ` b `  is a  ` numpy.ndarray `  of shape  ` (1, 1, 1, c_new) `  containing the biases applied to the convolution
*  ` padding `  is a string that is either  ` same `  or  ` valid ` , indicating the type of padding used
*  ` stride `  is a tuple of  ` (sh, sw) `  containing the strides for the convolution*  ` sh `  is the stride for the height
*  ` sw `  is the stride for the width

* you may  ` import numpy as np ` 
* Returns: the partial derivatives with respect to the previous layer ( ` dA_prev ` ), the kernels ( ` dW ` ), and the biases ( ` db ` ), respectively
```bash
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$ cat 2-main.py
#!/usr/bin/env python3

import numpy as np
conv_backward = __import__('2-conv_backward').conv_backward

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    _, h, w = X_train.shape
    X_train_c = X_train[:10].reshape((-1, h, w, 1))

    W = np.random.randn(3, 3, 1, 2)
    b = np.random.randn(1, 1, 1, 2)

    dZ = np.random.randn(10, h - 2, w - 2, 2)
    print(conv_backward(dZ, X_train_c, W, b, padding="valid"))
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$ ./2-main.py
(array([[[[-4.24205748],
         [ 0.19390938],
         [-2.80168847],
         ...,
         [-2.93059274],
         [-0.74257184],
         [ 1.23556676]],

        [[-1.00865794],
         [ 0.24929631],
         [ 2.99153975],
         ...,
         [ 0.43357888],
         [ 4.96415936],
         [-0.44667327]],

        [[-1.87909273],
         [-1.52898354],
         [-1.03988664],
         ...,
         [-2.08719794],
         [ 0.72118428],
         [ 0.43712847]],

        ...,

        [[-1.85981381],
         [-4.35543293],
         [ 3.57636107],
         ...,
         [ 2.11136296],
         [ 0.53958723],
         [-3.52000282]],

        [[-1.0499573 ],
         [-2.04696766],
         [-3.65137871],
         ...,
         [-0.52756967],
         [-0.08825488],
         [ 0.62158883]],

        [[-0.33383597],
         [-0.68426308],
         [-1.16737412],
         ...,
         [ 0.38028383],
         [ 0.90910959],
         [ 1.1018034 ]]],


       [[[ 1.3242862 ],
         [ 3.35050521],
         [-2.61244835],
         ...,
         [-5.35657632],
         [ 0.76179689],
         [ 2.18585273]],

        [[ 0.41947984],
         [ 2.29805997],
         [ 0.70448521],
         ...,
         [-0.15055621],
         [-1.85010471],
         [ 0.22182008]],

        [[-0.44134373],
         [ 1.70998625],
         [-0.43519259],
         ...,
         [-0.84228164],
         [ 0.06743225],
         [-0.33952493]],

        ...,

        [[-0.84072841],
         [ 2.23096657],
         [ 4.2740757 ],
         ...,
         [-1.64328314],
         [-1.00825088],
         [ 0.06493264]],

        [[ 0.51461905],
         [ 1.74947396],
         [ 3.12442805],
         ...,
         [ 2.64632352],
         [ 1.11166957],
         [-2.17004665]],

        [[-0.15977939],
         [ 0.71088702],
         [ 0.58870058],
         ...,
         [ 0.79037467],
         [-1.872449  ],
         [ 0.22958953]]],


       [[[-2.55102529],
         [-1.43443829],
         [-6.43429192],
         ...,
         [ 4.43919873],
         [-2.3961974 ],
         [ 1.12105391]],

        [[-3.49933601],
         [ 2.97808   ],
         [-5.94765644],
         ...,
         [-1.52227952],
         [ 0.71633969],
         [-2.69268038]],

        [[-0.6049378 ],
         [ 3.00515277],
         [-3.82581326],
         ...,
         [-0.82612782],
         [ 1.10270878],
         [ 0.57341665]],

        ...,

        [[ 2.47206612],
         [ 6.12030267],
         [ 4.85570283],
         ...,
         [ 1.7069348 ],
         [-3.26558701],
         [-2.19265787]],

        [[ 0.82794065],
         [ 2.50876332],
         [ 4.94170337],
         ...,
         [-4.11611469],
         [-1.89129533],
         [ 1.02817795]],

        [[ 0.61583613],
         [ 1.21100799],
         [ 1.26340831],
         ...,
         [-1.46870175],
         [-2.48288945],
         [-2.49803816]]],


       ...,


       [[[ 0.36480084],
         [ 4.05009666],
         [ 2.40882213],
         ...,
         [-1.39742733],
         [-1.58184928],
         [ 1.5492834 ]],

        [[-0.59246796],
         [-5.14195445],
         [-4.73361645],
         ...,
         [-6.29937402],
         [ 2.57781547],
         [-6.22413954]],

        [[ 0.28940123],
         [ 3.30399397],
         [-9.92107171],
         ...,
         [-4.7873951 ],
         [-5.51345667],
         [ 2.59603062]],

        ...,

        [[ 0.31895703],
         [ 2.7620854 ],
         [ 2.40446498],
         ...,
         [ 2.68160757],
         [ 2.3774331 ],
         [-5.17924359]],

        [[-0.84079478],
         [ 0.92656007],
         [ 1.69220611],
         ...,
         [ 0.23381858],
         [ 0.65019692],
         [ 2.52647242]],

        [[-0.21035363],
         [-0.49657321],
         [-0.97588817],
         ...,
         [ 1.37568796],
         [ 0.75783393],
         [-2.06076966]]],


       [[[-0.8764177 ],
         [ 0.04226753],
         [-3.92342249],
         ...,
         [-3.04784534],
         [-0.40436888],
         [ 0.42939003]],

        [[-1.99854061],
         [-1.36763433],
         [-3.31601105],
         ...,
         [ 3.56163624],
         [-5.45977866],
         [-1.1221114 ]],

        [[-2.97880521],
         [-7.02474334],
         [-2.6208715 ],
         ...,
         [-2.66868613],
         [-3.35947227],
         [ 1.52739149]],

        ...,

        [[-0.76204177],
         [-2.39471119],
         [ 1.88614862],
         ...,
         [ 8.52140674],
         [ 2.87244213],
         [ 5.4831909 ]],

        [[-0.28094631],
         [-1.54524622],
         [-2.26649997],
         ...,
         [ 4.01337541],
         [ 1.72949251],
         [ 0.26894907]],

        [[ 0.2333244 ],
         [ 0.15360826],
         [ 0.61304729],
         ...,
         [ 4.82873779],
         [ 1.58564885],
         [ 3.77278834]]],


       [[[-4.64117569],
         [-6.30127451],
         [-1.35549413],
         ...,
         [ 5.73490276],
         [ 4.48763997],
         [ 0.90584946]],

        [[-2.06780074],
         [ 0.74310235],
         [ 2.32306348],
         ...,
         [-1.93057052],
         [ 1.73865934],
         [ 1.29870813]],

        [[ 0.48429556],
         [-3.18452582],
         [-3.1882709 ],
         ...,
         [ 1.14229413],
         [-0.68614631],
         [ 0.48510011]],

        ...,

        [[ 1.31359094],
         [ 1.80393793],
         [-2.56324511],
         ...,
         [ 1.87402318],
         [ 2.10343171],
         [ 4.90609163]],

        [[ 0.984754  ],
         [ 0.49587505],
         [-0.26741779],
         ...,
         [ 1.93306272],
         [ 3.19125427],
         [-0.9173847 ]],

        [[ 0.87318188],
         [ 0.96086254],
         [ 1.69739496],
         ...,
         [-0.28586324],
         [ 2.24643738],
         [ 0.74045003]]]]), array([[[[ 10.13352674, -25.15674655]],

        [[ 33.27872337, -64.99062958]],

        [[ 31.29539025, -77.29275492]]],


       [[[ 10.61025981, -31.7337223 ]],

        [[ 10.34048231, -65.19271124]],

        [[ -1.73024336, -76.98703808]]],


       [[[ -1.49204439, -33.46094911]],

        [[  4.04542976, -63.47295685]],

        [[  2.9243666 , -64.29296016]]]]), array([[[[-113.18404846, -121.902714  ]]]]))
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x07-cnn ` 
* File:  ` 2-conv_backward.py ` 
 Self-paced manual review  Panel footer - Controls 
### 3. Pooling Back Prop
          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write a function   ` def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'): `   that performs back propagation over a pooling layer of a neural network:
*  ` dA `  is a  ` numpy.ndarray `  of shape  ` (m, h_new, w_new, c_new) `  containing the partial derivatives with respect to the output of the pooling layer*  ` m `  is the number of examples
*  ` h_new `  is the height of the output
*  ` w_new `  is the width of the output
*  ` c `  is the number of channels

*  ` A_prev `  is a  ` numpy.ndarray `  of shape  ` (m, h_prev, w_prev, c) `  containing the output of the previous layer*  ` h_prev `  is the height of the previous layer
*  ` w_prev `  is the width of the previous layer

*  ` kernel_shape `  is a tuple of  ` (kh, kw) `  containing the size of the kernel for the pooling*  ` kh `  is the kernel height
*  ` kw `  is the kernel width

*  ` stride `  is a tuple of  ` (sh, sw) `  containing the strides for the pooling*  ` sh `  is the stride for the height
*  ` sw `  is the stride for the width

*  ` mode `  is a string containing either  ` max `  or  ` avg ` , indicating whether to perform maximum or average pooling, respectively
* you may  ` import numpy as np ` 
* Returns: the partial derivatives with respect to the previous layer ( ` dA_prev ` )
```bash
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$ cat 3-main.py
#!/usr/bin/env python3

import numpy as np
pool_backward = __import__('3-pool_backward').pool_backward

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    _, h, w = X_train.shape
    X_train_a = X_train[:10].reshape((-1, h, w, 1))
    X_train_b = 1 - X_train_a
    X_train_c = np.concatenate((X_train_a, X_train_b), axis=3)

    dA = np.random.randn(10, h // 3, w // 3, 2)
    print(pool_backward(dA, X_train_c, (3, 3), stride=(3, 3)))
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$ ./3-main.py
[[[[ 1.76405235  0.40015721]
   [ 1.76405235  0.40015721]
   [ 0.97873798  2.2408932 ]
   ...
   [ 2.26975462 -1.45436567]
   [ 0.04575852 -0.18718385]
   [ 0.04575852 -0.18718385]]

  [[ 1.76405235  0.40015721]
   [ 1.76405235  0.40015721]
   [ 0.97873798  2.2408932 ]
   ...
   [ 2.26975462 -1.45436567]
   [ 0.04575852 -0.18718385]
   [ 0.04575852 -0.18718385]]

  [[ 1.53277921  1.46935877]
   [ 1.53277921  1.46935877]
   [ 0.15494743  0.37816252]
   ...
   [-0.51080514 -1.18063218]
   [-0.02818223  0.42833187]
   [-0.02818223  0.42833187]]

  ...

  [[-1.75589058  0.45093446]
   [-1.75589058  0.45093446]
   [-0.6840109   1.6595508 ]
   ...
   [ 0.69845715  0.00377089]
   [ 0.93184837  0.33996498]
   [ 0.93184837  0.33996498]]

  [[-0.01568211  0.16092817]
   [-0.01568211  0.16092817]
   [-0.19065349 -0.39484951]
   ...
   [ 1.64813493  0.16422776]
   [ 0.56729028 -0.2226751 ]
   [ 0.56729028 -0.2226751 ]]

  [[-0.01568211  0.16092817]
   [-0.01568211  0.16092817]
   [-0.19065349 -0.39484951]
   ...
   [ 1.64813493  0.16422776]
   [ 0.56729028 -0.2226751 ]
   [ 0.56729028 -0.2226751 ]]]


 [[[-0.35343175 -1.61647419]
   [-0.35343175 -1.61647419]
   [-0.29183736 -0.76149221]
   ...
   [ 0.370825    0.14206181]
   [ 1.51999486  1.71958931]
   [ 1.51999486  1.71958931]]

  [[-0.35343175 -1.61647419]
   [-0.35343175 -1.61647419]
   [-0.29183736 -0.76149221]
   ...
   [ 0.370825    0.14206181]
   [ 1.51999486  1.71958931]
   [ 1.51999486  1.71958931]]

  [[ 0.92950511  0.58222459]
   [ 0.92950511  0.58222459]
   [-2.09460307  0.12372191]
   ...
   [ 0.87583276 -0.11510747]
   [ 0.45741561 -0.96461201]
   [ 0.45741561 -0.96461201]]

  ...

  [[ 0.81267404  0.58725938]
   [ 0.81267404  0.58725938]
   [-0.50535832 -0.81579154]
   ...
   [ 0.44819528  1.69618157]
   [-0.0148577   0.82140594]
   [-0.0148577   0.82140594]]

  [[ 0.67057045 -0.7075057 ]
   [ 0.67057045 -0.7075057 ]
   [ 0.03976673 -1.56699471]
   ...
   [ 1.48935596  0.52130375]
   [ 0.61192719 -1.34149673]
   [ 0.61192719 -1.34149673]]

  [[ 0.67057045 -0.7075057 ]
   [ 0.67057045 -0.7075057 ]
   [ 0.03976673 -1.56699471]
   ...
   [ 1.48935596  0.52130375]
   [ 0.61192719 -1.34149673]
   [ 0.61192719 -1.34149673]]]


 [[[ 0.47689837  0.14844958]
   [ 0.47689837  0.14844958]
   [ 0.52904524  0.42262862]
   ...
   [-1.68823003 -0.11246598]
   [-0.53248992  0.64505527]
   [-0.53248992  0.64505527]]

  [[ 0.47689837  0.14844958]
   [ 0.47689837  0.14844958]
   [ 0.52904524  0.42262862]
   ...
   [-1.68823003 -0.11246598]
   [-0.53248992  0.64505527]
   [-0.53248992  0.64505527]]

  [[ 1.01184243 -0.65795104]
   [ 1.01184243 -0.65795104]
   [ 0.46838523  1.735879  ]
   ...
   [-0.84832052 -0.32566947]
   [ 0.47043314  0.31144707]
   [ 0.47043314  0.31144707]]

  ...

  [[ 0.73035179  1.10457847]
   [ 0.73035179  1.10457847]
   [-1.01482591 -0.60233185]
   ...
   [ 0.76449745 -0.26837274]
   [-0.16975829 -0.13413278]
   [-0.16975829 -0.13413278]]

  [[ 1.22138496 -0.19284183]
   [ 1.22138496 -0.19284183]
   [-0.03331928 -1.5308035 ]
   ...
   [ 0.77067305 -0.13043973]
   [ 1.8219151  -0.07565047]
   [ 1.8219151  -0.07565047]]

  [[ 1.22138496 -0.19284183]
   [ 1.22138496 -0.19284183]
   [-0.03331928 -1.5308035 ]
   ...
   [ 0.77067305 -0.13043973]
   [ 1.8219151  -0.07565047]
   [ 1.8219151  -0.07565047]]]


 ...


 [[[ 0.98232698  1.0374448 ]
   [ 0.98232698  1.0374448 ]
   [ 0.15919177 -0.98809669]
   ...
   [-1.04481632  0.78990494]
   [ 1.10228256 -0.69707307]
   [ 1.10228256 -0.69707307]]

  [[ 0.98232698  1.0374448 ]
   [ 0.98232698  1.0374448 ]
   [ 0.15919177 -0.98809669]
   ...
   [-1.04481632  0.78990494]
   [ 1.10228256 -0.69707307]
   [ 1.10228256 -0.69707307]]

  [[ 0.20733405  0.75915668]
   [ 0.20733405  0.75915668]
   [ 0.1005642  -0.95494276]
   ...
   [-0.90207197  0.32099947]
   [-1.39201592  0.59220568]
   [-1.39201592  0.59220568]]

  ...

  [[-0.38731344 -0.34758451]
   [-0.38731344 -0.34758451]
   [ 3.30657435 -1.51019964]
   ...
   [ 0.39597621  1.8115057 ]
   [-0.86907759 -0.45822915]
   [-0.86907759 -0.45822915]]

  [[-1.13832396  0.12916217]
   [-1.13832396  0.12916217]
   [ 0.0640242   0.7050811 ]
   ...
   [-1.38141165 -0.61263856]
   [-0.38128987 -1.24894893]
   [-0.38128987 -1.24894893]]

  [[-1.13832396  0.12916217]
   [-1.13832396  0.12916217]
   [ 0.0640242   0.7050811 ]
   ...
   [-1.38141165 -0.61263856]
   [-0.38128987 -1.24894893]
   [-0.38128987 -1.24894893]]]


 [[[-0.33023789 -0.83480716]
   [-0.33023789 -0.83480716]
   [ 1.23538239 -0.2438038 ]
   ...
   [ 1.14164795  1.27415503]
   [-1.66469804  0.43037888]
   [-1.66469804  0.43037888]]

  [[-0.33023789 -0.83480716]
   [-0.33023789 -0.83480716]
   [ 1.23538239 -0.2438038 ]
   ...
   [ 1.14164795  1.27415503]
   [-1.66469804  0.43037888]
   [-1.66469804  0.43037888]]

  [[-0.04260193  0.38828881]
   [-0.04260193  0.38828881]
   [ 1.11597653 -0.92053816]
   ...
   [-0.36167246  2.15371951]
   [ 0.84740836 -0.19871985]
   [ 0.84740836 -0.19871985]]

  ...

  [[ 0.06349048 -0.2213905 ]
   [ 0.06349048 -0.2213905 ]
   [-0.16339892 -0.15630347]
   ...
   [-1.20324346  1.17803124]
   [ 0.1086482   0.0441291 ]
   [ 0.1086482   0.0441291 ]]

  [[ 0.33831554  1.44679207]
   [ 0.33831554  1.44679207]
   [-0.21449511  1.66303896]
   ...
   [-1.81588884  0.7510996 ]
   [ 0.30028432  2.11060853]
   [ 0.30028432  2.11060853]]

  [[ 0.33831554  1.44679207]
   [ 0.33831554  1.44679207]
   [-0.21449511  1.66303896]
   ...
   [-1.81588884  0.7510996 ]
   [ 0.30028432  2.11060853]
   [ 0.30028432  2.11060853]]]


 [[[ 1.41308554  1.50698036]
   [ 1.41308554  1.50698036]
   [ 0.8173971   0.64661561]
   ...
   [ 0.21132303 -0.20239426]
   [-0.62192816  0.16377045]
   [-0.62192816  0.16377045]]

  [[ 1.41308554  1.50698036]
   [ 1.41308554  1.50698036]
   [ 0.8173971   0.64661561]
   ...
   [ 0.21132303 -0.20239426]
   [-0.62192816  0.16377045]
   [-0.62192816  0.16377045]]

  [[ 0.80243891  0.28900589]
   [ 0.80243891  0.28900589]
   [-0.55364239  0.33625402]
   ...
   [ 0.19677009  0.96962373]
   [-1.71864988 -1.05695677]
   [-1.71864988 -1.05695677]]

  ...

  [[-0.91176848  1.12190735]
   [-0.91176848  1.12190735]
   [ 1.39283743 -1.37701857]
   ...
   [ 0.36196071 -0.2927121 ]
   [ 0.62182153 -0.80323948]
   [ 0.62182153 -0.80323948]]

  [[-0.92196766  1.77405634]
   [-0.92196766  1.77405634]
   [ 0.02875624  0.55296385]
   ...
   [ 0.24288982 -0.40083471]
   [-1.02155985 -0.47002432]
   [-1.02155985 -0.47002432]]

  [[-0.92196766  1.77405634]
   [-0.92196766  1.77405634]
   [ 0.02875624  0.55296385]
   ...
   [ 0.24288982 -0.40083471]
   [-1.02155985 -0.47002432]
   [-1.02155985 -0.47002432]]]]
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x07-cnn ` 
* File:  ` 3-pool_backward.py ` 
 Self-paced manual review  Panel footer - Controls 
### 4. LeNet-5 (Tensorflow 1)
          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body  ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/4fb0e30dfb666ae3a592.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220610%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220610T222105Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c4e07b5511b77a677afb65870720fc73c2da73ba16e116f2676b75adda864c59) 

Write a function   ` def lenet5(x, y): `   that builds a modified version of the   ` LeNet-5 `   architecture using   ` tensorflow `  :
*  ` x `  is a  ` tf.placeholder `  of shape  ` (m, 28, 28, 1) `  containing the input images for the network*  ` m `  is the number of images

*  ` y `  is a  ` tf.placeholder `  of shape  ` (m, 10) `  containing the one-hot labels for the network
* The model should consist of the following layers in order:* Convolutional layer with 6 kernels of shape 5x5 with  ` same `  padding
* Max pooling layer with kernels of shape 2x2 with 2x2 strides
* Convolutional layer with 16 kernels of shape 5x5 with  ` valid `  padding
* Max pooling layer with kernels of shape 2x2 with 2x2 strides
* Fully connected layer with 120 nodes
* Fully connected layer with 84 nodes
* Fully connected softmax output layer with 10 nodes

* All layers requiring initialization should initialize their kernels with the  ` he_normal `  initialization method:  ` tf.keras.initializers.VarianceScaling(scale=2.0) ` 
* All hidden layers requiring activation should use the  ` relu `  activation function
* you may  ` import tensorflow.compat.v1 as tf ` 
* you may NOT use  ` tf.keras `  only for the  ` he_normal `  method.
* Returns:* a tensor for the softmax activated output
* a training operation that utilizes  ` Adam `  optimization (with default hyperparameters)
* a tensor for the loss of the netowrk
* a tensor for the accuracy of the network

```bash
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$ cat 4-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
lenet5 = __import__('4-lenet5').lenet5

if __name__ == "__main__":
    np.random.seed(0)
    tf.set_random_seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    Y_train = lib['Y_train']
    X_valid = lib['X_valid']
    Y_valid = lib['Y_valid']
    m, h, w = X_train.shape
    X_train_c = X_train.reshape((-1, h, w, 1))
    X_valid_c = X_valid.reshape((-1, h, w, 1))
    x = tf.placeholder(tf.float32, (None, h, w, 1))
    y = tf.placeholder(tf.int32, (None,))
    y_oh = tf.one_hot(y, 10)
    y_pred, train_op, loss, acc = lenet5(x, y_oh)
    batch_size = 32
    epochs = 10
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            cost, accuracy = sess.run((loss, acc), feed_dict={x:X_train_c, y:Y_train})
            cost_valid, accuracy_valid = sess.run((loss, acc), feed_dict={x:X_valid_c, y:Y_valid})
            print("After {} epochs: {} cost, {} accuracy, {} validation cost, {} validation accuracy".format(epoch, cost, accuracy, cost_valid, accuracy_valid))
            p = np.random.permutation(m)
            X_shuffle = X_train_c[p]
            Y_shuffle = Y_train[p]
            for i in range(0, m, batch_size):
                X_batch = X_shuffle[i:i+batch_size]
                Y_batch = Y_shuffle[i:i+batch_size]
                sess.run(train_op, feed_dict={x:X_batch, y:Y_batch})
        cost, accuracy = sess.run((loss, acc), feed_dict={x:X_train_c, y:Y_train})
        cost_valid, accuracy_valid = sess.run((loss, acc), feed_dict={x:X_valid_c, y:Y_valid})
        print("After {} epochs: {} cost, {} accuracy, {} validation cost, {} validation accuracy".format(epochs, cost, accuracy, cost_valid, accuracy_valid))
        Y_pred = sess.run(y_pred, feed_dict={x:X_valid_c, y:Y_valid})
        print(Y_pred[0])
        Y_pred = np.argmax(Y_pred, 1)
        plt.imshow(X_valid[0])
        plt.title(str(Y_valid[0]) + ' : ' + str(Y_pred[0]))
        plt.show()
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$ ./4-main.py 
2018-12-11 01:13:48.838837: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
After 0 epochs: 2.2976269721984863 cost, 0.08017999678850174 accuracy, 2.2957489490509033 validation cost, 0.08389999717473984 validation accuracy
After 1 epochs: 0.06289318203926086 cost, 0.9816200137138367 accuracy, 0.0687578096985817 validation cost, 0.9805999994277954 validation accuracy
After 2 epochs: 0.04042838513851166 cost, 0.987559974193573 accuracy, 0.04974357411265373 validation cost, 0.9861000180244446 validation accuracy
After 3 epochs: 0.033414799720048904 cost, 0.989300012588501 accuracy, 0.048249948769807816 validation cost, 0.9868000149726868 validation accuracy
After 4 epochs: 0.03417244181036949 cost, 0.989080011844635 accuracy, 0.06006946414709091 validation cost, 0.983299970626831 validation accuracy
After 5 epochs: 0.019328827038407326 cost, 0.9940000176429749 accuracy, 0.03986175358295441 validation cost, 0.9883999824523926 validation accuracy
[7.69712392e-14 1.46036297e-12 1.26758201e-10 9.99998450e-01
 2.11756339e-14 2.26456431e-09 1.75634965e-13 1.45111270e-10
 1.56041858e-06 1.31521265e-08]

```
 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/ff23c1be91ad2c4ec377.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220610%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220610T222105Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=674207760c7727a4224cd9594d45914feaf5ea85c86406c2bb36724ca74edd6c) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x07-cnn ` 
* File:  ` 4-lenet5.py ` 
 Self-paced manual review  Panel footer - Controls 
### 5. LeNet-5 (Keras)
          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write a function   ` def lenet5(X): `   that builds a modified version of the   ` LeNet-5 `   architecture using   ` keras `  :
*  ` X `  is a  ` K.Input `  of shape  ` (m, 28, 28, 1) `  containing the input images for the network*  ` m `  is the number of images

* The model should consist of the following layers in order:* Convolutional layer with 6 kernels of shape 5x5 with  ` same `  padding
* Max pooling layer with kernels of shape 2x2 with 2x2 strides
* Convolutional layer with 16 kernels of shape 5x5 with  ` valid `  padding
* Max pooling layer with kernels of shape 2x2 with 2x2 strides
* Fully connected layer with 120 nodes
* Fully connected layer with 84 nodes
* Fully connected softmax output layer with 10 nodes

* All layers requiring initialization should initialize their kernels with the  ` he_normal `  initialization method
* All hidden layers requiring activation should use the  ` relu `  activation function
* you may  ` import tensorflow.keras as K ` 
* Returns: a  ` K.Model `  compiled to use  ` Adam `  optimization (with default hyperparameters) and  ` accuracy `  metrics
```bash
ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$ cat 5-main.py
#!/usr/bin/env python3
"""
Main file
"""
# Force Seed - fix for Keras
SEED = 0
import matplotlib.pyplot as plt
import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
import tensorflow.keras as K

lenet5 = __import__('5-lenet5').lenet5

if __name__ == "__main__":
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    m, h, w = X_train.shape
    X_train_c = X_train.reshape((-1, h, w, 1))
    Y_train = lib['Y_train']
    Y_train_oh = K.utils.to_categorical(Y_train, num_classes=10)
    X_valid = lib['X_valid']
    X_valid_c = X_valid.reshape((-1, h, w, 1))
    Y_valid = lib['Y_valid']
    Y_valid_oh = K.utils.to_categorical(Y_valid, num_classes=10)
    X = K.Input(shape=(h, w, 1))
    model = lenet5(X)
    batch_size = 32
    epochs = 5
    model.fit(X_train_c, Y_train_oh, batch_size=batch_size, epochs=epochs,
                       validation_data=(X_valid_c, Y_valid_oh))
    Y_pred = model.predict(X_valid_c)
    print(Y_pred[0])
    Y_pred = np.argmax(Y_pred, 1)
    plt.imshow(X_valid[0])
    plt.title(str(Y_valid[0]) + ' : ' + str(Y_pred[0]))
    plt.show()

ubuntu@alexa-ml:~/supervised_learning/0x07-cnn$ ./5-main.py
Epoch 1/5
1563/1563 [==============================] - 11s 4ms/step - loss: 0.1665 - accuracy: 0.9489 - val_loss: 0.0596 - val_accuracy: 0.9813
Epoch 2/5
1563/1563 [==============================] - 6s 4ms/step - loss: 0.0594 - accuracy: 0.9820 - val_loss: 0.0489 - val_accuracy: 0.9859
Epoch 3/5
1563/1563 [==============================] - 6s 4ms/step - loss: 0.0408 - accuracy: 0.9869 - val_loss: 0.0469 - val_accuracy: 0.9870
Epoch 4/5
1563/1563 [==============================] - 6s 4ms/step - loss: 0.0346 - accuracy: 0.9889 - val_loss: 0.0482 - val_accuracy: 0.9870
Epoch 5/5
1563/1563 [==============================] - 6s 4ms/step - loss: 0.0255 - accuracy: 0.9917 - val_loss: 0.0483 - val_accuracy: 0.9875
[4.5772337e-16 2.6305156e-12 1.4343354e-13 1.0000000e+00 2.8758866e-17
 3.4095468e-08 3.7155215e-15 3.2845108e-13 3.5915697e-11 4.5209600e-11]

```
 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/ff23c1be91ad2c4ec377.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220610%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220610T222105Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=674207760c7727a4224cd9594d45914feaf5ea85c86406c2bb36724ca74edd6c) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x07-cnn ` 
* File:  ` 5-lenet5.py ` 
 Self-paced manual review  Panel footer - Controls 
### 6. Summarize Like a Pro
          mandatory         Progress vs Score  Task Body A common practice in the machine learning industry is to read and review journal articles on a weekly basis. Read and write a summary of Krizhevsky et. al.‘s 2012 paper  [ImageNet Classification with Deep Convolutional Neural Networks](https://intranet.hbtn.io/rltoken/hj0CacwoEVC2GY1StNsPlA) 
 . Your summary should include:
* Introduction: Give the necessary background to the study and state its purpose.
* Procedures: Describe the specifics of what this study involved.
* Results: In your own words, discuss the major findings and results.
* Conclusion: In your own words, summarize the researchers’ conclusions.
* Personal Notes: Give your reaction to the study.
Your posts should have examples and at least one picture, at the top. Publish your blog post on Medium or LinkedIn, and share it at least on Twitter and LinkedIn.
When done, please add all URLs below (blog post, tweet, etc.)
Please, remember that these blogs must be written in English to further your technical ability in a variety of settings.
 Task URLs #### Add URLs here:
                Save               Github information  Self-paced manual review  Panel footer - Controls 
Ready for a  manual review
