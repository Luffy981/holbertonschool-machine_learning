# 0x04. Error Analysis
## Details
      By Alexa Orrico, Software Engineer at Holberton School          Weight: 1                Ongoing project - started May 23, 2022 , must end by May 24, 2022           - you're done with 0% of tasks.              Checker was released at May 23, 2022 12:00 PM        An auto review will be launched at the deadline       ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/11/e3786a3d84e36ff800d8.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220524%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220524T031508Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=992e69b8c497e77ae91d9c71039d51211087d2bd8493764b9d08380379ba4fe5) 

## Resources
Read or watch :
* [Confusion matrix](https://intranet.hbtn.io/rltoken/Bn9M-MGfoJrw0-TpHTa9Uw) 

* [Type I and type II errors](https://intranet.hbtn.io/rltoken/fxhGH4L-87fD_e11L-T0IA) 

* [Sensitivity and specificity](https://intranet.hbtn.io/rltoken/jn65gXxuPRCOX3zo7ZMAVg) 

* [Precision and recall](https://intranet.hbtn.io/rltoken/a2j2_WIV27HgPCm2rXYW0A) 

* [F1 score](https://intranet.hbtn.io/rltoken/n0icgR0KqaHEdpn3FAbJoQ) 

* [What is a Confusion Matrix in Machine Learning?](https://intranet.hbtn.io/rltoken/qocVwJJrC7gC9cOUc2Wn7A) 

* [Simple guide to confusion matrix terminology](https://intranet.hbtn.io/rltoken/YSLbZZN4UAp33VXvfhFWyA) 

* [Bias-variance tradeoff](https://intranet.hbtn.io/rltoken/eWYy4ivH1yTEU0SYElZNxA) 

* [What is bias and variance](https://intranet.hbtn.io/rltoken/aPtj03_mws2J_d50hWU8TA) 

* [Bayes error rate](https://intranet.hbtn.io/rltoken/VC4wmuWuQH7Du-uLOZ2AZg) 

* [What is Bayes Error in machine learning?](https://intranet.hbtn.io/rltoken/x6wgEm5-QbyIehgFZCb2rQ) 

* [Bias/Variance](https://intranet.hbtn.io/rltoken/OXuEmLkHubDofoueWMlY7A) 
 (Note: I suggest watching this video at 1.5x - 2x speed)
* [Basic Recipe for Machine Learning](https://intranet.hbtn.io/rltoken/gVKdBNxmO8FU3eQaNClVZQ) 
 (Note: I suggest watching this video at 1.5x - 2x speed)
* [Why Human Level Performance](https://intranet.hbtn.io/rltoken/M6c62wjBk5AOOowkj14ITA) 
 (Note: I suggest watching this video at 1.5x - 2x speed)
* [Avoidable Bias](https://intranet.hbtn.io/rltoken/1yhh1YA_Xa_R3t0xUy4p9Q) 
 (Note: I suggest watching this video at 1.5x - 2x speed)
* [Understanding Human-Level Performance](https://intranet.hbtn.io/rltoken/YtDkYixp6TUAxMc4liTtBg) 
 (Note: I suggest watching this video at 1.5x - 2x speed)
## Learning Objectives
At the end of this project, you are expected to be able to  [explain to anyone](https://intranet.hbtn.io/rltoken/N7wXKztKmZ_m6XOhu3t4NQ) 
 ,  without the help of Google :
### General
* What is the confusion matrix?
* What is type I error? type II?
* What is sensitivity? specificity? precision? recall?
* What is an F1 score?
* What is bias? variance?
* What is irreducible error?
* What is Bayes error?
* How can you approximate Bayes error?
* How to calculate bias and variance
* How to create a confusion matrix
## Requirements
### General
* Allowed editors:  ` vi ` ,  ` vim ` ,  ` emacs ` 
* All your files will be interpreted/compiled on Ubuntu 16.04 LTS using  ` python3 `  (version 3.5)
* Your files will be executed with  ` numpy `  (version 1.15)
* All your files should end with a new line
* The first line of all your files should be exactly  ` #!/usr/bin/env python3 ` 
* A  ` README.md `  file, at the root of the folder of the project, is mandatory
* Your code should use the  ` pycodestyle `  style (version 2.4)
* All your modules should have documentation ( ` python3 -c 'print(__import__("my_module").__doc__)' ` )
* All your classes should have documentation ( ` python3 -c 'print(__import__("my_module").MyClass.__doc__)' ` )
* All your functions (inside and outside a class) should have documentation ( ` python3 -c 'print(__import__("my_module").my_function.__doc__)' `  and  ` python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)' ` )
* Unless otherwise noted, you are not allowed to import any module except  ` import numpy as np ` 
* All your files must be executable
* The length of your files will be tested using  ` wc ` 
## Tasks
### 0. Create Confusion
          mandatory         Progress vs Score  Task Body Write the function   ` def create_confusion_matrix(labels, logits): `   that creates a confusion matrix:
*  ` labels `  is a one-hot  ` numpy.ndarray `  of shape  ` (m, classes) `  containing the correct labels for each data point*  ` m `  is the number of data points
*  ` classes `  is the number of classes

*  ` logits `  is a one-hot  ` numpy.ndarray `  of shape  ` (m, classes) `  containing the predicted labels
* Returns: a confusion  ` numpy.ndarray `  of shape  ` (classes, classes) `  with row indices representing the correct labels and column indices representing the predicted labels
To accompany the following main file, you are provided with  [labels_logits.npz](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-ml/labels_logits.npz) 
 . This file does not need to be pushed to GitHub, nor will it be used to check your code.
```bash
alexa@ubuntu-xenial:0x04-error_analysis$ cat 0-main.py 
#!/usr/bin/env python3

import numpy as np
create_confusion_matrix = __import__('0-create_confusion').create_confusion_matrix

if __name__ == '__main__':
    lib = np.load('labels_logits.npz')
    labels = lib['labels']
    logits = lib['logits']

    np.set_printoptions(suppress=True)
    confusion = create_confusion_matrix(labels, logits)
    print(confusion)
    np.savez_compressed('confusion.npz', confusion=confusion)
alexa@ubuntu-xenial:0x04-error_analysis$ ./0-main.py 
[[4701.    0.   36.   17.   12.   81.   38.   11.   35.    1.]
 [   0. 5494.   36.   21.    3.   38.    7.   13.   59.    7.]
 [  64.   93. 4188.  103.  108.   17.  162.   80.  132.   21.]
 [  30.   48.  171. 4310.    2.  252.   22.   86.  128.   52.]
 [  17.   27.   35.    0. 4338.   11.   84.    9.   27.  311.]
 [  89.   57.   45.  235.   70. 3631.  123.   33.  163.   60.]
 [  47.   32.   87.    1.   64.   83. 4607.    0.   29.    1.]
 [  26.   95.   75.    7.   58.   18.    1. 4682.   13.  200.]
 [  31.  153.   82.  174.   27.  179.   64.    7. 4003.  122.]
 [  48.   37.   39.   71.  220.   49.    8.  244.   46. 4226.]]
alexa@ubuntu-xenial:0x04-error_analysis$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x04-error_analysis ` 
* File:  ` 0-create_confusion.py ` 
 Self-paced manual review  Panel footer - Controls 
### 1. Sensitivity
          mandatory         Progress vs Score  Task Body Write the function   ` def sensitivity(confusion): `   that calculates the sensitivity for each class in a confusion matrix:
*  ` confusion `  is a confusion  ` numpy.ndarray `   of shape  ` (classes, classes) `  where row indices represent the correct labels and column indices represent the predicted labels*  ` classes `  is the number of classes

* Returns: a  ` numpy.ndarray `  of shape  ` (classes,) `  containing the sensitivity of each class
```bash
alexa@ubuntu-xenial:0x04-error_analysis$ cat 1-main.py 
#!/usr/bin/env python3

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(sensitivity(confusion))
alexa@ubuntu-xenial:0x04-error_analysis$ ./1-main.py 
[0.95316302 0.96759422 0.84299517 0.84493237 0.89277629 0.80581447
 0.93051909 0.9047343  0.82672449 0.84723336]
alexa@ubuntu-xenial:0x04-error_analysis$ 

```
 ` confusion.npz `   :
* The file is coming from the output  ` 0-create_confusion.py ` 
* Or you can use this one: [confusion.npz](https://holbertonintranet.s3.amazonaws.com/uploads/misc/2021/1/3fb43ca796536b2fef793b68eca0b5adbb923fd9.npz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220524%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220524T031508Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=622b31e28b66872194acb087f53102674d78229e9f589e066db7bde91118014a) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x04-error_analysis ` 
* File:  ` 1-sensitivity.py ` 
 Self-paced manual review  Panel footer - Controls 
### 2. Precision
          mandatory         Progress vs Score  Task Body Write the function   ` def precision(confusion): `   that calculates the precision for each class in a confusion matrix:
*  ` confusion `  is a confusion  ` numpy.ndarray `   of shape  ` (classes, classes) `  where row indices represent the correct labels and column indices represent the predicted labels*  ` classes `  is the number of classes

* Returns: a  ` numpy.ndarray `  of shape  ` (classes,) `  containing the precision of each class
```bash
alexa@ubuntu-xenial:0x04-error_analysis$ cat 2-main.py 
#!/usr/bin/env python3

import numpy as np
precision = __import__('2-precision').precision

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(precision(confusion))
alexa@ubuntu-xenial:0x04-error_analysis$ ./2-main.py 
[0.93033841 0.91020543 0.87359199 0.87264628 0.88494492 0.83298922
 0.90050821 0.90648596 0.86364617 0.84503099]
alexa@ubuntu-xenial:0x04-error_analysis$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x04-error_analysis ` 
* File:  ` 2-precision.py ` 
 Self-paced manual review  Panel footer - Controls 
### 3. Specificity
          mandatory         Progress vs Score  Task Body Write the function   ` def specificity(confusion): `   that calculates the specificity for each class in a confusion matrix:
*  ` confusion `  is a confusion  ` numpy.ndarray `   of shape  ` (classes, classes) `  where row indices represent the correct labels and column indices represent the predicted labels*  ` classes `  is the number of classes

* Returns: a  ` numpy.ndarray `  of shape  ` (classes,) `  containing the specificity of each class
```bash
alexa@ubuntu-xenial:0x04-error_analysis$ cat 3-main.py 
#!/usr/bin/env python3

import numpy as np
specificity = __import__('3-specificity').specificity

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(specificity(confusion))
alexa@ubuntu-xenial:0x04-error_analysis$ ./3-main.py 
[0.99218958 0.98777131 0.9865429  0.98599078 0.98750582 0.98399789
 0.98870119 0.98922476 0.98600469 0.98278237]
alexa@ubuntu-xenial:0x04-error_analysis$

```
When there are more than two classes in a confusion matrix, specificity is not a useful metric as there are inherently more actual negatives than actual positives. It is much better to use sensitivity (recall) and precision.
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x04-error_analysis ` 
* File:  ` 3-specificity.py ` 
 Self-paced manual review  Panel footer - Controls 
### 4. F1 score
          mandatory         Progress vs Score  Task Body Write the function   ` def f1_score(confusion): `   that calculates the F1 score of a confusion matrix:
*  ` confusion `  is a confusion  ` numpy.ndarray `   of shape  ` (classes, classes) `  where row indices represent the correct labels and column indices represent the predicted labels*  ` classes `  is the number of classes

* Returns: a  ` numpy.ndarray `  of shape  ` (classes,) `  containing the F1 score of each class
* You must use  ` sensitivity = __import__('1-sensitivity').sensitivity `  and  ` precision = __import__('2-precision').precision `  create previously
```bash
alexa@ubuntu-xenial:0x04-error_analysis$ cat 4-main.py 
#!/usr/bin/env python3

import numpy as np
f1_score = __import__('4-f1_score').f1_score

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(f1_score(confusion))
alexa@ubuntu-xenial:0x04-error_analysis$ ./4-main.py 
[0.94161242 0.93802288 0.8580209  0.85856574 0.88884336 0.81917654
 0.91526771 0.90560928 0.8447821  0.84613074]
alexa@ubuntu-xenial:0x04-error_analysis$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x04-error_analysis ` 
* File:  ` 4-f1_score.py ` 
 Self-paced manual review  Panel footer - Controls 
### 5. Dealing with Error
          mandatory         Progress vs Score  Task Body In the text file   ` 5-error_handling `  , write the lettered answer to the question of how you should approach the following scenarios. Please write the answer to each scenario on a different line. If there is more than one way to approach a scenario, please use CSV formatting and place your answers in alphabetical order (ex.   ` A,B,C `  ):
Scenarios:
```bash
1. High Bias, High Variance
2. High Bias, Low Variance
3. Low Bias, High Variance
4. Low Bias, Low Variance

```
Approaches:
```bash
A. Train more
B. Try a different architecture
C. Get more data
D. Build a deeper network
E. Use regularization
F. Nothing

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x04-error_analysis ` 
* File:  ` 5-error_handling ` 
 Self-paced manual review  Panel footer - Controls 
### 6. Compare and Contrast
          mandatory         Progress vs Score  Task Body Given the following training and validation confusion matrices and the fact that human level performance has an error of ~14%, determine what the most important issue is and write the lettered answer in the file   ` 6-compare_and_contrast ` 
 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/11/03c511c109a790a30bbe.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220524%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220524T031508Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=a72d767247fc5465c2f58aae737c33e3f9e4e38147c7b98bca21e60d821e3d06) 

 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/11/8f5d5fdab6420a22471b.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220524%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220524T031508Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=b830227d88fbe82b453d7221527ac73c69af10c93f2c838af47594d79e6f6550) 

Most important issue:
 ` A. High Bias
B. High Variance
C. Nothing
 `  Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x04-error_analysis ` 
* File:  ` 6-compare_and_contrast ` 
 Self-paced manual review  Panel footer - Controls 
