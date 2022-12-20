# 0x04. Data Augmentation

### General
* What is data augmentation?
* When should you perform data augmentation?
* What are the benefits of using data augmentation?
* What are the various ways to perform data augmentation?
* How can you use ML to automate data augmentation?
## Download TF Datasets
 ` pip install --user tensorflow-datasets
 ` ## Tasks
### 0. Flip
          mandatory         Progress vs Score  Task Body Write a function   ` def flip_image(image): `   that flips an image horizontally:
*  ` image `  is a 3D  ` tf.Tensor `  containing the image to flip
* Returns the flipped image
```bash
$ cat 0-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
flip_image = __import__('0-flip').flip_image

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(0)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(flip_image(image))
    plt.show()
$ ./0-main.py

```
 ![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/10/3c70d4fb24140e583ec2cc640bba178f090c3829.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20221220%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20221220T120100Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e7265eb71617234e578ef46687935c9b2bd5d3fe9807313bc482533cc60dfff8) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` pipeline/0x04-data_augmentation ` 
* File:  ` 0-flip.py ` 
 Self-paced manual review  Panel footer - Controls 
### 1. Crop
          mandatory         Progress vs Score  Task Body Write a function   ` def crop_image(image, size): `   that performs a random crop of an image:
*  ` image `  is a 3D  ` tf.Tensor `  containing the image to crop
*  ` size `  is a tuple containing the size of the crop
* Returns the cropped image
```bash
$ cat 1-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
crop_image = __import__('1-crop').crop_image

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(1)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(crop_image(image, (200, 200, 3)))
    plt.show()
$ ./1-main.py

```
 ![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/10/e3b06484b6d2c0dcdd99a447fb2e83e2975b758a.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20221220%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20221220T120100Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=a3a5ef544d3225adc79f8081df918acc993f98f31f1a1711fb59669c065d44ea) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` pipeline/0x04-data_augmentation ` 
* File:  ` 1-crop.py ` 
 Self-paced manual review  Panel footer - Controls 
### 2. Rotate
          mandatory         Progress vs Score  Task Body Write a function   ` def rotate_image(image): `   that rotates an image by 90 degrees counter-clockwise:
*  ` image `  is a 3D  ` tf.Tensor `  containing the image to rotate
* Returns the rotated image
```bash
$ cat 2-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
rotate_image = __import__('2-rotate').rotate_image

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(2)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(rotate_image(image))
    plt.show()
$ ./2-main.py

```
 ![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/10/670106424f5b215f33b4c0f39699ae1ffe89dbb3.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20221220%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20221220T120100Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=ff55d2e78a8c9de11ba015cdd4ba89f9eaf0f48f9e71216a049303436376449a) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` pipeline/0x04-data_augmentation ` 
* File:  ` 2-rotate.py ` 
 Self-paced manual review  Panel footer - Controls 
### 3. Shear
          mandatory         Progress vs Score  Task Body Write a function   ` def shear_image(image, intensity): `   that randomly shears an image:
*  ` image `  is a 3D  ` tf.Tensor `  containing the image to shear
*  ` intensity `  is the intensity with which the image should be sheared
* Returns the sheared image
```bash
$ cat 3-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
shear_image = __import__('3-shear').shear_image

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(3)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(shear_image(image, 50))
    plt.show()
$ ./3-main.py

```
 ![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/10/cd5148646829e2f9b540cea1833d34d5f89faf2c.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20221220%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20221220T120100Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=7b1173ed6aa7a80c4218964ae5a7de739b195e5cea1a474f62e7dd853e65fb8e) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` pipeline/0x04-data_augmentation ` 
* File:  ` 3-shear.py ` 
 Self-paced manual review  Panel footer - Controls 
### 4. Brightness
          mandatory         Progress vs Score  Task Body Write a function   ` def change_brightness(image, max_delta): `   that randomly changes the brightness of an image:
*  ` image `  is a 3D  ` tf.Tensor `  containing the image to change
*  ` max_delta `  is the maximum amount the image should be brightened (or darkened)
* Returns the altered image
```bash
$ cat 4-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
change_brightness = __import__('4-brightness').change_brightness

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(4)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(change_brightness(image, 0.3))
    plt.show()
$ ./4-main.py

```
 ![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/10/3001edca791b04ccde934a44fe3095b1e544a425.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20221220%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20221220T120100Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=56a32b7f2b60ff0175666b5cc739575558460fe45a8425273d7fb243e2a13137) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` pipeline/0x04-data_augmentation ` 
* File:  ` 4-brightness.py ` 
 Self-paced manual review  Panel footer - Controls 
### 5. Hue
          mandatory         Progress vs Score  Task Body Write a function   ` def change_hue(image, delta): `   that changes the hue of an image:
*  ` image `  is a 3D  ` tf.Tensor `  containing the image to change
*  ` delta `  is the amount the hue should change
* Returns the altered image
```bash
$ cat 5-main.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
change_hue = __import__('5-hue').change_hue

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(5)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(change_hue(image, -0.5))
    plt.show()
$ ./5-main.py

```
 ![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/10/a1e9035f2000dbb5649032ac424d1ebe980e8a07.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20221220%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20221220T120100Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e109f859ca632b043ebba6f5bb302fe31b48a5d5d5c4a26918e44128e5adae46) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` pipeline/0x04-data_augmentation ` 
* File:  ` 5-hue.py ` 
 Self-paced manual review  Panel footer - Controls 
### 6. Automation
          mandatory         Progress vs Score  Task Body Write a blog post describing step by step how to perform automated data augmentation. Try to explain every step you know of, and give examples. A total beginner should understand what you have written.
* Have at least one picture, at the top of the blog post
* Publish your blog post on Medium or LinkedIn
* Share your blog post at least on LinkedIn
* Write professionally and intelligibly
* Please, remember that these blogs must be written in English to further your technical ability in a variety of settings
Remember, future employers will see your articles; take this seriously, and produce something that will be an asset to your future
When done, please add all urls below (blog post, LinkedIn post, etc.)
 Task URLs #### Add URLs here:
                Save               Github information  Self-paced manual review  Panel footer - Controls 
[Done with the mandatory tasks? Unlock 1 advanced task now!](https://intranet.hbtn.io/projects/852/unlock_optionals) 

Ready for a  manual review
