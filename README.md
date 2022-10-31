# Machine learning
[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/smith-flores/)

# Machine Learning Repository for School
This repository contains over 30 projects related to machine learning! For learning purposes, most of these implementations are done from scratch using numpy, although, for some of the projects there is a healthy amount of tensorflow, keras, pytorch, scipy, pandas,  and/or matplotlib. So far, this repo covers a very wide space of different machine learning algorithms. I'd be honored if you explore them. I'll list and link some of my favorite project folders in this readme.




# My Favorite Projects :blush: :monkey:



## [0. Object Detection Project](https://github.com/Luffy981/holbertonschool-machine_learning/tree/master/supervised_learning/0x0A-object_detection)
Object detection is a computer technology related to computer vision and image processing that deals with detecting instances of semantic objects of a certain class (such as humans, buildings, or cars) in digital images and videos. Well-researched domains of object detection include face detection and pedestrian detection. Object detection has applications in many areas of computer vision, including image retrieval and video surveillance.


### Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| tensorflow         | ^2.6.0  |
| keras              | ^2.6.0  |
| cv2                | ^4.1.0  |

### Model
Add to ./data folder
[yolo](https://intranet-projects-files.s3.amazonaws.com/holbertonschool-ml/yolo.h5)

### Tasks:
In this project I use the yolo v3 algorithm to perform object detection. There are multiple files building on the same class because of the structure of the assignment provided by Holberton school. The entire Yolo class can be found in 7-yolo.py which is linked below. The class methods are documented if you would like to know the inner workings.

#### [Yolo Class](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0A-object_detection/7-yolo.py "Yolo")

``` python
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('7-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('./data/yolo.h5', './data/coco_classes.txt', 0.6, 0.5, anchors)
    predictions, image_paths = yolo.predict('./data/yolo')
```
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0A-object_detection/images/yolo-1.png)
```
```
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0A-object_detection/images/yolo-2.png)
```
```
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0A-object_detection/images/yolo-3.png)
```
```
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0A-object_detection/images/yolo-4.png)
```
```
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0A-object_detection/images/yolo-5.png)
```
```
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0A-object_detection/images/yolo-6.png)
```
```

:small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond:

---


## [1. Transformer For Machine Translation](https://github.com/Luffy981/holbertonschool-machine_learning/tree/master/supervised_learning/0x12-transformer_apps)


### Dependencies
| Library/Framework              | Version |
| ------------------------------ | ------- |
| Python                         | ^3.7.3  |
| numpy                          | ^1.19.5 |
| matplotlib                     | ^3.4.3  |
| tensorflow                     | ^2.6.0  |
| keras                          | ^2.6.0  |
| tensorflow-datasets            | ^4.5.2  |

### Tasks

#### [Class Dataset](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x12-transformer_apps/3-dataset.py "Class Dataset")
Encodes a translation into tokens and sets up a data pipeline for the transformer model.

#### [Class Transformer](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x12-transformer_apps/5-transformer.py "Class Transformer")
Series of classes to build transformer for machine translation.

#### [Training Function](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x12-transformer_apps/5-train.py "Training Function")
``` python
#!/usr/bin/env python3
import tensorflow as tf
train_transformer = __import__('5-train').train_transformer

tf.compat.v1.set_random_seed(0)
transformer = train_transformer(4, 128, 8, 512, 32, 40, 20)
```

```
Epoch 1, batch 0: loss 10.26855754852295 accuracy 0.0
Epoch 1, batch 50: loss 10.23129940032959 accuracy 0.0009087905054911971

...

Epoch 1, batch 600: loss 7.164522647857666 accuracy 0.06743457913398743
Epoch 1, batch 650: loss 7.076988220214844 accuracy 0.07054812461137772
Epoch 1: loss 7.038494110107422 accuracy 0.07192815840244293
Epoch 2, batch 0: loss 5.177524089813232 accuracy 0.1298387050628662
Epoch 2, batch 50: loss 5.189461708068848 accuracy 0.14023463428020477

...

Epoch 2, batch 600: loss 4.870367527008057 accuracy 0.15659810602664948
Epoch 2, batch 650: loss 4.858142375946045 accuracy 0.15731287002563477
Epoch 2: loss 4.852652549743652 accuracy 0.15768977999687195

...

Epoch 20 batch 550 Loss 1.1597 Accuracy 0.3445
Epoch 20 batch 600 Loss 1.1653 Accuracy 0.3442
Epoch 20 batch 650 Loss 1.1696 Accuracy 0.3438
```

#### [Translator](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x12-transformer_apps/translator.py "Translator")
Class for translating Portuguese to english via the transformer model.
Here's a script and results trained on 50 epochs. Much better performance can be achieved with hyperparameter tuning and more training. You'll find some of the translations are pretty good.

``` python
#!/usr/bin/env python3
import tensorflow as tf
train_transformer = __import__('5-train').train_transformer
translator = __import__(translator.py)

tf.compat.v1.set_random_seed(0)
transformer, data = train_transformer(4, 128, 8, 512, 32, 40, 50, ret_data=True)
translator = Translator(data, transformer)


## Some sentences that I know get good results

sentences = [
             "este é um problema que temos que resolver.",
             "os meus vizinhos ouviram sobre esta ideia.",
             "vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.",
             "este é o primeiro livro que eu fiz."
    ]

true = [
        "this is a problem we have to solve .",
        "and my neighboring homes heard about this idea .",
        "so i 'll just share with you some stories very quickly of some magical things that have happened .",
        "this is the first book i've ever done."
]

for sen, t in zip(sentences, true):
    translator.translate(sen)
    print("Real Translation: ", t, end="\n\n")


print("\n\n\n\n\n------------------------------------------\n\n\n\n\n")
print("From Test Set:\n")

test_set = tfds.load('ted_hrlr_translate/pt_to_en', split='test', as_supervised=True)

for pt, true_translation in test_set.take(32):
    translator.translate(pt.numpy().decode('utf-8'))
    print("Real translation: ", true_translation.numpy().decode('utf-8'), end="\n\n")

```

```
Input: este é um problema que temos que resolver.
Prediction: this is a problem that we have to solve .
Real Translation:  this is a problem we have to solve .

Input: os meus vizinhos ouviram sobre esta ideia.
Prediction: my neighbors heard about this idea in the united states .
Real Translation:  and my neighboring homes heard about this idea .

Input: vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.
Prediction: so i 'm going to share with you some very quickly stories that happened to be an entire magic .
Real Translation:  so i 'll just share with you some stories very quickly of some magical things that have happened .

Input: este é o primeiro livro que eu fiz.
Prediction: this is the first book i did .
Real Translation:  this is the first book i've ever done.
```

<details>
<summary>Results from test set!</summary>

```
From Test Set:

Input: depois , podem fazer-se e testar-se previsões .
Prediction: then they can do it and test them forecast .
Real translation:  then , predictions can be made and tested .

Input: forçou a parar múltiplos laboratórios que ofereciam testes brca .
Prediction: it forced to stop multiple laboratories , they offer brand-warranry .
Real translation:  it had forced multiple labs that were offering brca testing to stop .

Input: as formigas são um exemplo clássico ; as operárias trabalham para as rainhas e vice-versa .
Prediction: re-tech is a classic ; opec donors work for ques and vice versions .
Real translation:  ants are a classic example ; workers work for queens and queens work for workers .

Input: uma em cada cem crianças no mundo nascem com uma doença cardíaca .
Prediction: one in every hundred kids in the world are born with a heart disease .
Real translation:  one of every hundred children born worldwide has some kind of heart disease .

Input: neste momento da sua vida , ela está a sofrer de sida no seu expoente máximo e tinha pneumonia .
Prediction: at this moment of life , she 's suffering from aids expose in its full and i had five-scale neutral .
Real translation:  at this point in her life , she 's suffering with full-blown aids and had pneumonia .

Input: onde estão as redes económicas ?
Prediction: where are the economic networks ?
Real translation:  where are economic networks ?

Input: ( aplausos )
Prediction: ( applause )
Real translation:  ( applause )

Input: eu usei os contentores de transporte , e também os alunos ajudaram-nos a fazer toda a mobília dos edifícios , para torná-los confortáveis​​ , dentro do orçamento do governo , mas também com a mesma a área da casa , mas muito mais confortável .
Prediction: but i had really powerful transportation , and these students helped us do all the way to make them feel all the same building .
Real translation:  i used the shipping container and also the students helped us to make all the building furniture to make them comfortable , within the budget of the government but also the area of the house is exactly the same , but much more comfortable .

Input: e , no entanto , a ironia é que a única maneira de podermos fazer qualquer coisa nova é dar um passo nessa direção .
Prediction: and yet , the irony , though , is the only way we can do anything new thing is take into that direction .
Real translation:  and yet , the irony is , the only way we can ever do anything new is to step into that space .

Input: a luz nunca desaparece .
Prediction: light never disappear .
Real translation:  the light never goes out .

Input: `` agora , `` '' tweets '' '' , quem está a `` '' tweetar '' '' ? ''
Prediction: `` now tweet , '' '' who is tweet , '' to tweet , '' now ? '' ''
Real translation:  now , tweets , who 's tweeting ?

Input: no egito , por exemplo , 91 % das mulheres que vivem hoje no egito foram mutiladas sexualmente dessa forma .
Prediction: in egypt , for example , 91 percent of women who live today in egypt today were mutually just from this way .
Real translation:  in egypt , for instance , 91 percent of all the females that live in egypt today have been sexually mutilated in that way .

Input: por outro lado , os bebés de 15 meses ficavam a olhar para ela durante muito tempo caso ela agisse como se preferisse os brócolos , como se não percebessem a situação .
Prediction: on the other side , 15 months would take look at it for a very long time and she was willing to see a broccolic dog , like they did n't notice .
Real translation:  on the other hand , 15 month-olds would stare at her for a long time if she acted as if she liked the broccoli , like they could n't figure this out .

Input: naquele momento , percebi quanta energia negativa é precisa para conservar aquele ódio dentro de nós .
Prediction: at that moment , i realized how much energy is needed to conservate that hate us in us .
Real translation:  in that instant , i realized how much negative energy it takes to hold that hatred inside of you .

Input: e a discussão é : o que é que isso interessa .
Prediction: and the argument is what it matters .
Real translation:  and the discussion is , who cares ? right ?

Input: se escolhermos um lugar e formos cuidadosos , as coisas estarão sempre lá quando as procurarmos .
Prediction: if you choose a place and you can get careful , things will always be there when you look there .
Real translation:  if you designate a spot and you 're scrupulous about it , your things will always be there when you look for them .

Input: é um museu muito popular agora , e criei um grande monumento para o governo .
Prediction: it 's a very popular museum now , and i set up a large monument to the government .
Real translation:  it 's a very popular museum now , and i created a big monument for the government .

Input: é completamente irrelevante .
Prediction: it 's completely irrele .
Real translation:  it 's completely irrelevant .

Input: todos defenderam que a sua técnica era a melhor , mas nenhum deles tinha a certeza disso e admitiram-no .
Prediction: they all advocate for their technique was better , but none of them was sure of them about it , and i admitted it .
Real translation:  `` they all argued that , `` '' my technique is the best , '' '' but none of them actually knew , and they admitted that . ''

Input: a partir daquele momento , comecei a pensar .
Prediction: from that moment , i started to think .
Real translation:  at that moment , i started thinking .

Input: mt : portanto , aqui temos a maré baixa e aqui a maré alta e no centro temos a lua .
Prediction: mt : so here we have the sea-down-down , and here is high center and on the moon .
Real translation:  mt : so over here is low tide , and over here is high tide , and in the middle is the moon .

Input: então , este jogo é muito simples .
Prediction: so this game is pretty simple .
Real translation:  beau lotto : so , this game is very simple .

Input: então , propus a reconstrução . angariei , recolhi fundos .
Prediction: so i proposed to rebuilding . i raised fundamentally , i collected funds .
Real translation:  so i proposed to rebuild . i raised — did fundraising .

Input: o que nós - betty rapacholi , minha aluna , e eu - fizemos foi dar aos bebés dois pratos de comida : um prato com brócolos crus e um com bolachas deliciosas em forma de peixinho .
Prediction: what we do — jeff atmosque , and i was making carava — on the two dig babies with colonies : a rubber and a rubber ball .
Real translation:  what we did — betty rapacholi , who was one of my students , and i — was actually to give the babies two bowls of food : one bowl of raw broccoli and one bowl of delicious goldfish crackers .

Input: é algo que nos acontece sem o nosso consentimento .
Prediction: it 's something that happens without our consent .
Real translation:  it 's something that happens to us without our consent .

Input: ardemos de paixão .
Prediction: we are burning passion .
Real translation:  we burn with passion .

Input: `` a mutilação genital é horrível , e desconhecida pelas mulheres americanas . mas , nalguns países , em muitos países , quando uma menina nasce , muito cedo na sua vida , os seus genitais são completamente removidos por um chamado `` '' cortador '' '' que , com uma lâmina de navalha , sem recurso à esterilização , corta as partes exteriores dos genitais femininos . ''
Prediction: manso , so , so , inorganic parts of the american women , but , in some countries at very young countries , when they 're born very born .
Real translation:  genital mutilation is horrible and not known by american women , but in some countries , many countries , when a child is born that 's a girl , very soon in her life , her genitals are completely cut away by a so-called cutter who has a razor blade and , in a non-sterilized way , they remove the exterior parts of a woman 's genitalia .

Input: isto significa 20 % do orçamento , do orçamento relativo a cuidados de saúde do país .
Prediction: this means 20 percent of budget budget budget to health care care .
Real translation:  that 's 20 percent of the budget , of the healthcare budget of the country .

Input: conheci-o num evento 46664 .
Prediction: i know it in a 4646 event .
Real translation:  i met him at a 46664 event .

Input: deixem-me mostrar-vos o que quero dizer .
Prediction: let me show you what i mean .
Real translation:  let me show you what i mean .

Input: acho que este é o problema .
Prediction: i think this is the problem .
Real translation:  i think this is a problem .

Input: mt : oh , 365 , o número de dias num ano , o número de dias entre cada aniversário .
Prediction: mt : oh , 36 , number 5 , the number of days ago , the number of days between every birthday .
Real translation:  mt : oh , 365 , the number of days in a year , the number of days between each birthday .
```
</details>


:small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond:

---




## 2. [QA Bot Using Bert Transformer a Universal Word Encoder for Embeddings](https://github.com/Luffy981/holbertonschool-machine_learning/tree/master/supervised_learning/0x13-qa_bot)
I can't share the corpus that was provided for this project by the school.

### Dependencies
| Library/Framework         | Version |
| ------------------------- | ------- |
| Python                    | ^3.7.3  |
| numpy                     | ^1.19.5 |
| tensorflow                | ^2.6.0  |
| tensorflow-hub            | ^0.12.0 |
| transformers              | ^4.17.0 |

### Tasks

#### [Question Answer](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x13-qa_bot/2-qa.py "Question Answer")
Answers a question, given a reference text.
``` python
#!/usr/bin/env python3

answer_loop = __import__('2-qa').answer_loop

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

answer_loop(reference)
```

```
Q: When are PLDs?
A: from 9 : 00 am to 3 : 00 pm
Q: What are Mock Interviews?
A: Sorry, I do not understand your question.
Q: What does PLD stand for?
A: peer learning days
Q: EXIT
A: Goodbye
```

#### [Semantic Search](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x13-qa_bot/3-semantic_search.py "Semantic Search")
Performs semantic search on a corpus of documents.
``` python
#!/usr/bin/env python3

semantic_search = __import__('3-semantic_search').semantic_search

## corpus_path = ZendeskArticles

print(semantic_search('corpus_path', 'When are PLDs?'))
```

```
PLD Overview
Peer Learning Days (PLDs) are a time for you and your peers to ensure that each of you understands the concepts you've encountered in your projects, as well as a time for everyone to collectively grow in technical, professional, and soft skills. During PLD, you will collaboratively review prior projects with a group of cohort peers.
PLD Basics
PLDs are mandatory on-site days from 9:00 AM to 3:00 PM. If you cannot be present or on time, you must use a PTO. 
No laptops, tablets, or screens are allowed until all tasks have been whiteboarded and understood by the entirety of your group. This time is for whiteboarding, dialogue, and active peer collaboration. After this, you may return to computers with each other to pair or group program. 
Peer Learning Days are not about sharing solutions. This doesn't empower peers with the ability to solve problems themselves! Peer learning is when you share your thought process, whether through conversation, whiteboarding, debugging, or live coding. 
When a peer has a question, rather than offering the solution, ask the following:
"How did you come to that conclusion?"
"What have you tried?"
"Did the man page give you a lead?"
"Did you think about this concept?"
Modeling this form of thinking for one another is invaluable and will strengthen your entire cohort.
Your ability to articulate your knowledge is a crucial skill and will be required to succeed during technical interviews and through your career. 
```

#### [QA Bot](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x13-qa_bot/4-qa.py "QA Bot")
Answers questions from a corpus of multiple reference texts.
``` python
#!/usr/bin/env python3

question_answer = __import__('4-qa').question_answer

## corpus_path = ZendeskArticles

question_answer('corpus_path')
```

```
Q: When are PLDs?
A: on - site days from 9 : 00 am to 3 : 00 pm
Q: What are Mock Interviews?
A: help you train for technical interviews
Q: What does PLD stand for?
A: peer learning days
Q: goodbye
A: Goodbye
```

:small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond:

---


## [3. Clustering Project](https://github.com/Luffy981/holbertonschool-machine_learning/tree/master/unsupervised_learning/0x01-clustering)

### Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| matplotlib         | ^3.4.3  |
| scipy              | ^1.7.3  |

### Tasks

#### K-Means Algorithm
k-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. k-means clustering minimizes within-cluster variances. The unsupervised k-means algorithm has a loose relationship to the k-nearest neighbor classifier, a popular supervised machine learning technique for classification that is often confused with k-means due to the name.

#### [Kmeans](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/1-kmeans.py "Kmeans")
Performs K-means on a dataset.

``` python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
kmeans = __import__('1-kmeans').kmeans

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    C, clss = kmeans(X, 5)
    print(C)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(C[:, 0], C[:, 1], s=50, marker='*', c=list(range(5)))
    plt.show()
```
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/images/k-means.png)

#### [Optimize K](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/3-optimum.py "Optimize K")
Tests for the optimum number of clusters by variance.

``` python
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
optimum_k = __import__('3-optimum').optimum_k

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    results, d_vars = optimum_k(X, kmax=10)
    plt.scatter(list(range(1, 11)), d_vars)
    plt.xlabel('Clusters')
    plt.ylabel('Delta Variance')
    plt.title('Optimizing K-means')
    plt.show()
```
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/images/optimal-k.png)

---

#### Multivariate Gaussian Mixture Model
Formally a mixture model corresponds to the mixture distribution that represents the probability distribution of observations in the overall population. Density plots are used to analyze the density of high dimensional features. If multi-model densities are observed, then it is assumed that a finite set of densities are formed by a finite set of normal mixtures. A multivariate Gaussian mixture model is used to cluster the feature data into k number of groups where k represents each state of the machine. The machine state can be a normal state, power off state, or faulty state. Each formed cluster can be diagnosed using techniques such as spectral analysis.


#### [Expectation](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/6-expectation.py "Expectation")
Calculates the expectation step in the EM algorithm for a GMM.


#### [Maximization](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/7-maximization.py "Maximization")
Calculates the maximization step in the EM algorithm for a GMM.


#### [Expectation Maximization](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/8-EM.py "Expectation Maximization")
Performs the expectation maximization for a GMM.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    k = 4
    pi, m, S, g, l = expectation_maximization(X, k, 150, verbose=True)
    clss = np.sum(g * np.arange(k).reshape(k, 1), axis=0)
    plt.scatter(X[:, 0], X[:, 1], s=20, c=clss)
    plt.scatter(m[:, 0], m[:, 1], s=50, c=np.arange(k), marker='*')
    plt.show()
```

```
Log Likelihood after 0 iterations: -652797.78665
Log Likelihood after 10 iterations: -94855.45662
Log Likelihood after 20 iterations: -94714.52057
Log Likelihood after 30 iterations: -94590.87362
Log Likelihood after 40 iterations: -94440.40559
Log Likelihood after 50 iterations: -94439.93891
Log Likelihood after 52 iterations: -94439.93889
```

![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/images/expectation-maximization.png)

---

#### [Bayesian Information Criterion](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/9-BIC.py "Bayesian Information Criterion")
Finds the best number of clusters for a GMM using the bayesian information criterion.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
BIC = __import__('9-BIC').BIC

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    best_k, best_result, l, b = BIC(X, kmin=1, kmax=10)
    ## print(best_k)
    ## print(best_result)
    ## print(l)
    ## print(b)
    x = np.arange(1, 11)
    plt.plot(x, l, 'r')
    plt.xlabel('Clusters')
    plt.ylabel('Log Likelihood')
    plt.tight_layout()
    plt.show()
    plt.plot(x, b, 'b')
    plt.xlabel('Clusters')
    plt.ylabel('BIC')
    plt.tight_layout()
    plt.show()
```

![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/images/bayes-info-criterion-1.png)
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/images/bayes-info-criterion-2.png)

---

#### [K-Means Sklearn](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/10-kmeans.py "K-Means Sklearn")
K-means clustering using sklearn.

#### [GMM Sklearn](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/11-gmm.py "GMM Sklearn")
Gaussian mixture model using sklearn.

---

#### [Agglomerative](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/12-agglomerative.py "Agglomerative")
Performs agglomerative clustering on a dataset using sklearn.cluster.hierarchy.

``` python
#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
agglomerative = __import__('12-agglomerative').agglomerative

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=100)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=100)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    clss = agglomerative(X, 100)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.show()
```
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/images/agg-hierarchy.png)
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x01-clustering/images/agg-cluster.png)


:small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond:

---


## 4. [Gaussian Processes and Bayesian Optimization](https://github.com/Luffy981/holbertonschool-machine_learning/tree/master/unsupervised_learning/0x03-hyperparameter_tuning)

Used to model a function given a small amount of data points, Gaussian process is a stochastic process (a collection of random variables indexed by time or space), such that every finite collection of those random variables has a multivariate normal distribution. Gaussian processes can be seen as an infinite-dimensional generalization of multivariate normal distributions. Gaussian processes fit the data by getting the mean over distributions and define uncertainty along the model by the variance along those sampled distributions.

Bayesian optimization is a sequential design strategy for global optimization of black-box functions that does not assume any functional forms. It is usually employed to optimize expensive-to-evaluate functions. It takes a model like a gaussian process to define a posterior and works by finding the sample location that maximizes the expected improvement of the objective function.

![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x03-hyperparameter_tuning/images/plot_posterior.jpg)

In this project, we create a Gaussian process from scratch and use it for Bayesian optimization of a blackbox function. The school project continues the class for each file, the complete classes can be found in 2-gp.py and 5-bayes_opt.py


### Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| matplotlib         | ^3.4.3  |

### Tasks

#### [Gaussian Process](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x03-hyperparameter_tuning/2-gp.py "Gaussian Process")
Class that represents a noiseless 1D Gaussian Process.

``` python
#!/usr/bin/env python3

GP = __import__('2-gp').GaussianProcess
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GP(X_init, Y_init, l=0.6, sigma_f=2)
    X_new = np.random.uniform(-np.pi, 2*np.pi, 1)
    Y_new = f(X_new)
    gp.update(X_new, Y_new)
```
---

#### [Bayesian Optimization](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x03-hyperparameter_tuning/5-bayes_opt.py "Bayesian Optimization")
Class for bayesian optimization.

``` python
#!/usr/bin/env python3

BO = __import__('5-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=0.6, sigma_f=2)
    X_opt, Y_opt = bo.optimize(50)
    print('Optimal X:', X_opt)
    print('Optimal Y:', Y_opt)
```
---

#### Plot Bayesian Optimization Aquisition Function
``` python
#!/usr/bin/env python3

BO = __import__('4-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=0.6, sigma_f=2, xsi=0.05)
    X_next, EI = bo.acquisition()

    print(EI)
    print(X_next)

    plt.scatter(X_init.reshape(-1), Y_init.reshape(-1), color='g')
    plt.plot(bo.X_s.reshape(-1), EI.reshape(-1), color='r')
    plt.axvline(x=X_next)
    plt.show()
```

![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x03-hyperparameter_tuning/images/BO_Aquisition.jpg)
---

#### [Plot Gaussian Process](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x03-hyperparameter_tuning/Plot_Gaussian_Process.py "Plot Gaussian Process")
Generates the plot at the top of the readme.

| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| matplotlib         | ^3.4.3  |
| scipy              | ^1.7.3  |

---

:small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond:

---



### Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| matplotlib         | ^3.4.3  |
| tensorflow         | ^2.6.0  |
| keras              | ^2.6.0  |
| cv2                | ^4.1.0  |
| dlib               | ^19.17.0 |


:small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond:

---




## [5. Recurrent Neural Networks](https://github.com/Luffy981/holbertonschool-machine_learning/tree/master/supervised_learning/0x0D-RNNs)
Recurrent Neural Networks (RNNs) are state of the art algorithms for sequential data. They are a powerful machine learning technique that achieves an internal memory which makes them perfectly suited for solving problems involving sequential data. RNNs are a robust type of neural network and gain their power by including an internal memory of input history to help make predictions on future time steps. RNNs are used to model data that have temporal dynamics. To understand RNNs it is important to have knowledge of a normal feed-forward neural network.

The meaningful difference between RNNs and traditional feed-forward neural networks is the way that they channel information. In a feed-forward network, the information moves in a single direction: from the input layer, through the hidden layers, to the output layer. The information never touches the same node twice, moving straight through the network. Each prediction is deterministic with respect to the network’s inputs. This is due to it’s set internal parameters which directly influences the forward pass of operations that make up the final prediction. Feed-forward networks only consider the current input and have no memory of previous inputs; thus, having no notion of order in time. The state of a feed-forward network after training could be thought of as “memory” from training, but this has nothing to do with a learned representation of temporal dynamics. They simply can’t remember anything from the past except for their internal weights learned during training. Here’s a truth to get a better picture: every unique input passed through a feed-forward network has the same output every time. In contrast, RNNs, can make a different prediction at a given time step for the same input.

In an RNN the information cycles through a loop. When it makes a prediction, it considers the current input and also what it has observed from previously received inputs. Here is an image representing the architectural difference between a Recurrent Neural Network and a traditional feed-forward neural network.

![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0D-RNNs/images/RNN.jpeg)

To put it simply, an RNN adds the immediate past to the present. Thus, an RNN has two inputs: the recurrent past and the present input. Like feed-forward networks, RNNs assign a weight matrix to it’s layer inputs, but differ from traditional networks in their application of weights to previous inputs. Their optimization procedure (backpropagation through time) also sets them apart from traditional feed-forward networks.

To understand backpropagation through time it’s important to understand traditional forward-propagation and backpropagation. These concepts are a lot to dive into, so I will try to be as brief as possible.


Forward-propagation is the prediction step of a neural network. It is simply a series of linear and non-linear operations that works on an initial input. The weights associated with each node in the layers of the network parameterize each operation. By the end of a single forward pass a prediction is made, allowing for an error to be calculated with respect to the ground truth that the model should have predicted. The error represents how bad the network’s prediction turned out.

Backpropagation (Backprop) is a machine learning optimization algorithm used for training a neural network. It is used for calculating the gradients of an error function with respect to the network’s weights. The algorithm works it’s way backwards through the layers of the network to find the partial derivatives of the errors with respect to the network’s weights. These derivatives are then used to adjust the weights of the network to decrease prediction error when training on a dataset.

In a basic sense, training a neural network is an iterative algorithm that consists of two steps: first, using forward-propagation, given an input, to make a prediction and calculate an error. Second, performing backprop to adjust the internal parameters of the model with respect to the error, intended to improve the model’s performance in the next iteration.

Backpropagation through time (BPTT) is simply the backpropagation algorithm on an unrolled RNN, allowing the optimization algorithm to take into account the temporal dynamics in the architecture. The BPTT is a necessary adjustment since the error of a given time step depends on the previous time step. Within BPTT the error from the last time step is back propagated to the first time step, while unrolling all the time steps. It’s important to note that BPTT can be computationally expensive with a large number of time steps.

There are two big obstacles that RNNs need to deal with, but it is important to first understand what a gradient is. A gradient is the partial derivative of a function with respect to its inputs. To put it simply, a gradient measures the effect that a small change to a function’s inputs has on the function's output. You can also think of a gradient as the tangent line across a function, or the slope of a function at a given point. The two big problems with standard RNNs are exploding and vanishing gradients. Exploding gradients happen when large error gradients accumulate resulting in the model assigning progressively larger value updates to the model weights. This can often be solved with gradient clipping and gradient squashing. Vanishing gradients, the bigger problem of the two, happens when gradients of layers close to the output become small, increasing the speed at which the gradients of earlier layers will approach zero. This is because, as layers and/or steps increase, the product of gradients calculated in backprop becomes the product of values much smaller than one. Standard RNNs are extremely susceptible to the vanishing gradient problem, but, fortunately, additions like Long Short-Term Memory units (LSTMs) are well suited for solving this problem.


### Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |

### Tasks
This project implements the forward pass of the following recurrent neural network architectures from scratch in numpy. There is nothing visually interesting about this project. More interesting implementations of RNNs using the keras API can be found in the root of this repository.

#### [RNN Cell](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0D-RNNs/0-rnn_cell.py "RNN Cell")
Represents a cell of a simple RNN.

``` python
#!/usr/bin/env python3

import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell

np.random.seed(0)
rnn_cell = RNNCell(10, 15, 5)
print("Wh:", rnn_cell.Wh)
print("Wy:", rnn_cell.Wy)
print("bh:", rnn_cell.bh)
print("by:", rnn_cell.by)
rnn_cell.bh = np.random.randn(1, 15)
rnn_cell.by = np.random.randn(1, 5)
h_prev = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h, y = rnn_cell.forward(h_prev, x_t)
print(h.shape)
print(h)
print(y.shape)
print(y)
```
---

#### [RNN](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0D-RNNs/1-rnn.py "RNN")
Performs forward propagation for a simple RNN.

``` python
#!/usr/bin/env python3

import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell
rnn = __import__('1-rnn').rnn

np.random.seed(1)
rnn_cell = RNNCell(10, 15, 5)
rnn_cell.bh = np.random.randn(1, 15)
rnn_cell.by = np.random.randn(1, 5)
X = np.random.randn(6, 8, 10)
h_0 = np.zeros((8, 15))
H, Y = rnn(rnn_cell, X, h_0)
print(H.shape)
print(H)
print(Y.shape)
print(Y)
```
---

#### [GRU Cell](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0D-RNNs/2-gru_cell.py "GRU Cell")
Represents a gated recurrent unit for a RNN.

``` python
#!/usr/bin/env python3

import numpy as np
GRUCell = __import__('2-gru_cell').GRUCell

np.random.seed(2)
gru_cell = GRUCell(10, 15, 5)
print("Wz:", gru_cell.Wz)
print("Wr:", gru_cell.Wr)
print("Wh:", gru_cell.Wh)
print("Wy:", gru_cell.Wy)
print("bz:", gru_cell.bz)
print("br:", gru_cell.br)
print("bh:", gru_cell.bh)
print("by:", gru_cell.by)
gru_cell.bz = np.random.randn(1, 15)
gru_cell.br = np.random.randn(1, 15)
gru_cell.bh = np.random.randn(1, 15)
gru_cell.by = np.random.randn(1, 5)
h_prev = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h, y = gru_cell.forward(h_prev, x_t)
print(h.shape)
print(h)
print(y.shape)
print(y)
```
---

#### [LSTM Cell](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0D-RNNs/3-lstm_cell.py Cell")
Represents an LSTM unit for a RNN.

``` python
#!/usr/bin/env python3

import numpy as np
LSTMCell = __import__('3-lstm_cell').LSTMCell

np.random.seed(3)
lstm_cell = LSTMCell(10, 15, 5)
print("Wf:", lstm_cell.Wf)
print("Wu:", lstm_cell.Wu)
print("Wc:", lstm_cell.Wc)
print("Wo:", lstm_cell.Wo)
print("Wy:", lstm_cell.Wy)
print("bf:", lstm_cell.bf)
print("bu:", lstm_cell.bu)
print("bc:", lstm_cell.bc)
print("bo:", lstm_cell.bo)
print("by:", lstm_cell.by)
lstm_cell.bf = np.random.randn(1, 15)
lstm_cell.bu = np.random.randn(1, 15)
lstm_cell.bc = np.random.randn(1, 15)
lstm_cell.bo = np.random.randn(1, 15)
lstm_cell.by = np.random.randn(1, 5)
h_prev = np.random.randn(8, 15)
c_prev = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h, c, y = lstm_cell.forward(h_prev, c_prev, x_t)
print(h.shape)
print(h)
print(c.shape)
print(c)
print(y.shape)
print(y)
```
---

#### [Deep RNN](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0D-RNNs/4-deep_rnn.py "Deep RNN")
Performs forward propagation for a deep RNN.

``` python
#!/usr/bin/env python3

import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell
deep_rnn = __import__('4-deep_rnn').deep_rnn

np.random.seed(1)
cell1 = RNNCell(10, 15, 1)
cell2 = RNNCell(15, 15, 1)
cell3 = RNNCell(15, 15, 5)
rnn_cells = [cell1, cell2, cell3]
for rnn_cell in rnn_cells:
    rnn_cell.bh = np.random.randn(1, 15)
cell3.by = np.random.randn(1, 5)
X = np.random.randn(6, 8, 10)
H_0 = np.zeros((3, 8, 15))
H, Y = deep_rnn(rnn_cells, X, H_0)
print(H.shape)
print(H)
print(Y.shape)
print(Y)
```
---

#### [Bidirectional Cell Forward](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0D-RNNs/5-bi_forward.py "Bidirectional Cell Forward")
BidirectionalCell represents a bidirectional cell of an RNN - forward method calculates the hidden state in the backward direction for one time step.

``` python
#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('5-bi_forward'). BidirectionalCell

np.random.seed(5)
bi_cell =  BidirectionalCell(10, 15, 5)
print("Whf:", bi_cell.Whf)
print("Whb:", bi_cell.Whb)
print("Wy:", bi_cell.Wy)
print("bhf:", bi_cell.bhf)
print("bhb:", bi_cell.bhb)
print("by:", bi_cell.by)
bi_cell.bhf = np.random.randn(1, 15)
h_prev = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h = bi_cell.forward(h_prev, x_t)
print(h.shape)
print(h)
```
---

#### [Bidirectional Cell Backward](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0D-RNNs/6-bi_backward.py Cell Backward")
BidirectionalCell represents a bidirectional cell of an RNN - backward method calculates the hidden state in the backward direction for one time step.

``` python
#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('6-bi_backward'). BidirectionalCell

np.random.seed(6)
bi_cell =  BidirectionalCell(10, 15, 5)
bi_cell.bhb = np.random.randn(1, 15)
h_next = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h = bi_cell.backward(h_next, x_t)
print(h.shape)
print(h)
```
---

#### [Bidirectional Output](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0D-RNNs/7-bi_output.py "Bidirectional Output")
Calculates all outputs for the RNN.

``` python
#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('7-bi_output'). BidirectionalCell

np.random.seed(7)
bi_cell =  BidirectionalCell(10, 15, 5)
bi_cell.by = np.random.randn(1, 5)
H = np.random.randn(6, 8, 30)
Y = bi_cell.output(H)
print(Y.shape)
print(Y)
```
---

#### [Bidirectional RNN](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x0D-RNNs/8-bi_rnn.py RNN")
Performs forward propagation for a bidirectional RNN.

``` python
#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('7-bi_output').BidirectionalCell
bi_rnn = __import__('8-bi_rnn').bi_rnn

np.random.seed(8)
bi_cell =  BidirectionalCell(10, 15, 5)
X = np.random.randn(6, 8, 10)
h_0 = np.zeros((8, 15))
h_T = np.zeros((8, 15))
H, Y = bi_rnn(bi_cell, X, h_0, h_T)
print(H.shape)
print(H)
print(Y.shape)
print(Y)
```
---

:small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond:

---


## [6. Neural Networks](https://github.com/Luffy981/holbertonschool-machine_learning/tree/master/supervised_learning/0x01-classification)
Making and training a classification neural network from scratch using numpy to classify hand written digits provided in the MNIST dataset. The assignment starts with building a single neuron for binary classification and builds up to a multi-layer perceptron network for classification of digits 0-9.

A neural network is a network or circuit of  artificial neurons or nodes. The connections of biological neurons are modeled in artificial neural networks as weights between nodes. A positive weight reflects an excitatory connection, while negative values mean inhibitory connections. All inputs are modified by a weight and summed. This activity is referred to as a linear combination. Finally, an activation function controls the amplitude of the output. For example, an acceptable range of output is usually between 0 and 1, or it could be −1 and 1.

These artificial networks may be used for predictive modeling, adaptive control and applications where they can be trained via a dataset. Self-learning resulting from experience can occur within networks, which can derive conclusions from a complex and seemingly unrelated set of information.

### Classification
In machine learning, classification refers to a predictive modeling problem where a class label is predicted for a given example of input data. Examples of classification problems include: Given an example, classify if it is spam or not. Given a handwritten character, classify it as one of the known characters.


### Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| matplotlib         | ^3.4.3  |


### Tasks


#### [Class Neuron]https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/7-neuron.py "Class Neuron")
Defines and trains a single neuron performing binary classification.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Neuron = __import__('7-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X_train.shape[0])
A, cost = neuron.train(X_train, Y_train, iterations=3000)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = neuron.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

```
Cost after 0 iterations: 4.365104944262272
Cost after 100 iterations: 0.11955134491351888

...

Cost after 3000 iterations: 0.013386353289868338
```
---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/neuron-1.png)

---

```
Train cost: 0.013386353289868338
Train accuracy: 99.66837741808132%
Dev cost: 0.010803484515167197
Dev accuracy: 99.81087470449172%
```
---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/neuron-2.png)

---

#### [Class NeuralNetwork](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/15-neural_network.py "Class NeuralNetwork")
Defines and trains a neural network with one hidden layer performing binary classification.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

NN = __import__('15-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X_train.shape[0], 3)
A, cost = nn.train(X_train, Y_train)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = nn.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

```
Cost after 0 iterations: 0.7917984405648547
Cost after 100 iterations: 0.4680930945144984

...

Cost after 5000 iterations: 0.024369225667283875
```

---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/NeuralNetwork-1.png)

---

```
Train cost: 0.024369225667283875
Train accuracy: 99.3999210422424%
Dev cost: 0.020330639788072768
Dev accuracy: 99.57446808510639%
```

---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/NeuralNetwork-2.png)

---


#### [Class DeepNeuralNetwork](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/23-deep_neural_network.py "Class DeepNeuralNetwork")
Defines and trains a deep neural network performing binary classification.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep = __import__('23-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X_train.shape[0], [5, 3, 1])
A, cost = deep.train(X_train, Y_train)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = deep.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

```
Cost after 0 iterations: 0.6958649419170609
Cost after 100 iterations: 0.6444304786060048

...

Cost after 5000 iterations: 0.011671820326008168
```

---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/DeepNeuralNetwork-1.png)

---

```
Train cost: 0.011671820326008168
Train accuracy: 99.88945913936044%
Dev cost: 0.00924955213227925
Dev accuracy: 99.95271867612293%
```

---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/DeepNeuralNetwork-2.png)

---

#### [Multiclass Classification](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/28-deep_neural_network.py "Multiclass Classification")
Updates the class DeepNeuralNetwork to perform multiclass classification.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep28 = __import__('28-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode
one_hot_decode = __import__('25-one_hot_decode').one_hot_decode

lib= np.load('../data/MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_valid_3D = lib['X_valid']
Y_valid = lib['Y_valid']
X_test_3D = lib['X_test']
Y_test = lib['Y_test']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1)).T
X_test = X_test_3D.reshape((X_test_3D.shape[0], -1)).T
Y_train_one_hot = one_hot_encode(Y_train, 10)
Y_valid_one_hot = one_hot_encode(Y_valid, 10)
Y_test_one_hot = one_hot_encode(Y_test, 10)

print('Sigmoid activation:')
deep28 = Deep28(X_train.shape[0], [5, 3, 1], activation='sig')
A_one_hot28, cost28 = deep28.train(X_train, Y_train_one_hot, iterations=100,
                                step=10, graph=False)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_train == A28) / Y_train.shape[0] * 100
print("Train cost:", cost28)
print("Train accuracy: {}%".format(accuracy28))
A_one_hot28, cost28 = deep28.evaluate(X_valid, Y_valid_one_hot)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_valid == A28) / Y_valid.shape[0] * 100
print("Validation cost:", cost28)
print("Validation accuracy: {}%".format(accuracy28))
A_one_hot28, cost28 = deep28.evaluate(X_test, Y_test_one_hot)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_test == A28) / Y_test.shape[0] * 100
print("Test cost:", cost28)
print("Test accuracy: {}%".format(accuracy28))
deep28.save('28-output')

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(A28[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.close()

print('\nTanh activaiton:')

deep28 = Deep28(X_train.shape[0], [5, 3, 1], activation='tanh')
A_one_hot28, cost28 = deep28.train(X_train, Y_train_one_hot, iterations=100,
                                step=10, graph=False)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_train == A28) / Y_train.shape[0] * 100
print("Train cost:", cost28)
print("Train accuracy: {}%".format(accuracy28))
A_one_hot28, cost28 = deep28.evaluate(X_valid, Y_valid_one_hot)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_valid == A28) / Y_valid.shape[0] * 100
print("Validation cost:", cost28)
print("Validation accuracy: {}%".format(accuracy28))
A_one_hot28, cost28 = deep28.evaluate(X_test, Y_test_one_hot)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_test == A28) / Y_test.shape[0] * 100
print("Test cost:", cost28)
print("Test accuracy: {}%".format(accuracy28))
deep28.save('28-output')

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(A28[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

```
Sigmoid activation:
Cost after 0 iterations: 0.4388904112857044
Cost after 10 iterations: 0.4377828804163359
Cost after 20 iterations: 0.43668839872612714
Cost after 30 iterations: 0.43560674736059446
Cost after 40 iterations: 0.43453771176806555
Cost after 50 iterations: 0.4334810815993252
Cost after 60 iterations: 0.43243665061046205
Cost after 70 iterations: 0.4314042165687683
Cost after 80 iterations: 0.4303835811615513
Cost after 90 iterations: 0.4293745499077264
Cost after 100 iterations: 0.42837693207206473
Train cost: 0.42837693207206456
Train accuracy: 88.442%
Validation cost: 0.39517557351173044
Validation accuracy: 89.64%
Test cost: 0.4074169894615401
Test accuracy: 89.0%
```

---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/multiclass-1.png)

---

```
Tanh activaiton:
Cost after 0 iterations: 0.1806181562229199
Cost after 10 iterations: 0.1801200954271858
Cost after 20 iterations: 0.1796242897834926
Cost after 30 iterations: 0.17913072860418564
Cost after 40 iterations: 0.1786394012066576
Cost after 50 iterations: 0.17815029691267442
Cost after 60 iterations: 0.1776634050478437
Cost after 70 iterations: 0.1771787149412177
Cost after 80 iterations: 0.1766962159250237
Cost after 90 iterations: 0.1762158973345138
Cost after 100 iterations: 0.1757377485079266
Train cost: 0.1757377485079266
Train accuracy: 95.006%
Validation cost: 0.17689309600397934
Validation accuracy: 95.13000000000001%
Test cost: 0.1809489808838737
Test accuracy: 94.77%
```

---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-classification/images/multiclass-2.png)

---

:small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond:

---


## [7. Convolutional Neural Network Project](https://github.com/Luffy981/holbertonschool-machine_learning/tree/master/supervised_learning/0x07-cnn)
Convolutional neural network (CNN, or ConvNet) is a class of Artificial Neural Network, most commonly applied to analyze visual imagery. CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons usually mean fully connected networks, that is, each neuron in one layer is connected to all neurons in the next layer. The "full connectivity" of these networks make them prone to overfitting data. Typical ways of regularization, or preventing overfitting, include: penalizing parameters during training (such as weight decay) or trimming connectivity (skipped connections, dropout, etc.) CNNs take a different approach towards regularization: they take advantage of the hierarchical pattern in data and assemble patterns of increasing complexity using smaller and simpler patterns embossed in their filters. Therefore, on a scale of connectivity and complexity, CNNs are on the lower extreme.

Convolutional networks were inspired by biological processes in that the connectivity pattern between neurons resembles the organization of the animal visual cortex. Individual cortical neurons respond to stimuli only in a restricted region of the visual field known as the receptive field. The receptive fields of different neurons partially overlap such that they cover the entire visual field.

### Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| matplotlib         | ^3.4.3  |
| tensorflow         | 1.12  |
| keras              | 1.12  |

### Tasks
Performing forward and backward prop on conv nets with numpy, as well as building lenet5 convolutional architectures in tensorflow and keras.

#### [conv_forward](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/0-conv_forward.py "conv_forward")
Performs forward propagation over a convolutional layer of a neural network.
``` python
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
    plt.imshow(A[0, :, :, 0])
    plt.show()
    plt.imshow(A[0, :, :, 1])
    plt.show()
```
Training set...

---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/images/cnn-1-trainset.png)

---
Convolutional output...

---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/images/cnn-1-cnn1.png)
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/images/cnn-1-cnn2.png)

---

#### [pool_forward](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/1-pool_forward.py "pool_forward")
Performs forward propagation over a pooling layer of a neural network.
``` python
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

    plt.imshow(X_train_c[0, :, :, 0])
    plt.show()
    plt.imshow(X_train_c[0, :, :, 1])
    plt.show()
    A = pool_forward(X_train_c, (2, 2), stride=(2, 2))
    plt.imshow(A[0, :, :, 0])
    plt.show()
    plt.imshow(A[0, :, :, 1])
    plt.show()
```
Training set...

---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/images/cnn-2-trainset1.png)
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/images/cnn-2-trainset2.png)

---
Pooling output...

---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/images/cnn-2-pool1.png)
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/images/cnn-2-pool2.png)

---

#### [conv_backward](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/2-conv_backward.py "conv_backward")
Performs back propagation over a convolutional layer of a neural network.
``` python
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
    derivatives = conv_backward(dZ, X_train_c, W, b, padding="valid")
```
---

#### [pool_backward](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/3-pool_backward.py "pool_backward")

Performs back propagation over a pooling layer of a neural network.
``` python
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
    derivatives = pool_backward(dA, X_train_c, (3, 3), stride=(3, 3))
```
---

### Lenet5 Architecture
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/images/lenet5.png)



#### [lenet5 tensorflow](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/4-lenet5.py "lenet5 tensorflow")
Modified version of the LeNet-5 architecture using tensorflow.
``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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
```

```
After 0 epochs: 3.6953983306884766 cost, 0.09554000198841095 accuracy, 3.6907131671905518 validation cost, 0.09960000216960907 validation accuracy
After 1 epochs: 0.07145008444786072 cost, 0.9778800010681152 accuracy, 0.07876613736152649 validation cost, 0.9760000109672546 validation accuracy
After 2 epochs: 0.052659813314676285 cost, 0.9831399917602539 accuracy, 0.06290869414806366 validation cost, 0.9807999730110168 validation accuracy
After 3 epochs: 0.04133499041199684 cost, 0.9859799742698669 accuracy, 0.05631111562252045 validation cost, 0.9818000197410583 validation accuracy
After 4 epochs: 0.02096478082239628 cost, 0.9934599995613098 accuracy, 0.04536684602499008 validation cost, 0.988099992275238 validation accuracy
After 5 epochs: 0.01851615309715271 cost, 0.9940599799156189 accuracy, 0.04946666955947876 validation cost, 0.9879999756813049 validation accuracy

```
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/images/lenet5-pred-1.png)

#### [lenet5 keras](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/5-lenet5.py "lenet5 keras")
Modified version of the LeNet-5 architecture using keras.

``` python
#!/usr/bin/env python3
"""
Main file
"""
## Force Seed - fix for Keras
SEED = 0
import matplotlib.pyplot as plt
import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import tensorflow.keras as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.backend.set_session(sess)

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
    Y_pred = np.argmax(Y_pred, 1)
    plt.imshow(X_valid[0])
    plt.title(str(Y_valid[0]) + ' : ' + str(Y_pred[0]))
    plt.show()
```

```
Train on 50000 samples, validate on 10000 samples
Epoch 1/5
50000/50000 [==============================] - 34s 680us/step - loss: 0.1775 - acc: 0.9459 - val_loss: 0.0764 - val_acc: 0.9785
Epoch 2/5
50000/50000 [==============================] - 33s 652us/step - loss: 0.0650 - acc: 0.9791 - val_loss: 0.0623 - val_acc: 0.9819
Epoch 3/5
50000/50000 [==============================] - 37s 737us/step - loss: 0.0471 - acc: 0.9851 - val_loss: 0.0588 - val_acc: 0.9834
Epoch 4/5
50000/50000 [==============================] - 32s 646us/step - loss: 0.0376 - acc: 0.9879 - val_loss: 0.0476 - val_acc: 0.9861
Epoch 5/5
50000/50000 [==============================] - 33s 653us/step - loss: 0.0289 - acc: 0.9907 - val_loss: 0.0509 - val_acc: 0.9870
```

![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x07-cnn/images/lenet5-pred-2.png)

---

:small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond:

---


## [8. Deep Convolutional Neural Networks](https://github.com/Luffy981/holbertonschool-machine_learning/tree/master/supervised_learning/0x08-deep_cnns)
Convolutional neural network (CNN, or ConvNet) is a class of Artificial Neural Network, most commonly applied to analyze visual imagery. CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons usually mean fully connected networks, that is, each neuron in one layer is connected to all neurons in the next layer. The "full connectivity" of these networks make them prone to overfitting data. Typical ways of regularization, or preventing overfitting, include: penalizing parameters during training (such as weight decay) or trimming connectivity (skipped connections, dropout, etc.) CNNs take a different approach towards regularization: they take advantage of the hierarchical pattern in data and assemble patterns of increasing complexity using smaller and simpler patterns embossed in their filters. Therefore, on a scale of connectivity and complexity, CNNs are on the lower extreme.

Convolutional networks were inspired by biological processes in that the connectivity pattern between neurons resembles the organization of the animal visual cortex. Individual cortical neurons respond to stimuli only in a restricted region of the visual field known as the receptive field. The receptive fields of different neurons partially overlap such that they cover the entire visual field.

### Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| tensorflow         | ^2.6.0  |
| keras              | ^2.6.0  |


### Diving into some of the best existing CNN architectures for avoiding the vanishing gradient problem.
[vanishing gradient](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
Corresponding papers linked with each task discription.

#### [inception_block](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/0-inception_block.py "inception_block")
Builds an inception block as described in [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf "Going Deeper with Convolutions")
---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/images/inception-block.png)
---
``` python
#!/usr/bin/env python3

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block

if __name__ == '__main__':
    X = K.Input(shape=(224, 224, 3))
    Y = inception_block(X, [64, 96, 128, 16, 32, 32])
    model = K.models.Model(inputs=X, outputs=Y)
    model.summary()
```

<details>
  <summary>Model Summary</summary>

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param ##     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 224, 224, 3)  0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 224, 224, 96) 384         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 224, 224, 16) 64          input_1[0][0]                    
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 224, 224, 3)  0           input_1[0][0]                    
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 224, 224, 64) 256         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 224, 224, 128 110720      conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 224, 224, 32) 12832       conv2d_3[0][0]                   
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 224, 224, 32) 128         max_pooling2d[0][0]              
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 224, 224, 256 0           conv2d[0][0]                     
                                                                 conv2d_2[0][0]                   
                                                                 conv2d_4[0][0]                   
                                                                 conv2d_5[0][0]                   
==================================================================================================
Total params: 124,384
Trainable params: 124,384
Non-trainable params: 0
__________________________________________________________________________________________________
```
</details>


#### [inception_network](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/1-inception_network.py "inception_network")
Builds an inception block as described in [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf)
---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/images/inception-network.png)
---
``` python
#!/usr/bin/env python3

import tensorflow.keras as K
inception_network = __import__('1-inception_network').inception_network

if __name__ == '__main__':
    model = inception_network()
    model.summary()
```

<details>
  <summary>Model Summary</summary>

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param ##     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 224, 224, 3)  0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 112, 112, 64) 9472        input_1[0][0]                    
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 56, 56, 64)   0           conv2d[0][0]                     
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 56, 56, 64)   4160        max_pooling2d[0][0]              
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 56, 56, 192)  110784      conv2d_1[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 28, 28, 192)  0           conv2d_2[0][0]                   
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 28, 28, 96)   18528       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 28, 28, 16)   3088        max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 28, 28, 192)  0           max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 28, 28, 64)   12352       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 28, 28, 128)  110720      conv2d_4[0][0]                   
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 28, 28, 32)   12832       conv2d_6[0][0]                   
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 28, 28, 32)   6176        max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 28, 28, 256)  0           conv2d_3[0][0]                   
                                                                 conv2d_5[0][0]                   
                                                                 conv2d_7[0][0]                   
                                                                 conv2d_8[0][0]                   
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 28, 28, 128)  32896       concatenate[0][0]                
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 28, 28, 32)   8224        concatenate[0][0]                
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 28, 28, 256)  0           concatenate[0][0]                
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 28, 28, 128)  32896       concatenate[0][0]                
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 28, 28, 192)  221376      conv2d_10[0][0]                  
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 28, 28, 96)   76896       conv2d_12[0][0]                  
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 28, 28, 64)   16448       max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 28, 28, 480)  0           conv2d_9[0][0]                   
                                                                 conv2d_11[0][0]                  
                                                                 conv2d_13[0][0]                  
                                                                 conv2d_14[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 14, 14, 480)  0           concatenate_1[0][0]              
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 14, 14, 96)   46176       max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 14, 14, 16)   7696        max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 14, 14, 480)  0           max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 14, 14, 192)  92352       max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 14, 14, 208)  179920      conv2d_16[0][0]                  
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 14, 14, 48)   19248       conv2d_18[0][0]                  
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 14, 14, 64)   30784       max_pooling2d_5[0][0]            
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 14, 14, 512)  0           conv2d_15[0][0]                  
                                                                 conv2d_17[0][0]                  
                                                                 conv2d_19[0][0]                  
                                                                 conv2d_20[0][0]                  
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 14, 14, 112)  57456       concatenate_2[0][0]              
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 14, 14, 24)   12312       concatenate_2[0][0]              
__________________________________________________________________________________________________
max_pooling2d_6 (MaxPooling2D)  (None, 14, 14, 512)  0           concatenate_2[0][0]              
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 14, 14, 160)  82080       concatenate_2[0][0]              
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 14, 14, 224)  226016      conv2d_22[0][0]                  
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 14, 14, 64)   38464       conv2d_24[0][0]                  
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 14, 14, 64)   32832       max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 14, 14, 512)  0           conv2d_21[0][0]                  
                                                                 conv2d_23[0][0]                  
                                                                 conv2d_25[0][0]                  
                                                                 conv2d_26[0][0]                  
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 14, 14, 128)  65664       concatenate_3[0][0]              
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 14, 14, 24)   12312       concatenate_3[0][0]              
__________________________________________________________________________________________________
max_pooling2d_7 (MaxPooling2D)  (None, 14, 14, 512)  0           concatenate_3[0][0]              
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 14, 14, 128)  65664       concatenate_3[0][0]              
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 14, 14, 256)  295168      conv2d_28[0][0]                  
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 14, 14, 64)   38464       conv2d_30[0][0]                  
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 14, 14, 64)   32832       max_pooling2d_7[0][0]            
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 14, 14, 512)  0           conv2d_27[0][0]                  
                                                                 conv2d_29[0][0]                  
                                                                 conv2d_31[0][0]                  
                                                                 conv2d_32[0][0]                  
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 14, 14, 144)  73872       concatenate_4[0][0]              
__________________________________________________________________________________________________
conv2d_36 (Conv2D)              (None, 14, 14, 32)   16416       concatenate_4[0][0]              
__________________________________________________________________________________________________
max_pooling2d_8 (MaxPooling2D)  (None, 14, 14, 512)  0           concatenate_4[0][0]              
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 14, 14, 112)  57456       concatenate_4[0][0]              
__________________________________________________________________________________________________
conv2d_35 (Conv2D)              (None, 14, 14, 288)  373536      conv2d_34[0][0]                  
__________________________________________________________________________________________________
conv2d_37 (Conv2D)              (None, 14, 14, 64)   51264       conv2d_36[0][0]                  
__________________________________________________________________________________________________
conv2d_38 (Conv2D)              (None, 14, 14, 64)   32832       max_pooling2d_8[0][0]            
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 14, 14, 528)  0           conv2d_33[0][0]                  
                                                                 conv2d_35[0][0]                  
                                                                 conv2d_37[0][0]                  
                                                                 conv2d_38[0][0]                  
__________________________________________________________________________________________________
conv2d_40 (Conv2D)              (None, 14, 14, 160)  84640       concatenate_5[0][0]              
__________________________________________________________________________________________________
conv2d_42 (Conv2D)              (None, 14, 14, 32)   16928       concatenate_5[0][0]              
__________________________________________________________________________________________________
max_pooling2d_9 (MaxPooling2D)  (None, 14, 14, 528)  0           concatenate_5[0][0]              
__________________________________________________________________________________________________
conv2d_39 (Conv2D)              (None, 14, 14, 256)  135424      concatenate_5[0][0]              
__________________________________________________________________________________________________
conv2d_41 (Conv2D)              (None, 14, 14, 320)  461120      conv2d_40[0][0]                  
__________________________________________________________________________________________________
conv2d_43 (Conv2D)              (None, 14, 14, 128)  102528      conv2d_42[0][0]                  
__________________________________________________________________________________________________
conv2d_44 (Conv2D)              (None, 14, 14, 128)  67712       max_pooling2d_9[0][0]            
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 14, 14, 832)  0           conv2d_39[0][0]                  
                                                                 conv2d_41[0][0]                  
                                                                 conv2d_43[0][0]                  
                                                                 conv2d_44[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_10 (MaxPooling2D) (None, 7, 7, 832)    0           concatenate_6[0][0]              
__________________________________________________________________________________________________
conv2d_46 (Conv2D)              (None, 7, 7, 160)    133280      max_pooling2d_10[0][0]           
__________________________________________________________________________________________________
conv2d_48 (Conv2D)              (None, 7, 7, 32)     26656       max_pooling2d_10[0][0]           
__________________________________________________________________________________________________
max_pooling2d_11 (MaxPooling2D) (None, 7, 7, 832)    0           max_pooling2d_10[0][0]           
__________________________________________________________________________________________________
conv2d_45 (Conv2D)              (None, 7, 7, 256)    213248      max_pooling2d_10[0][0]           
__________________________________________________________________________________________________
conv2d_47 (Conv2D)              (None, 7, 7, 320)    461120      conv2d_46[0][0]                  
__________________________________________________________________________________________________
conv2d_49 (Conv2D)              (None, 7, 7, 128)    102528      conv2d_48[0][0]                  
__________________________________________________________________________________________________
conv2d_50 (Conv2D)              (None, 7, 7, 128)    106624      max_pooling2d_11[0][0]           
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 7, 7, 832)    0           conv2d_45[0][0]                  
                                                                 conv2d_47[0][0]                  
                                                                 conv2d_49[0][0]                  
                                                                 conv2d_50[0][0]                  
__________________________________________________________________________________________________
conv2d_52 (Conv2D)              (None, 7, 7, 192)    159936      concatenate_7[0][0]              
__________________________________________________________________________________________________
conv2d_54 (Conv2D)              (None, 7, 7, 48)     39984       concatenate_7[0][0]              
__________________________________________________________________________________________________
max_pooling2d_12 (MaxPooling2D) (None, 7, 7, 832)    0           concatenate_7[0][0]              
__________________________________________________________________________________________________
conv2d_51 (Conv2D)              (None, 7, 7, 384)    319872      concatenate_7[0][0]              
__________________________________________________________________________________________________
conv2d_53 (Conv2D)              (None, 7, 7, 384)    663936      conv2d_52[0][0]                  
__________________________________________________________________________________________________
conv2d_55 (Conv2D)              (None, 7, 7, 128)    153728      conv2d_54[0][0]                  
__________________________________________________________________________________________________
conv2d_56 (Conv2D)              (None, 7, 7, 128)    106624      max_pooling2d_12[0][0]           
__________________________________________________________________________________________________
concatenate_8 (Concatenate)     (None, 7, 7, 1024)   0           conv2d_51[0][0]                  
                                                                 conv2d_53[0][0]                  
                                                                 conv2d_55[0][0]                  
                                                                 conv2d_56[0][0]                  
__________________________________________________________________________________________________
average_pooling2d (AveragePooli (None, 1, 1, 1024)   0           concatenate_8[0][0]              
__________________________________________________________________________________________________
dropout (Dropout)               (None, 1, 1, 1024)   0           average_pooling2d[0][0]          
__________________________________________________________________________________________________
dense (Dense)                   (None, 1, 1, 1000)   1025000     dropout[0][0]                    
==================================================================================================
Total params: 6,998,552
Trainable params: 6,998,552
Non-trainable params: 0
__________________________________________________________________________________________________
```
</details>

#### [identity_block](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/2-identity_block.py "identity_block")
Builds an identity block as described in [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/images/identity-block.png)
---
``` python
#!/usr/bin/env python3

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block

if __name__ == '__main__':
    X = K.Input(shape=(224, 224, 256))
    Y = identity_block(X, [64, 64, 256])
    model = K.models.Model(inputs=X, outputs=Y)
    model.summary()
```

<details>
  <summary>Model Summary</summary>

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param ##     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 224, 224, 256 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 224, 224, 64) 16448       input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 224, 224, 64) 256         conv2d[0][0]                     
__________________________________________________________________________________________________
activation (Activation)         (None, 224, 224, 64) 0           batch_normalization[0][0]        
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 224, 224, 64) 36928       activation[0][0]                 
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 224, 224, 64) 256         conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 224, 224, 64) 0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 224, 224, 256 16640       activation_1[0][0]               
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 224, 224, 256 1024        conv2d_2[0][0]                   
__________________________________________________________________________________________________
add (Add)                       (None, 224, 224, 256 0           batch_normalization_2[0][0]      
                                                                 input_1[0][0]                    
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 224, 224, 256 0           add[0][0]                        
==================================================================================================
Total params: 71,552
Trainable params: 70,784
Non-trainable params: 768
__________________________________________________________________________________________________
```
</details>

#### [projection_block](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/3-projection_block.py "projection_block")
Builds a projection block as described in [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/images/projection-block.png)
---
``` python
#!/usr/bin/env python3

import tensorflow.keras as K
projection_block = __import__('3-projection_block').projection_block

if __name__ == '__main__':
    X = K.Input(shape=(224, 224, 3))
    Y = projection_block(X, [64, 64, 256])
    model = K.models.Model(inputs=X, outputs=Y)
    model.summary()
```

<details>
  <summary>Model Summary</summary>

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param ##     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 224, 224, 3)  0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 112, 112, 64) 256         input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 112, 112, 64) 256         conv2d[0][0]                     
__________________________________________________________________________________________________
activation (Activation)         (None, 112, 112, 64) 0           batch_normalization[0][0]        
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 112, 112, 64) 36928       activation[0][0]                 
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 112, 112, 64) 256         conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 112, 112, 64) 0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 112, 112, 256 16640       activation_1[0][0]               
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 112, 112, 256 1024        input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 112, 112, 256 1024        conv2d_2[0][0]                   
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 112, 112, 256 1024        conv2d_3[0][0]                   
__________________________________________________________________________________________________
add (Add)                       (None, 112, 112, 256 0           batch_normalization_2[0][0]      
                                                                 batch_normalization_3[0][0]      
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 112, 112, 256 0           add[0][0]                        
==================================================================================================
Total params: 57,408
Trainable params: 56,128
Non-trainable params: 1,280
__________________________________________________________________________________________________
```
</details>


#### [resnet50](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/4-resnet50.py "resnet50")
Builds the ResNet-50 architecture as described in [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/images/resnet50.png)
---
``` python
#!/usr/bin/env python3

import tensorflow.keras as K
resnet50 = __import__('4-resnet50').resnet50

if __name__ == '__main__':
    model = resnet50()
    model.summary()
```

<details>
  <summary>Model Summary</summary>

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param ##     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 224, 224, 3)  0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 112, 112, 64) 9472        input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 112, 112, 64) 256         conv2d[0][0]                     
__________________________________________________________________________________________________
activation (Activation)         (None, 112, 112, 64) 0           batch_normalization[0][0]        
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 56, 56, 64)   0           activation[0][0]                 
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 56, 56, 64)   4160        max_pooling2d[0][0]              
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 56, 56, 64)   256         conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 56, 56, 64)   0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 56, 56, 64)   36928       activation_1[0][0]               
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 56, 56, 64)   256         conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 56, 56, 64)   0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 56, 56, 256)  16640       activation_2[0][0]               
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 56, 56, 256)  16640       max_pooling2d[0][0]              
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 56, 56, 256)  1024        conv2d_3[0][0]                   
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 56, 56, 256)  1024        conv2d_4[0][0]                   
__________________________________________________________________________________________________
add (Add)                       (None, 56, 56, 256)  0           batch_normalization_3[0][0]      
                                                                 batch_normalization_4[0][0]      
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 56, 56, 256)  0           add[0][0]                        
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 56, 56, 64)   16448       activation_3[0][0]               
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 56, 56, 64)   256         conv2d_5[0][0]                   
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 56, 56, 64)   0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 56, 56, 64)   36928       activation_4[0][0]               
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 56, 56, 64)   256         conv2d_6[0][0]                   
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 56, 56, 64)   0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 56, 56, 256)  16640       activation_5[0][0]               
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 56, 56, 256)  1024        conv2d_7[0][0]                   
__________________________________________________________________________________________________
add_1 (Add)                     (None, 56, 56, 256)  0           batch_normalization_7[0][0]      
                                                                 activation_3[0][0]               
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 56, 56, 256)  0           add_1[0][0]                      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 56, 56, 64)   16448       activation_6[0][0]               
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 56, 56, 64)   256         conv2d_8[0][0]                   
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 56, 56, 64)   0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 56, 56, 64)   36928       activation_7[0][0]               
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 56, 56, 64)   256         conv2d_9[0][0]                   
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 56, 56, 64)   0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 56, 56, 256)  16640       activation_8[0][0]               
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 56, 56, 256)  1024        conv2d_10[0][0]                  
__________________________________________________________________________________________________
add_2 (Add)                     (None, 56, 56, 256)  0           batch_normalization_10[0][0]     
                                                                 activation_6[0][0]               
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 56, 56, 256)  0           add_2[0][0]                      
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 28, 28, 128)  32896       activation_9[0][0]               
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 28, 28, 128)  512         conv2d_11[0][0]                  
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 28, 28, 128)  0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 28, 28, 128)  147584      activation_10[0][0]              
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 28, 28, 128)  512         conv2d_12[0][0]                  
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 28, 28, 128)  0           batch_normalization_12[0][0]     
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 28, 28, 512)  66048       activation_11[0][0]              
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 28, 28, 512)  131584      activation_9[0][0]               
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 28, 28, 512)  2048        conv2d_13[0][0]                  
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 28, 28, 512)  2048        conv2d_14[0][0]                  
__________________________________________________________________________________________________
add_3 (Add)                     (None, 28, 28, 512)  0           batch_normalization_13[0][0]     
                                                                 batch_normalization_14[0][0]     
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 28, 28, 512)  0           add_3[0][0]                      
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 28, 28, 128)  65664       activation_12[0][0]              
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 28, 28, 128)  512         conv2d_15[0][0]                  
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 28, 28, 128)  0           batch_normalization_15[0][0]     
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 28, 28, 128)  147584      activation_13[0][0]              
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 28, 28, 128)  512         conv2d_16[0][0]                  
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 28, 28, 128)  0           batch_normalization_16[0][0]     
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 28, 28, 512)  66048       activation_14[0][0]              
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 28, 28, 512)  2048        conv2d_17[0][0]                  
__________________________________________________________________________________________________
add_4 (Add)                     (None, 28, 28, 512)  0           batch_normalization_17[0][0]     
                                                                 activation_12[0][0]              
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 28, 28, 512)  0           add_4[0][0]                      
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 28, 28, 128)  65664       activation_15[0][0]              
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 28, 28, 128)  512         conv2d_18[0][0]                  
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 28, 28, 128)  0           batch_normalization_18[0][0]     
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 28, 28, 128)  147584      activation_16[0][0]              
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 28, 28, 128)  512         conv2d_19[0][0]                  
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 28, 28, 128)  0           batch_normalization_19[0][0]     
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 28, 28, 512)  66048       activation_17[0][0]              
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 28, 28, 512)  2048        conv2d_20[0][0]                  
__________________________________________________________________________________________________
add_5 (Add)                     (None, 28, 28, 512)  0           batch_normalization_20[0][0]     
                                                                 activation_15[0][0]              
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 28, 28, 512)  0           add_5[0][0]                      
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 28, 28, 128)  65664       activation_18[0][0]              
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 28, 28, 128)  512         conv2d_21[0][0]                  
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 28, 28, 128)  0           batch_normalization_21[0][0]     
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 28, 28, 128)  147584      activation_19[0][0]              
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 28, 28, 128)  512         conv2d_22[0][0]                  
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 28, 28, 128)  0           batch_normalization_22[0][0]     
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 28, 28, 512)  66048       activation_20[0][0]              
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 28, 28, 512)  2048        conv2d_23[0][0]                  
__________________________________________________________________________________________________
add_6 (Add)                     (None, 28, 28, 512)  0           batch_normalization_23[0][0]     
                                                                 activation_18[0][0]              
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 28, 28, 512)  0           add_6[0][0]                      
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 14, 14, 256)  131328      activation_21[0][0]              
__________________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, 14, 14, 256)  1024        conv2d_24[0][0]                  
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 14, 14, 256)  0           batch_normalization_24[0][0]     
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 14, 14, 256)  590080      activation_22[0][0]              
__________________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, 14, 14, 256)  1024        conv2d_25[0][0]                  
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 14, 14, 256)  0           batch_normalization_25[0][0]     
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 14, 14, 1024) 263168      activation_23[0][0]              
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 14, 14, 1024) 525312      activation_21[0][0]              
__________________________________________________________________________________________________
batch_normalization_26 (BatchNo (None, 14, 14, 1024) 4096        conv2d_26[0][0]                  
__________________________________________________________________________________________________
batch_normalization_27 (BatchNo (None, 14, 14, 1024) 4096        conv2d_27[0][0]                  
__________________________________________________________________________________________________
add_7 (Add)                     (None, 14, 14, 1024) 0           batch_normalization_26[0][0]     
                                                                 batch_normalization_27[0][0]     
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 14, 14, 1024) 0           add_7[0][0]                      
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 14, 14, 256)  262400      activation_24[0][0]              
__________________________________________________________________________________________________
batch_normalization_28 (BatchNo (None, 14, 14, 256)  1024        conv2d_28[0][0]                  
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 14, 14, 256)  0           batch_normalization_28[0][0]     
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 14, 14, 256)  590080      activation_25[0][0]              
__________________________________________________________________________________________________
batch_normalization_29 (BatchNo (None, 14, 14, 256)  1024        conv2d_29[0][0]                  
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 14, 14, 256)  0           batch_normalization_29[0][0]     
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 14, 14, 1024) 263168      activation_26[0][0]              
__________________________________________________________________________________________________
batch_normalization_30 (BatchNo (None, 14, 14, 1024) 4096        conv2d_30[0][0]                  
__________________________________________________________________________________________________
add_8 (Add)                     (None, 14, 14, 1024) 0           batch_normalization_30[0][0]     
                                                                 activation_24[0][0]              
__________________________________________________________________________________________________
activation_27 (Activation)      (None, 14, 14, 1024) 0           add_8[0][0]                      
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 14, 14, 256)  262400      activation_27[0][0]              
__________________________________________________________________________________________________
batch_normalization_31 (BatchNo (None, 14, 14, 256)  1024        conv2d_31[0][0]                  
__________________________________________________________________________________________________
activation_28 (Activation)      (None, 14, 14, 256)  0           batch_normalization_31[0][0]     
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 14, 14, 256)  590080      activation_28[0][0]              
__________________________________________________________________________________________________
batch_normalization_32 (BatchNo (None, 14, 14, 256)  1024        conv2d_32[0][0]                  
__________________________________________________________________________________________________
activation_29 (Activation)      (None, 14, 14, 256)  0           batch_normalization_32[0][0]     
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 14, 14, 1024) 263168      activation_29[0][0]              
__________________________________________________________________________________________________
batch_normalization_33 (BatchNo (None, 14, 14, 1024) 4096        conv2d_33[0][0]                  
__________________________________________________________________________________________________
add_9 (Add)                     (None, 14, 14, 1024) 0           batch_normalization_33[0][0]     
                                                                 activation_27[0][0]              
__________________________________________________________________________________________________
activation_30 (Activation)      (None, 14, 14, 1024) 0           add_9[0][0]                      
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 14, 14, 256)  262400      activation_30[0][0]              
__________________________________________________________________________________________________
batch_normalization_34 (BatchNo (None, 14, 14, 256)  1024        conv2d_34[0][0]                  
__________________________________________________________________________________________________
activation_31 (Activation)      (None, 14, 14, 256)  0           batch_normalization_34[0][0]     
__________________________________________________________________________________________________
conv2d_35 (Conv2D)              (None, 14, 14, 256)  590080      activation_31[0][0]              
__________________________________________________________________________________________________
batch_normalization_35 (BatchNo (None, 14, 14, 256)  1024        conv2d_35[0][0]                  
__________________________________________________________________________________________________
activation_32 (Activation)      (None, 14, 14, 256)  0           batch_normalization_35[0][0]     
__________________________________________________________________________________________________
conv2d_36 (Conv2D)              (None, 14, 14, 1024) 263168      activation_32[0][0]              
__________________________________________________________________________________________________
batch_normalization_36 (BatchNo (None, 14, 14, 1024) 4096        conv2d_36[0][0]                  
__________________________________________________________________________________________________
add_10 (Add)                    (None, 14, 14, 1024) 0           batch_normalization_36[0][0]     
                                                                 activation_30[0][0]              
__________________________________________________________________________________________________
activation_33 (Activation)      (None, 14, 14, 1024) 0           add_10[0][0]                     
__________________________________________________________________________________________________
conv2d_37 (Conv2D)              (None, 14, 14, 256)  262400      activation_33[0][0]              
__________________________________________________________________________________________________
batch_normalization_37 (BatchNo (None, 14, 14, 256)  1024        conv2d_37[0][0]                  
__________________________________________________________________________________________________
activation_34 (Activation)      (None, 14, 14, 256)  0           batch_normalization_37[0][0]     
__________________________________________________________________________________________________
conv2d_38 (Conv2D)              (None, 14, 14, 256)  590080      activation_34[0][0]              
__________________________________________________________________________________________________
batch_normalization_38 (BatchNo (None, 14, 14, 256)  1024        conv2d_38[0][0]                  
__________________________________________________________________________________________________
activation_35 (Activation)      (None, 14, 14, 256)  0           batch_normalization_38[0][0]     
__________________________________________________________________________________________________
conv2d_39 (Conv2D)              (None, 14, 14, 1024) 263168      activation_35[0][0]              
__________________________________________________________________________________________________
batch_normalization_39 (BatchNo (None, 14, 14, 1024) 4096        conv2d_39[0][0]                  
__________________________________________________________________________________________________
add_11 (Add)                    (None, 14, 14, 1024) 0           batch_normalization_39[0][0]     
                                                                 activation_33[0][0]              
__________________________________________________________________________________________________
activation_36 (Activation)      (None, 14, 14, 1024) 0           add_11[0][0]                     
__________________________________________________________________________________________________
conv2d_40 (Conv2D)              (None, 14, 14, 256)  262400      activation_36[0][0]              
__________________________________________________________________________________________________
batch_normalization_40 (BatchNo (None, 14, 14, 256)  1024        conv2d_40[0][0]                  
__________________________________________________________________________________________________
activation_37 (Activation)      (None, 14, 14, 256)  0           batch_normalization_40[0][0]     
__________________________________________________________________________________________________
conv2d_41 (Conv2D)              (None, 14, 14, 256)  590080      activation_37[0][0]              
__________________________________________________________________________________________________
batch_normalization_41 (BatchNo (None, 14, 14, 256)  1024        conv2d_41[0][0]                  
__________________________________________________________________________________________________
activation_38 (Activation)      (None, 14, 14, 256)  0           batch_normalization_41[0][0]     
__________________________________________________________________________________________________
conv2d_42 (Conv2D)              (None, 14, 14, 1024) 263168      activation_38[0][0]              
__________________________________________________________________________________________________
batch_normalization_42 (BatchNo (None, 14, 14, 1024) 4096        conv2d_42[0][0]                  
__________________________________________________________________________________________________
add_12 (Add)                    (None, 14, 14, 1024) 0           batch_normalization_42[0][0]     
                                                                 activation_36[0][0]              
__________________________________________________________________________________________________
activation_39 (Activation)      (None, 14, 14, 1024) 0           add_12[0][0]                     
__________________________________________________________________________________________________
conv2d_43 (Conv2D)              (None, 7, 7, 512)    524800      activation_39[0][0]              
__________________________________________________________________________________________________
batch_normalization_43 (BatchNo (None, 7, 7, 512)    2048        conv2d_43[0][0]                  
__________________________________________________________________________________________________
activation_40 (Activation)      (None, 7, 7, 512)    0           batch_normalization_43[0][0]     
__________________________________________________________________________________________________
conv2d_44 (Conv2D)              (None, 7, 7, 512)    2359808     activation_40[0][0]              
__________________________________________________________________________________________________
batch_normalization_44 (BatchNo (None, 7, 7, 512)    2048        conv2d_44[0][0]                  
__________________________________________________________________________________________________
activation_41 (Activation)      (None, 7, 7, 512)    0           batch_normalization_44[0][0]     
__________________________________________________________________________________________________
conv2d_45 (Conv2D)              (None, 7, 7, 2048)   1050624     activation_41[0][0]              
__________________________________________________________________________________________________
conv2d_46 (Conv2D)              (None, 7, 7, 2048)   2099200     activation_39[0][0]              
__________________________________________________________________________________________________
batch_normalization_45 (BatchNo (None, 7, 7, 2048)   8192        conv2d_45[0][0]                  
__________________________________________________________________________________________________
batch_normalization_46 (BatchNo (None, 7, 7, 2048)   8192        conv2d_46[0][0]                  
__________________________________________________________________________________________________
add_13 (Add)                    (None, 7, 7, 2048)   0           batch_normalization_45[0][0]     
                                                                 batch_normalization_46[0][0]     
__________________________________________________________________________________________________
activation_42 (Activation)      (None, 7, 7, 2048)   0           add_13[0][0]                     
__________________________________________________________________________________________________
conv2d_47 (Conv2D)              (None, 7, 7, 512)    1049088     activation_42[0][0]              
__________________________________________________________________________________________________
batch_normalization_47 (BatchNo (None, 7, 7, 512)    2048        conv2d_47[0][0]                  
__________________________________________________________________________________________________
activation_43 (Activation)      (None, 7, 7, 512)    0           batch_normalization_47[0][0]     
__________________________________________________________________________________________________
conv2d_48 (Conv2D)              (None, 7, 7, 512)    2359808     activation_43[0][0]              
__________________________________________________________________________________________________
batch_normalization_48 (BatchNo (None, 7, 7, 512)    2048        conv2d_48[0][0]                  
__________________________________________________________________________________________________
activation_44 (Activation)      (None, 7, 7, 512)    0           batch_normalization_48[0][0]     
__________________________________________________________________________________________________
conv2d_49 (Conv2D)              (None, 7, 7, 2048)   1050624     activation_44[0][0]              
__________________________________________________________________________________________________
batch_normalization_49 (BatchNo (None, 7, 7, 2048)   8192        conv2d_49[0][0]                  
__________________________________________________________________________________________________
add_14 (Add)                    (None, 7, 7, 2048)   0           batch_normalization_49[0][0]     
                                                                 activation_42[0][0]              
__________________________________________________________________________________________________
activation_45 (Activation)      (None, 7, 7, 2048)   0           add_14[0][0]                     
__________________________________________________________________________________________________
conv2d_50 (Conv2D)              (None, 7, 7, 512)    1049088     activation_45[0][0]              
__________________________________________________________________________________________________
batch_normalization_50 (BatchNo (None, 7, 7, 512)    2048        conv2d_50[0][0]                  
__________________________________________________________________________________________________
activation_46 (Activation)      (None, 7, 7, 512)    0           batch_normalization_50[0][0]     
__________________________________________________________________________________________________
conv2d_51 (Conv2D)              (None, 7, 7, 512)    2359808     activation_46[0][0]              
__________________________________________________________________________________________________
batch_normalization_51 (BatchNo (None, 7, 7, 512)    2048        conv2d_51[0][0]                  
__________________________________________________________________________________________________
activation_47 (Activation)      (None, 7, 7, 512)    0           batch_normalization_51[0][0]     
__________________________________________________________________________________________________
conv2d_52 (Conv2D)              (None, 7, 7, 2048)   1050624     activation_47[0][0]              
__________________________________________________________________________________________________
batch_normalization_52 (BatchNo (None, 7, 7, 2048)   8192        conv2d_52[0][0]                  
__________________________________________________________________________________________________
add_15 (Add)                    (None, 7, 7, 2048)   0           batch_normalization_52[0][0]     
                                                                 activation_45[0][0]              
__________________________________________________________________________________________________
activation_48 (Activation)      (None, 7, 7, 2048)   0           add_15[0][0]                     
__________________________________________________________________________________________________
average_pooling2d (AveragePooli (None, 1, 1, 2048)   0           activation_48[0][0]              
__________________________________________________________________________________________________
dense (Dense)                   (None, 1, 1, 1000)   2049000     average_pooling2d[0][0]          
==================================================================================================
Total params: 25,636,712
Trainable params: 25,583,592
Non-trainable params: 53,120
__________________________________________________________________________________________________
```
</details>

#### [dense_block](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/5-dense_block.py "dense_block")
Builds a dense block as described in [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/images/dense-block.png)
---
``` python
#!/usr/bin/env python3

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block

if __name__ == '__main__':
    X = K.Input(shape=(56, 56, 64))
    Y, nb_filters = dense_block(X, 64, 32, 6)
    model = K.models.Model(inputs=X, outputs=Y)
    model.summary()
    print(nb_filters)

```

<details>
  <summary>Model Summary</summary>

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param ##     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 56, 56, 64)   0                                            
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 56, 56, 64)   256         input_1[0][0]                    
__________________________________________________________________________________________________
activation (Activation)         (None, 56, 56, 64)   0           batch_normalization[0][0]        
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 56, 56, 128)  8320        activation[0][0]                 
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 56, 56, 128)  512         conv2d[0][0]                     
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 56, 56, 128)  0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 56, 56, 32)   36896       activation_1[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 56, 56, 96)   0           input_1[0][0]                    
                                                                 conv2d_1[0][0]                   
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 56, 56, 96)   384         concatenate[0][0]                
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 56, 56, 96)   0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 56, 56, 128)  12416       activation_2[0][0]               
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 56, 56, 128)  512         conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 56, 56, 128)  0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 56, 56, 32)   36896       activation_3[0][0]               
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 56, 56, 128)  0           concatenate[0][0]                
                                                                 conv2d_3[0][0]                   
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 56, 56, 128)  512         concatenate_1[0][0]              
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 56, 56, 128)  0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 56, 56, 128)  16512       activation_4[0][0]               
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 56, 56, 128)  512         conv2d_4[0][0]                   
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 56, 56, 128)  0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 56, 56, 32)   36896       activation_5[0][0]               
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 56, 56, 160)  0           concatenate_1[0][0]              
                                                                 conv2d_5[0][0]                   
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 56, 56, 160)  640         concatenate_2[0][0]              
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 56, 56, 160)  0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 56, 56, 128)  20608       activation_6[0][0]               
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 56, 56, 128)  512         conv2d_6[0][0]                   
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 56, 56, 128)  0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 56, 56, 32)   36896       activation_7[0][0]               
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 56, 56, 192)  0           concatenate_2[0][0]              
                                                                 conv2d_7[0][0]                   
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 56, 56, 192)  768         concatenate_3[0][0]              
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 56, 56, 192)  0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 56, 56, 128)  24704       activation_8[0][0]               
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 56, 56, 128)  512         conv2d_8[0][0]                   
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 56, 56, 128)  0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 56, 56, 32)   36896       activation_9[0][0]               
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 56, 56, 224)  0           concatenate_3[0][0]              
                                                                 conv2d_9[0][0]                   
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 56, 56, 224)  896         concatenate_4[0][0]              
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 56, 56, 224)  0           batch_normalization_10[0][0]     
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 56, 56, 128)  28800       activation_10[0][0]              
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 56, 56, 128)  512         conv2d_10[0][0]                  
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 56, 56, 128)  0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 56, 56, 32)   36896       activation_11[0][0]              
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 56, 56, 256)  0           concatenate_4[0][0]              
                                                                 conv2d_11[0][0]                  
==================================================================================================
Total params: 339,264
Trainable params: 336,000
Non-trainable params: 3,264
__________________________________________________________________________________________________
256
```
</details>

#### [transition_layer](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/6-transition_layer.py "transition_layer")
Builds a transition layer as described in [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)

``` python
#!/usr/bin/env python3

import tensorflow.keras as K
transition_layer = __import__('6-transition_layer').transition_layer

if __name__ == '__main__':
    X = K.Input(shape=(56, 56, 256))
    Y, nb_filters = transition_layer(X, 256, 0.5)
    model = K.models.Model(inputs=X, outputs=Y)
    model.summary()
    print(nb_filters)
```

<details>
  <summary>Model Summary</summary>

```
_________________________________________________________________
Layer (type)                 Output Shape              Param ##   
=================================================================
input_1 (InputLayer)         (None, 56, 56, 256)       0         
_________________________________________________________________
batch_normalization (BatchNo (None, 56, 56, 256)       1024      
_________________________________________________________________
activation (Activation)      (None, 56, 56, 256)       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 56, 56, 128)       32896     
_________________________________________________________________
average_pooling2d (AveragePo (None, 28, 28, 128)       0         
=================================================================
Total params: 33,920
Trainable params: 33,408
Non-trainable params: 512
_________________________________________________________________
128
```
</details>

#### [densenet121](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/7-densenet121.py "densenet121")
Builds the DenseNet-121 architecture as described in [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
---
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/supervised_learning/0x08-deep_cnns/images/dense-net.png)
---
``` python
#!/usr/bin/env python3

densenet121 = __import__('7-densenet121').densenet121

if __name__ == '__main__':
    model = densenet121(32, 0.5)
    model.summary()
```

<details>
  <summary>Model Summary</summary>

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param ##     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 224, 224, 3)  0                                            
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 224, 224, 3)  12          input_1[0][0]                    
__________________________________________________________________________________________________
activation (Activation)         (None, 224, 224, 3)  0           batch_normalization[0][0]        
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 112, 112, 64) 9472        activation[0][0]                 
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 56, 56, 64)   0           conv2d[0][0]                     
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 56, 56, 64)   256         max_pooling2d[0][0]              
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 56, 56, 64)   0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 56, 56, 128)  8320        activation_1[0][0]               
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 56, 56, 128)  512         conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 56, 56, 128)  0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 56, 56, 32)   36896       activation_2[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 56, 56, 96)   0           max_pooling2d[0][0]              
                                                                 conv2d_2[0][0]                   
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 56, 56, 96)   384         concatenate[0][0]                
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 56, 56, 96)   0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 56, 56, 128)  12416       activation_3[0][0]               
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 56, 56, 128)  512         conv2d_3[0][0]                   
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 56, 56, 128)  0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 56, 56, 32)   36896       activation_4[0][0]               
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 56, 56, 128)  0           concatenate[0][0]                
                                                                 conv2d_4[0][0]                   
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 56, 56, 128)  512         concatenate_1[0][0]              
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 56, 56, 128)  0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 56, 56, 128)  16512       activation_5[0][0]               
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 56, 56, 128)  512         conv2d_5[0][0]                   
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 56, 56, 128)  0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 56, 56, 32)   36896       activation_6[0][0]               
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 56, 56, 160)  0           concatenate_1[0][0]              
                                                                 conv2d_6[0][0]                   
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 56, 56, 160)  640         concatenate_2[0][0]              
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 56, 56, 160)  0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 56, 56, 128)  20608       activation_7[0][0]               
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 56, 56, 128)  512         conv2d_7[0][0]                   
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 56, 56, 128)  0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 56, 56, 32)   36896       activation_8[0][0]               
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 56, 56, 192)  0           concatenate_2[0][0]              
                                                                 conv2d_8[0][0]                   
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 56, 56, 192)  768         concatenate_3[0][0]              
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 56, 56, 192)  0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 56, 56, 128)  24704       activation_9[0][0]               
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 56, 56, 128)  512         conv2d_9[0][0]                   
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 56, 56, 128)  0           batch_normalization_10[0][0]     
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 56, 56, 32)   36896       activation_10[0][0]              
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 56, 56, 224)  0           concatenate_3[0][0]              
                                                                 conv2d_10[0][0]                  
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 56, 56, 224)  896         concatenate_4[0][0]              
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 56, 56, 224)  0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 56, 56, 128)  28800       activation_11[0][0]              
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 56, 56, 128)  512         conv2d_11[0][0]                  
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 56, 56, 128)  0           batch_normalization_12[0][0]     
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 56, 56, 32)   36896       activation_12[0][0]              
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 56, 56, 256)  0           concatenate_4[0][0]              
                                                                 conv2d_12[0][0]                  
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 56, 56, 256)  1024        concatenate_5[0][0]              
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 56, 56, 256)  0           batch_normalization_13[0][0]     
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 56, 56, 128)  32896       activation_13[0][0]              
__________________________________________________________________________________________________
average_pooling2d (AveragePooli (None, 28, 28, 128)  0           conv2d_13[0][0]                  
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 28, 28, 128)  512         average_pooling2d[0][0]          
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 28, 28, 128)  0           batch_normalization_14[0][0]     
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 28, 28, 128)  16512       activation_14[0][0]              
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 28, 28, 128)  512         conv2d_14[0][0]                  
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 28, 28, 128)  0           batch_normalization_15[0][0]     
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 28, 28, 32)   36896       activation_15[0][0]              
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 28, 28, 160)  0           average_pooling2d[0][0]          
                                                                 conv2d_15[0][0]                  
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 28, 28, 160)  640         concatenate_6[0][0]              
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 28, 28, 160)  0           batch_normalization_16[0][0]     
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 28, 28, 128)  20608       activation_16[0][0]              
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 28, 28, 128)  512         conv2d_16[0][0]                  
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 28, 28, 128)  0           batch_normalization_17[0][0]     
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 28, 28, 32)   36896       activation_17[0][0]              
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 28, 28, 192)  0           concatenate_6[0][0]              
                                                                 conv2d_17[0][0]                  
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 28, 28, 192)  768         concatenate_7[0][0]              
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 28, 28, 192)  0           batch_normalization_18[0][0]     
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 28, 28, 128)  24704       activation_18[0][0]              
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 28, 28, 128)  512         conv2d_18[0][0]                  
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 28, 28, 128)  0           batch_normalization_19[0][0]     
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 28, 28, 32)   36896       activation_19[0][0]              
__________________________________________________________________________________________________
concatenate_8 (Concatenate)     (None, 28, 28, 224)  0           concatenate_7[0][0]              
                                                                 conv2d_19[0][0]                  
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 28, 28, 224)  896         concatenate_8[0][0]              
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 28, 28, 224)  0           batch_normalization_20[0][0]     
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 28, 28, 128)  28800       activation_20[0][0]              
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 28, 28, 128)  512         conv2d_20[0][0]                  
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 28, 28, 128)  0           batch_normalization_21[0][0]     
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 28, 28, 32)   36896       activation_21[0][0]              
__________________________________________________________________________________________________
concatenate_9 (Concatenate)     (None, 28, 28, 256)  0           concatenate_8[0][0]              
                                                                 conv2d_21[0][0]                  
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 28, 28, 256)  1024        concatenate_9[0][0]              
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 28, 28, 256)  0           batch_normalization_22[0][0]     
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 28, 28, 128)  32896       activation_22[0][0]              
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 28, 28, 128)  512         conv2d_22[0][0]                  
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 28, 28, 128)  0           batch_normalization_23[0][0]     
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 28, 28, 32)   36896       activation_23[0][0]              
__________________________________________________________________________________________________
concatenate_10 (Concatenate)    (None, 28, 28, 288)  0           concatenate_9[0][0]              
                                                                 conv2d_23[0][0]                  
__________________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, 28, 28, 288)  1152        concatenate_10[0][0]             
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 28, 28, 288)  0           batch_normalization_24[0][0]     
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 28, 28, 128)  36992       activation_24[0][0]              
__________________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, 28, 28, 128)  512         conv2d_24[0][0]                  
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 28, 28, 128)  0           batch_normalization_25[0][0]     
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 28, 28, 32)   36896       activation_25[0][0]              
__________________________________________________________________________________________________
concatenate_11 (Concatenate)    (None, 28, 28, 320)  0           concatenate_10[0][0]             
                                                                 conv2d_25[0][0]                  
__________________________________________________________________________________________________
batch_normalization_26 (BatchNo (None, 28, 28, 320)  1280        concatenate_11[0][0]             
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 28, 28, 320)  0           batch_normalization_26[0][0]     
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 28, 28, 128)  41088       activation_26[0][0]              
__________________________________________________________________________________________________
batch_normalization_27 (BatchNo (None, 28, 28, 128)  512         conv2d_26[0][0]                  
__________________________________________________________________________________________________
activation_27 (Activation)      (None, 28, 28, 128)  0           batch_normalization_27[0][0]     
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 28, 28, 32)   36896       activation_27[0][0]              
__________________________________________________________________________________________________
concatenate_12 (Concatenate)    (None, 28, 28, 352)  0           concatenate_11[0][0]             
                                                                 conv2d_27[0][0]                  
__________________________________________________________________________________________________
batch_normalization_28 (BatchNo (None, 28, 28, 352)  1408        concatenate_12[0][0]             
__________________________________________________________________________________________________
activation_28 (Activation)      (None, 28, 28, 352)  0           batch_normalization_28[0][0]     
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 28, 28, 128)  45184       activation_28[0][0]              
__________________________________________________________________________________________________
batch_normalization_29 (BatchNo (None, 28, 28, 128)  512         conv2d_28[0][0]                  
__________________________________________________________________________________________________
activation_29 (Activation)      (None, 28, 28, 128)  0           batch_normalization_29[0][0]     
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 28, 28, 32)   36896       activation_29[0][0]              
__________________________________________________________________________________________________
concatenate_13 (Concatenate)    (None, 28, 28, 384)  0           concatenate_12[0][0]             
                                                                 conv2d_29[0][0]                  
__________________________________________________________________________________________________
batch_normalization_30 (BatchNo (None, 28, 28, 384)  1536        concatenate_13[0][0]             
__________________________________________________________________________________________________
activation_30 (Activation)      (None, 28, 28, 384)  0           batch_normalization_30[0][0]     
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 28, 28, 128)  49280       activation_30[0][0]              
__________________________________________________________________________________________________
batch_normalization_31 (BatchNo (None, 28, 28, 128)  512         conv2d_30[0][0]                  
__________________________________________________________________________________________________
activation_31 (Activation)      (None, 28, 28, 128)  0           batch_normalization_31[0][0]     
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 28, 28, 32)   36896       activation_31[0][0]              
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 28, 28, 416)  0           concatenate_13[0][0]             
                                                                 conv2d_31[0][0]                  
__________________________________________________________________________________________________
batch_normalization_32 (BatchNo (None, 28, 28, 416)  1664        concatenate_14[0][0]             
__________________________________________________________________________________________________
activation_32 (Activation)      (None, 28, 28, 416)  0           batch_normalization_32[0][0]     
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 28, 28, 128)  53376       activation_32[0][0]              
__________________________________________________________________________________________________
batch_normalization_33 (BatchNo (None, 28, 28, 128)  512         conv2d_32[0][0]                  
__________________________________________________________________________________________________
activation_33 (Activation)      (None, 28, 28, 128)  0           batch_normalization_33[0][0]     
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 28, 28, 32)   36896       activation_33[0][0]              
__________________________________________________________________________________________________
concatenate_15 (Concatenate)    (None, 28, 28, 448)  0           concatenate_14[0][0]             
                                                                 conv2d_33[0][0]                  
__________________________________________________________________________________________________
batch_normalization_34 (BatchNo (None, 28, 28, 448)  1792        concatenate_15[0][0]             
__________________________________________________________________________________________________
activation_34 (Activation)      (None, 28, 28, 448)  0           batch_normalization_34[0][0]     
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 28, 28, 128)  57472       activation_34[0][0]              
__________________________________________________________________________________________________
batch_normalization_35 (BatchNo (None, 28, 28, 128)  512         conv2d_34[0][0]                  
__________________________________________________________________________________________________
activation_35 (Activation)      (None, 28, 28, 128)  0           batch_normalization_35[0][0]     
__________________________________________________________________________________________________
conv2d_35 (Conv2D)              (None, 28, 28, 32)   36896       activation_35[0][0]              
__________________________________________________________________________________________________
concatenate_16 (Concatenate)    (None, 28, 28, 480)  0           concatenate_15[0][0]             
                                                                 conv2d_35[0][0]                  
__________________________________________________________________________________________________
batch_normalization_36 (BatchNo (None, 28, 28, 480)  1920        concatenate_16[0][0]             
__________________________________________________________________________________________________
activation_36 (Activation)      (None, 28, 28, 480)  0           batch_normalization_36[0][0]     
__________________________________________________________________________________________________
conv2d_36 (Conv2D)              (None, 28, 28, 128)  61568       activation_36[0][0]              
__________________________________________________________________________________________________
batch_normalization_37 (BatchNo (None, 28, 28, 128)  512         conv2d_36[0][0]                  
__________________________________________________________________________________________________
activation_37 (Activation)      (None, 28, 28, 128)  0           batch_normalization_37[0][0]     
__________________________________________________________________________________________________
conv2d_37 (Conv2D)              (None, 28, 28, 32)   36896       activation_37[0][0]              
__________________________________________________________________________________________________
concatenate_17 (Concatenate)    (None, 28, 28, 512)  0           concatenate_16[0][0]             
                                                                 conv2d_37[0][0]                  
__________________________________________________________________________________________________
batch_normalization_38 (BatchNo (None, 28, 28, 512)  2048        concatenate_17[0][0]             
__________________________________________________________________________________________________
activation_38 (Activation)      (None, 28, 28, 512)  0           batch_normalization_38[0][0]     
__________________________________________________________________________________________________
conv2d_38 (Conv2D)              (None, 28, 28, 256)  131328      activation_38[0][0]              
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 14, 14, 256)  0           conv2d_38[0][0]                  
__________________________________________________________________________________________________
batch_normalization_39 (BatchNo (None, 14, 14, 256)  1024        average_pooling2d_1[0][0]        
__________________________________________________________________________________________________
activation_39 (Activation)      (None, 14, 14, 256)  0           batch_normalization_39[0][0]     
__________________________________________________________________________________________________
conv2d_39 (Conv2D)              (None, 14, 14, 128)  32896       activation_39[0][0]              
__________________________________________________________________________________________________
batch_normalization_40 (BatchNo (None, 14, 14, 128)  512         conv2d_39[0][0]                  
__________________________________________________________________________________________________
activation_40 (Activation)      (None, 14, 14, 128)  0           batch_normalization_40[0][0]     
__________________________________________________________________________________________________
conv2d_40 (Conv2D)              (None, 14, 14, 32)   36896       activation_40[0][0]              
__________________________________________________________________________________________________
concatenate_18 (Concatenate)    (None, 14, 14, 288)  0           average_pooling2d_1[0][0]        
                                                                 conv2d_40[0][0]                  
__________________________________________________________________________________________________
batch_normalization_41 (BatchNo (None, 14, 14, 288)  1152        concatenate_18[0][0]             
__________________________________________________________________________________________________
activation_41 (Activation)      (None, 14, 14, 288)  0           batch_normalization_41[0][0]     
__________________________________________________________________________________________________
conv2d_41 (Conv2D)              (None, 14, 14, 128)  36992       activation_41[0][0]              
__________________________________________________________________________________________________
batch_normalization_42 (BatchNo (None, 14, 14, 128)  512         conv2d_41[0][0]                  
__________________________________________________________________________________________________
activation_42 (Activation)      (None, 14, 14, 128)  0           batch_normalization_42[0][0]     
__________________________________________________________________________________________________
conv2d_42 (Conv2D)              (None, 14, 14, 32)   36896       activation_42[0][0]              
__________________________________________________________________________________________________
concatenate_19 (Concatenate)    (None, 14, 14, 320)  0           concatenate_18[0][0]             
                                                                 conv2d_42[0][0]                  
__________________________________________________________________________________________________
batch_normalization_43 (BatchNo (None, 14, 14, 320)  1280        concatenate_19[0][0]             
__________________________________________________________________________________________________
activation_43 (Activation)      (None, 14, 14, 320)  0           batch_normalization_43[0][0]     
__________________________________________________________________________________________________
conv2d_43 (Conv2D)              (None, 14, 14, 128)  41088       activation_43[0][0]              
__________________________________________________________________________________________________
batch_normalization_44 (BatchNo (None, 14, 14, 128)  512         conv2d_43[0][0]                  
__________________________________________________________________________________________________
activation_44 (Activation)      (None, 14, 14, 128)  0           batch_normalization_44[0][0]     
__________________________________________________________________________________________________
conv2d_44 (Conv2D)              (None, 14, 14, 32)   36896       activation_44[0][0]              
__________________________________________________________________________________________________
concatenate_20 (Concatenate)    (None, 14, 14, 352)  0           concatenate_19[0][0]             
                                                                 conv2d_44[0][0]                  
__________________________________________________________________________________________________
batch_normalization_45 (BatchNo (None, 14, 14, 352)  1408        concatenate_20[0][0]             
__________________________________________________________________________________________________
activation_45 (Activation)      (None, 14, 14, 352)  0           batch_normalization_45[0][0]     
__________________________________________________________________________________________________
conv2d_45 (Conv2D)              (None, 14, 14, 128)  45184       activation_45[0][0]              
__________________________________________________________________________________________________
batch_normalization_46 (BatchNo (None, 14, 14, 128)  512         conv2d_45[0][0]                  
__________________________________________________________________________________________________
activation_46 (Activation)      (None, 14, 14, 128)  0           batch_normalization_46[0][0]     
__________________________________________________________________________________________________
conv2d_46 (Conv2D)              (None, 14, 14, 32)   36896       activation_46[0][0]              
__________________________________________________________________________________________________
concatenate_21 (Concatenate)    (None, 14, 14, 384)  0           concatenate_20[0][0]             
                                                                 conv2d_46[0][0]                  
__________________________________________________________________________________________________
batch_normalization_47 (BatchNo (None, 14, 14, 384)  1536        concatenate_21[0][0]             
__________________________________________________________________________________________________
activation_47 (Activation)      (None, 14, 14, 384)  0           batch_normalization_47[0][0]     
__________________________________________________________________________________________________
conv2d_47 (Conv2D)              (None, 14, 14, 128)  49280       activation_47[0][0]              
__________________________________________________________________________________________________
batch_normalization_48 (BatchNo (None, 14, 14, 128)  512         conv2d_47[0][0]                  
__________________________________________________________________________________________________
activation_48 (Activation)      (None, 14, 14, 128)  0           batch_normalization_48[0][0]     
__________________________________________________________________________________________________
conv2d_48 (Conv2D)              (None, 14, 14, 32)   36896       activation_48[0][0]              
__________________________________________________________________________________________________
concatenate_22 (Concatenate)    (None, 14, 14, 416)  0           concatenate_21[0][0]             
                                                                 conv2d_48[0][0]                  
__________________________________________________________________________________________________
batch_normalization_49 (BatchNo (None, 14, 14, 416)  1664        concatenate_22[0][0]             
__________________________________________________________________________________________________
activation_49 (Activation)      (None, 14, 14, 416)  0           batch_normalization_49[0][0]     
__________________________________________________________________________________________________
conv2d_49 (Conv2D)              (None, 14, 14, 128)  53376       activation_49[0][0]              
__________________________________________________________________________________________________
batch_normalization_50 (BatchNo (None, 14, 14, 128)  512         conv2d_49[0][0]                  
__________________________________________________________________________________________________
activation_50 (Activation)      (None, 14, 14, 128)  0           batch_normalization_50[0][0]     
__________________________________________________________________________________________________
conv2d_50 (Conv2D)              (None, 14, 14, 32)   36896       activation_50[0][0]              
__________________________________________________________________________________________________
concatenate_23 (Concatenate)    (None, 14, 14, 448)  0           concatenate_22[0][0]             
                                                                 conv2d_50[0][0]                  
__________________________________________________________________________________________________
batch_normalization_51 (BatchNo (None, 14, 14, 448)  1792        concatenate_23[0][0]             
__________________________________________________________________________________________________
activation_51 (Activation)      (None, 14, 14, 448)  0           batch_normalization_51[0][0]     
__________________________________________________________________________________________________
conv2d_51 (Conv2D)              (None, 14, 14, 128)  57472       activation_51[0][0]              
__________________________________________________________________________________________________
batch_normalization_52 (BatchNo (None, 14, 14, 128)  512         conv2d_51[0][0]                  
__________________________________________________________________________________________________
activation_52 (Activation)      (None, 14, 14, 128)  0           batch_normalization_52[0][0]     
__________________________________________________________________________________________________
conv2d_52 (Conv2D)              (None, 14, 14, 32)   36896       activation_52[0][0]              
__________________________________________________________________________________________________
concatenate_24 (Concatenate)    (None, 14, 14, 480)  0           concatenate_23[0][0]             
                                                                 conv2d_52[0][0]                  
__________________________________________________________________________________________________
batch_normalization_53 (BatchNo (None, 14, 14, 480)  1920        concatenate_24[0][0]             
__________________________________________________________________________________________________
activation_53 (Activation)      (None, 14, 14, 480)  0           batch_normalization_53[0][0]     
__________________________________________________________________________________________________
conv2d_53 (Conv2D)              (None, 14, 14, 128)  61568       activation_53[0][0]              
__________________________________________________________________________________________________
batch_normalization_54 (BatchNo (None, 14, 14, 128)  512         conv2d_53[0][0]                  
__________________________________________________________________________________________________
activation_54 (Activation)      (None, 14, 14, 128)  0           batch_normalization_54[0][0]     
__________________________________________________________________________________________________
conv2d_54 (Conv2D)              (None, 14, 14, 32)   36896       activation_54[0][0]              
__________________________________________________________________________________________________
concatenate_25 (Concatenate)    (None, 14, 14, 512)  0           concatenate_24[0][0]             
                                                                 conv2d_54[0][0]                  
__________________________________________________________________________________________________
batch_normalization_55 (BatchNo (None, 14, 14, 512)  2048        concatenate_25[0][0]             
__________________________________________________________________________________________________
activation_55 (Activation)      (None, 14, 14, 512)  0           batch_normalization_55[0][0]     
__________________________________________________________________________________________________
conv2d_55 (Conv2D)              (None, 14, 14, 128)  65664       activation_55[0][0]              
__________________________________________________________________________________________________
batch_normalization_56 (BatchNo (None, 14, 14, 128)  512         conv2d_55[0][0]                  
__________________________________________________________________________________________________
activation_56 (Activation)      (None, 14, 14, 128)  0           batch_normalization_56[0][0]     
__________________________________________________________________________________________________
conv2d_56 (Conv2D)              (None, 14, 14, 32)   36896       activation_56[0][0]              
__________________________________________________________________________________________________
concatenate_26 (Concatenate)    (None, 14, 14, 544)  0           concatenate_25[0][0]             
                                                                 conv2d_56[0][0]                  
__________________________________________________________________________________________________
batch_normalization_57 (BatchNo (None, 14, 14, 544)  2176        concatenate_26[0][0]             
__________________________________________________________________________________________________
activation_57 (Activation)      (None, 14, 14, 544)  0           batch_normalization_57[0][0]     
__________________________________________________________________________________________________
conv2d_57 (Conv2D)              (None, 14, 14, 128)  69760       activation_57[0][0]              
__________________________________________________________________________________________________
batch_normalization_58 (BatchNo (None, 14, 14, 128)  512         conv2d_57[0][0]                  
__________________________________________________________________________________________________
activation_58 (Activation)      (None, 14, 14, 128)  0           batch_normalization_58[0][0]     
__________________________________________________________________________________________________
conv2d_58 (Conv2D)              (None, 14, 14, 32)   36896       activation_58[0][0]              
__________________________________________________________________________________________________
concatenate_27 (Concatenate)    (None, 14, 14, 576)  0           concatenate_26[0][0]             
                                                                 conv2d_58[0][0]                  
__________________________________________________________________________________________________
batch_normalization_59 (BatchNo (None, 14, 14, 576)  2304        concatenate_27[0][0]             
__________________________________________________________________________________________________
activation_59 (Activation)      (None, 14, 14, 576)  0           batch_normalization_59[0][0]     
__________________________________________________________________________________________________
conv2d_59 (Conv2D)              (None, 14, 14, 128)  73856       activation_59[0][0]              
__________________________________________________________________________________________________
batch_normalization_60 (BatchNo (None, 14, 14, 128)  512         conv2d_59[0][0]                  
__________________________________________________________________________________________________
activation_60 (Activation)      (None, 14, 14, 128)  0           batch_normalization_60[0][0]     
__________________________________________________________________________________________________
conv2d_60 (Conv2D)              (None, 14, 14, 32)   36896       activation_60[0][0]              
__________________________________________________________________________________________________
concatenate_28 (Concatenate)    (None, 14, 14, 608)  0           concatenate_27[0][0]             
                                                                 conv2d_60[0][0]                  
__________________________________________________________________________________________________
batch_normalization_61 (BatchNo (None, 14, 14, 608)  2432        concatenate_28[0][0]             
__________________________________________________________________________________________________
activation_61 (Activation)      (None, 14, 14, 608)  0           batch_normalization_61[0][0]     
__________________________________________________________________________________________________
conv2d_61 (Conv2D)              (None, 14, 14, 128)  77952       activation_61[0][0]              
__________________________________________________________________________________________________
batch_normalization_62 (BatchNo (None, 14, 14, 128)  512         conv2d_61[0][0]                  
__________________________________________________________________________________________________
activation_62 (Activation)      (None, 14, 14, 128)  0           batch_normalization_62[0][0]     
__________________________________________________________________________________________________
conv2d_62 (Conv2D)              (None, 14, 14, 32)   36896       activation_62[0][0]              
__________________________________________________________________________________________________
concatenate_29 (Concatenate)    (None, 14, 14, 640)  0           concatenate_28[0][0]             
                                                                 conv2d_62[0][0]                  
__________________________________________________________________________________________________
batch_normalization_63 (BatchNo (None, 14, 14, 640)  2560        concatenate_29[0][0]             
__________________________________________________________________________________________________
activation_63 (Activation)      (None, 14, 14, 640)  0           batch_normalization_63[0][0]     
__________________________________________________________________________________________________
conv2d_63 (Conv2D)              (None, 14, 14, 128)  82048       activation_63[0][0]              
__________________________________________________________________________________________________
batch_normalization_64 (BatchNo (None, 14, 14, 128)  512         conv2d_63[0][0]                  
__________________________________________________________________________________________________
activation_64 (Activation)      (None, 14, 14, 128)  0           batch_normalization_64[0][0]     
__________________________________________________________________________________________________
conv2d_64 (Conv2D)              (None, 14, 14, 32)   36896       activation_64[0][0]              
__________________________________________________________________________________________________
concatenate_30 (Concatenate)    (None, 14, 14, 672)  0           concatenate_29[0][0]             
                                                                 conv2d_64[0][0]                  
__________________________________________________________________________________________________
batch_normalization_65 (BatchNo (None, 14, 14, 672)  2688        concatenate_30[0][0]             
__________________________________________________________________________________________________
activation_65 (Activation)      (None, 14, 14, 672)  0           batch_normalization_65[0][0]     
__________________________________________________________________________________________________
conv2d_65 (Conv2D)              (None, 14, 14, 128)  86144       activation_65[0][0]              
__________________________________________________________________________________________________
batch_normalization_66 (BatchNo (None, 14, 14, 128)  512         conv2d_65[0][0]                  
__________________________________________________________________________________________________
activation_66 (Activation)      (None, 14, 14, 128)  0           batch_normalization_66[0][0]     
__________________________________________________________________________________________________
conv2d_66 (Conv2D)              (None, 14, 14, 32)   36896       activation_66[0][0]              
__________________________________________________________________________________________________
concatenate_31 (Concatenate)    (None, 14, 14, 704)  0           concatenate_30[0][0]             
                                                                 conv2d_66[0][0]                  
__________________________________________________________________________________________________
batch_normalization_67 (BatchNo (None, 14, 14, 704)  2816        concatenate_31[0][0]             
__________________________________________________________________________________________________
activation_67 (Activation)      (None, 14, 14, 704)  0           batch_normalization_67[0][0]     
__________________________________________________________________________________________________
conv2d_67 (Conv2D)              (None, 14, 14, 128)  90240       activation_67[0][0]              
__________________________________________________________________________________________________
batch_normalization_68 (BatchNo (None, 14, 14, 128)  512         conv2d_67[0][0]                  
__________________________________________________________________________________________________
activation_68 (Activation)      (None, 14, 14, 128)  0           batch_normalization_68[0][0]     
__________________________________________________________________________________________________
conv2d_68 (Conv2D)              (None, 14, 14, 32)   36896       activation_68[0][0]              
__________________________________________________________________________________________________
concatenate_32 (Concatenate)    (None, 14, 14, 736)  0           concatenate_31[0][0]             
                                                                 conv2d_68[0][0]                  
__________________________________________________________________________________________________
batch_normalization_69 (BatchNo (None, 14, 14, 736)  2944        concatenate_32[0][0]             
__________________________________________________________________________________________________
activation_69 (Activation)      (None, 14, 14, 736)  0           batch_normalization_69[0][0]     
__________________________________________________________________________________________________
conv2d_69 (Conv2D)              (None, 14, 14, 128)  94336       activation_69[0][0]              
__________________________________________________________________________________________________
batch_normalization_70 (BatchNo (None, 14, 14, 128)  512         conv2d_69[0][0]                  
__________________________________________________________________________________________________
activation_70 (Activation)      (None, 14, 14, 128)  0           batch_normalization_70[0][0]     
__________________________________________________________________________________________________
conv2d_70 (Conv2D)              (None, 14, 14, 32)   36896       activation_70[0][0]              
__________________________________________________________________________________________________
concatenate_33 (Concatenate)    (None, 14, 14, 768)  0           concatenate_32[0][0]             
                                                                 conv2d_70[0][0]                  
__________________________________________________________________________________________________
batch_normalization_71 (BatchNo (None, 14, 14, 768)  3072        concatenate_33[0][0]             
__________________________________________________________________________________________________
activation_71 (Activation)      (None, 14, 14, 768)  0           batch_normalization_71[0][0]     
__________________________________________________________________________________________________
conv2d_71 (Conv2D)              (None, 14, 14, 128)  98432       activation_71[0][0]              
__________________________________________________________________________________________________
batch_normalization_72 (BatchNo (None, 14, 14, 128)  512         conv2d_71[0][0]                  
__________________________________________________________________________________________________
activation_72 (Activation)      (None, 14, 14, 128)  0           batch_normalization_72[0][0]     
__________________________________________________________________________________________________
conv2d_72 (Conv2D)              (None, 14, 14, 32)   36896       activation_72[0][0]              
__________________________________________________________________________________________________
concatenate_34 (Concatenate)    (None, 14, 14, 800)  0           concatenate_33[0][0]             
                                                                 conv2d_72[0][0]                  
__________________________________________________________________________________________________
batch_normalization_73 (BatchNo (None, 14, 14, 800)  3200        concatenate_34[0][0]             
__________________________________________________________________________________________________
activation_73 (Activation)      (None, 14, 14, 800)  0           batch_normalization_73[0][0]     
__________________________________________________________________________________________________
conv2d_73 (Conv2D)              (None, 14, 14, 128)  102528      activation_73[0][0]              
__________________________________________________________________________________________________
batch_normalization_74 (BatchNo (None, 14, 14, 128)  512         conv2d_73[0][0]                  
__________________________________________________________________________________________________
activation_74 (Activation)      (None, 14, 14, 128)  0           batch_normalization_74[0][0]     
__________________________________________________________________________________________________
conv2d_74 (Conv2D)              (None, 14, 14, 32)   36896       activation_74[0][0]              
__________________________________________________________________________________________________
concatenate_35 (Concatenate)    (None, 14, 14, 832)  0           concatenate_34[0][0]             
                                                                 conv2d_74[0][0]                  
__________________________________________________________________________________________________
batch_normalization_75 (BatchNo (None, 14, 14, 832)  3328        concatenate_35[0][0]             
__________________________________________________________________________________________________
activation_75 (Activation)      (None, 14, 14, 832)  0           batch_normalization_75[0][0]     
__________________________________________________________________________________________________
conv2d_75 (Conv2D)              (None, 14, 14, 128)  106624      activation_75[0][0]              
__________________________________________________________________________________________________
batch_normalization_76 (BatchNo (None, 14, 14, 128)  512         conv2d_75[0][0]                  
__________________________________________________________________________________________________
activation_76 (Activation)      (None, 14, 14, 128)  0           batch_normalization_76[0][0]     
__________________________________________________________________________________________________
conv2d_76 (Conv2D)              (None, 14, 14, 32)   36896       activation_76[0][0]              
__________________________________________________________________________________________________
concatenate_36 (Concatenate)    (None, 14, 14, 864)  0           concatenate_35[0][0]             
                                                                 conv2d_76[0][0]                  
__________________________________________________________________________________________________
batch_normalization_77 (BatchNo (None, 14, 14, 864)  3456        concatenate_36[0][0]             
__________________________________________________________________________________________________
activation_77 (Activation)      (None, 14, 14, 864)  0           batch_normalization_77[0][0]     
__________________________________________________________________________________________________
conv2d_77 (Conv2D)              (None, 14, 14, 128)  110720      activation_77[0][0]              
__________________________________________________________________________________________________
batch_normalization_78 (BatchNo (None, 14, 14, 128)  512         conv2d_77[0][0]                  
__________________________________________________________________________________________________
activation_78 (Activation)      (None, 14, 14, 128)  0           batch_normalization_78[0][0]     
__________________________________________________________________________________________________
conv2d_78 (Conv2D)              (None, 14, 14, 32)   36896       activation_78[0][0]              
__________________________________________________________________________________________________
concatenate_37 (Concatenate)    (None, 14, 14, 896)  0           concatenate_36[0][0]             
                                                                 conv2d_78[0][0]                  
__________________________________________________________________________________________________
batch_normalization_79 (BatchNo (None, 14, 14, 896)  3584        concatenate_37[0][0]             
__________________________________________________________________________________________________
activation_79 (Activation)      (None, 14, 14, 896)  0           batch_normalization_79[0][0]     
__________________________________________________________________________________________________
conv2d_79 (Conv2D)              (None, 14, 14, 128)  114816      activation_79[0][0]              
__________________________________________________________________________________________________
batch_normalization_80 (BatchNo (None, 14, 14, 128)  512         conv2d_79[0][0]                  
__________________________________________________________________________________________________
activation_80 (Activation)      (None, 14, 14, 128)  0           batch_normalization_80[0][0]     
__________________________________________________________________________________________________
conv2d_80 (Conv2D)              (None, 14, 14, 32)   36896       activation_80[0][0]              
__________________________________________________________________________________________________
concatenate_38 (Concatenate)    (None, 14, 14, 928)  0           concatenate_37[0][0]             
                                                                 conv2d_80[0][0]                  
__________________________________________________________________________________________________
batch_normalization_81 (BatchNo (None, 14, 14, 928)  3712        concatenate_38[0][0]             
__________________________________________________________________________________________________
activation_81 (Activation)      (None, 14, 14, 928)  0           batch_normalization_81[0][0]     
__________________________________________________________________________________________________
conv2d_81 (Conv2D)              (None, 14, 14, 128)  118912      activation_81[0][0]              
__________________________________________________________________________________________________
batch_normalization_82 (BatchNo (None, 14, 14, 128)  512         conv2d_81[0][0]                  
__________________________________________________________________________________________________
activation_82 (Activation)      (None, 14, 14, 128)  0           batch_normalization_82[0][0]     
__________________________________________________________________________________________________
conv2d_82 (Conv2D)              (None, 14, 14, 32)   36896       activation_82[0][0]              
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 14, 14, 960)  0           concatenate_38[0][0]             
                                                                 conv2d_82[0][0]                  
__________________________________________________________________________________________________
batch_normalization_83 (BatchNo (None, 14, 14, 960)  3840        concatenate_39[0][0]             
__________________________________________________________________________________________________
activation_83 (Activation)      (None, 14, 14, 960)  0           batch_normalization_83[0][0]     
__________________________________________________________________________________________________
conv2d_83 (Conv2D)              (None, 14, 14, 128)  123008      activation_83[0][0]              
__________________________________________________________________________________________________
batch_normalization_84 (BatchNo (None, 14, 14, 128)  512         conv2d_83[0][0]                  
__________________________________________________________________________________________________
activation_84 (Activation)      (None, 14, 14, 128)  0           batch_normalization_84[0][0]     
__________________________________________________________________________________________________
conv2d_84 (Conv2D)              (None, 14, 14, 32)   36896       activation_84[0][0]              
__________________________________________________________________________________________________
concatenate_40 (Concatenate)    (None, 14, 14, 992)  0           concatenate_39[0][0]             
                                                                 conv2d_84[0][0]                  
__________________________________________________________________________________________________
batch_normalization_85 (BatchNo (None, 14, 14, 992)  3968        concatenate_40[0][0]             
__________________________________________________________________________________________________
activation_85 (Activation)      (None, 14, 14, 992)  0           batch_normalization_85[0][0]     
__________________________________________________________________________________________________
conv2d_85 (Conv2D)              (None, 14, 14, 128)  127104      activation_85[0][0]              
__________________________________________________________________________________________________
batch_normalization_86 (BatchNo (None, 14, 14, 128)  512         conv2d_85[0][0]                  
__________________________________________________________________________________________________
activation_86 (Activation)      (None, 14, 14, 128)  0           batch_normalization_86[0][0]     
__________________________________________________________________________________________________
conv2d_86 (Conv2D)              (None, 14, 14, 32)   36896       activation_86[0][0]              
__________________________________________________________________________________________________
concatenate_41 (Concatenate)    (None, 14, 14, 1024) 0           concatenate_40[0][0]             
                                                                 conv2d_86[0][0]                  
__________________________________________________________________________________________________
batch_normalization_87 (BatchNo (None, 14, 14, 1024) 4096        concatenate_41[0][0]             
__________________________________________________________________________________________________
activation_87 (Activation)      (None, 14, 14, 1024) 0           batch_normalization_87[0][0]     
__________________________________________________________________________________________________
conv2d_87 (Conv2D)              (None, 14, 14, 512)  524800      activation_87[0][0]              
__________________________________________________________________________________________________
average_pooling2d_2 (AveragePoo (None, 7, 7, 512)    0           conv2d_87[0][0]                  
__________________________________________________________________________________________________
batch_normalization_88 (BatchNo (None, 7, 7, 512)    2048        average_pooling2d_2[0][0]        
__________________________________________________________________________________________________
activation_88 (Activation)      (None, 7, 7, 512)    0           batch_normalization_88[0][0]     
__________________________________________________________________________________________________
conv2d_88 (Conv2D)              (None, 7, 7, 128)    65664       activation_88[0][0]              
__________________________________________________________________________________________________
batch_normalization_89 (BatchNo (None, 7, 7, 128)    512         conv2d_88[0][0]                  
__________________________________________________________________________________________________
activation_89 (Activation)      (None, 7, 7, 128)    0           batch_normalization_89[0][0]     
__________________________________________________________________________________________________
conv2d_89 (Conv2D)              (None, 7, 7, 32)     36896       activation_89[0][0]              
__________________________________________________________________________________________________
concatenate_42 (Concatenate)    (None, 7, 7, 544)    0           average_pooling2d_2[0][0]        
                                                                 conv2d_89[0][0]                  
__________________________________________________________________________________________________
batch_normalization_90 (BatchNo (None, 7, 7, 544)    2176        concatenate_42[0][0]             
__________________________________________________________________________________________________
activation_90 (Activation)      (None, 7, 7, 544)    0           batch_normalization_90[0][0]     
__________________________________________________________________________________________________
conv2d_90 (Conv2D)              (None, 7, 7, 128)    69760       activation_90[0][0]              
__________________________________________________________________________________________________
batch_normalization_91 (BatchNo (None, 7, 7, 128)    512         conv2d_90[0][0]                  
__________________________________________________________________________________________________
activation_91 (Activation)      (None, 7, 7, 128)    0           batch_normalization_91[0][0]     
__________________________________________________________________________________________________
conv2d_91 (Conv2D)              (None, 7, 7, 32)     36896       activation_91[0][0]              
__________________________________________________________________________________________________
concatenate_43 (Concatenate)    (None, 7, 7, 576)    0           concatenate_42[0][0]             
                                                                 conv2d_91[0][0]                  
__________________________________________________________________________________________________
batch_normalization_92 (BatchNo (None, 7, 7, 576)    2304        concatenate_43[0][0]             
__________________________________________________________________________________________________
activation_92 (Activation)      (None, 7, 7, 576)    0           batch_normalization_92[0][0]     
__________________________________________________________________________________________________
conv2d_92 (Conv2D)              (None, 7, 7, 128)    73856       activation_92[0][0]              
__________________________________________________________________________________________________
batch_normalization_93 (BatchNo (None, 7, 7, 128)    512         conv2d_92[0][0]                  
__________________________________________________________________________________________________
activation_93 (Activation)      (None, 7, 7, 128)    0           batch_normalization_93[0][0]     
__________________________________________________________________________________________________
conv2d_93 (Conv2D)              (None, 7, 7, 32)     36896       activation_93[0][0]              
__________________________________________________________________________________________________
concatenate_44 (Concatenate)    (None, 7, 7, 608)    0           concatenate_43[0][0]             
                                                                 conv2d_93[0][0]                  
__________________________________________________________________________________________________
batch_normalization_94 (BatchNo (None, 7, 7, 608)    2432        concatenate_44[0][0]             
__________________________________________________________________________________________________
activation_94 (Activation)      (None, 7, 7, 608)    0           batch_normalization_94[0][0]     
__________________________________________________________________________________________________
conv2d_94 (Conv2D)              (None, 7, 7, 128)    77952       activation_94[0][0]              
__________________________________________________________________________________________________
batch_normalization_95 (BatchNo (None, 7, 7, 128)    512         conv2d_94[0][0]                  
__________________________________________________________________________________________________
activation_95 (Activation)      (None, 7, 7, 128)    0           batch_normalization_95[0][0]     
__________________________________________________________________________________________________
conv2d_95 (Conv2D)              (None, 7, 7, 32)     36896       activation_95[0][0]              
__________________________________________________________________________________________________
concatenate_45 (Concatenate)    (None, 7, 7, 640)    0           concatenate_44[0][0]             
                                                                 conv2d_95[0][0]                  
__________________________________________________________________________________________________
batch_normalization_96 (BatchNo (None, 7, 7, 640)    2560        concatenate_45[0][0]             
__________________________________________________________________________________________________
activation_96 (Activation)      (None, 7, 7, 640)    0           batch_normalization_96[0][0]     
__________________________________________________________________________________________________
conv2d_96 (Conv2D)              (None, 7, 7, 128)    82048       activation_96[0][0]              
__________________________________________________________________________________________________
batch_normalization_97 (BatchNo (None, 7, 7, 128)    512         conv2d_96[0][0]                  
__________________________________________________________________________________________________
activation_97 (Activation)      (None, 7, 7, 128)    0           batch_normalization_97[0][0]     
__________________________________________________________________________________________________
conv2d_97 (Conv2D)              (None, 7, 7, 32)     36896       activation_97[0][0]              
__________________________________________________________________________________________________
concatenate_46 (Concatenate)    (None, 7, 7, 672)    0           concatenate_45[0][0]             
                                                                 conv2d_97[0][0]                  
__________________________________________________________________________________________________
batch_normalization_98 (BatchNo (None, 7, 7, 672)    2688        concatenate_46[0][0]             
__________________________________________________________________________________________________
activation_98 (Activation)      (None, 7, 7, 672)    0           batch_normalization_98[0][0]     
__________________________________________________________________________________________________
conv2d_98 (Conv2D)              (None, 7, 7, 128)    86144       activation_98[0][0]              
__________________________________________________________________________________________________
batch_normalization_99 (BatchNo (None, 7, 7, 128)    512         conv2d_98[0][0]                  
__________________________________________________________________________________________________
activation_99 (Activation)      (None, 7, 7, 128)    0           batch_normalization_99[0][0]     
__________________________________________________________________________________________________
conv2d_99 (Conv2D)              (None, 7, 7, 32)     36896       activation_99[0][0]              
__________________________________________________________________________________________________
concatenate_47 (Concatenate)    (None, 7, 7, 704)    0           concatenate_46[0][0]             
                                                                 conv2d_99[0][0]                  
__________________________________________________________________________________________________
batch_normalization_100 (BatchN (None, 7, 7, 704)    2816        concatenate_47[0][0]             
__________________________________________________________________________________________________
activation_100 (Activation)     (None, 7, 7, 704)    0           batch_normalization_100[0][0]    
__________________________________________________________________________________________________
conv2d_100 (Conv2D)             (None, 7, 7, 128)    90240       activation_100[0][0]             
__________________________________________________________________________________________________
batch_normalization_101 (BatchN (None, 7, 7, 128)    512         conv2d_100[0][0]                 
__________________________________________________________________________________________________
activation_101 (Activation)     (None, 7, 7, 128)    0           batch_normalization_101[0][0]    
__________________________________________________________________________________________________
conv2d_101 (Conv2D)             (None, 7, 7, 32)     36896       activation_101[0][0]             
__________________________________________________________________________________________________
concatenate_48 (Concatenate)    (None, 7, 7, 736)    0           concatenate_47[0][0]             
                                                                 conv2d_101[0][0]                 
__________________________________________________________________________________________________
batch_normalization_102 (BatchN (None, 7, 7, 736)    2944        concatenate_48[0][0]             
__________________________________________________________________________________________________
activation_102 (Activation)     (None, 7, 7, 736)    0           batch_normalization_102[0][0]    
__________________________________________________________________________________________________
conv2d_102 (Conv2D)             (None, 7, 7, 128)    94336       activation_102[0][0]             
__________________________________________________________________________________________________
batch_normalization_103 (BatchN (None, 7, 7, 128)    512         conv2d_102[0][0]                 
__________________________________________________________________________________________________
activation_103 (Activation)     (None, 7, 7, 128)    0           batch_normalization_103[0][0]    
__________________________________________________________________________________________________
conv2d_103 (Conv2D)             (None, 7, 7, 32)     36896       activation_103[0][0]             
__________________________________________________________________________________________________
concatenate_49 (Concatenate)    (None, 7, 7, 768)    0           concatenate_48[0][0]             
                                                                 conv2d_103[0][0]                 
__________________________________________________________________________________________________
batch_normalization_104 (BatchN (None, 7, 7, 768)    3072        concatenate_49[0][0]             
__________________________________________________________________________________________________
activation_104 (Activation)     (None, 7, 7, 768)    0           batch_normalization_104[0][0]    
__________________________________________________________________________________________________
conv2d_104 (Conv2D)             (None, 7, 7, 128)    98432       activation_104[0][0]             
__________________________________________________________________________________________________
batch_normalization_105 (BatchN (None, 7, 7, 128)    512         conv2d_104[0][0]                 
__________________________________________________________________________________________________
activation_105 (Activation)     (None, 7, 7, 128)    0           batch_normalization_105[0][0]    
__________________________________________________________________________________________________
conv2d_105 (Conv2D)             (None, 7, 7, 32)     36896       activation_105[0][0]             
__________________________________________________________________________________________________
concatenate_50 (Concatenate)    (None, 7, 7, 800)    0           concatenate_49[0][0]             
                                                                 conv2d_105[0][0]                 
__________________________________________________________________________________________________
batch_normalization_106 (BatchN (None, 7, 7, 800)    3200        concatenate_50[0][0]             
__________________________________________________________________________________________________
activation_106 (Activation)     (None, 7, 7, 800)    0           batch_normalization_106[0][0]    
__________________________________________________________________________________________________
conv2d_106 (Conv2D)             (None, 7, 7, 128)    102528      activation_106[0][0]             
__________________________________________________________________________________________________
batch_normalization_107 (BatchN (None, 7, 7, 128)    512         conv2d_106[0][0]                 
__________________________________________________________________________________________________
activation_107 (Activation)     (None, 7, 7, 128)    0           batch_normalization_107[0][0]    
__________________________________________________________________________________________________
conv2d_107 (Conv2D)             (None, 7, 7, 32)     36896       activation_107[0][0]             
__________________________________________________________________________________________________
concatenate_51 (Concatenate)    (None, 7, 7, 832)    0           concatenate_50[0][0]             
                                                                 conv2d_107[0][0]                 
__________________________________________________________________________________________________
batch_normalization_108 (BatchN (None, 7, 7, 832)    3328        concatenate_51[0][0]             
__________________________________________________________________________________________________
activation_108 (Activation)     (None, 7, 7, 832)    0           batch_normalization_108[0][0]    
__________________________________________________________________________________________________
conv2d_108 (Conv2D)             (None, 7, 7, 128)    106624      activation_108[0][0]             
__________________________________________________________________________________________________
batch_normalization_109 (BatchN (None, 7, 7, 128)    512         conv2d_108[0][0]                 
__________________________________________________________________________________________________
activation_109 (Activation)     (None, 7, 7, 128)    0           batch_normalization_109[0][0]    
__________________________________________________________________________________________________
conv2d_109 (Conv2D)             (None, 7, 7, 32)     36896       activation_109[0][0]             
__________________________________________________________________________________________________
concatenate_52 (Concatenate)    (None, 7, 7, 864)    0           concatenate_51[0][0]             
                                                                 conv2d_109[0][0]                 
__________________________________________________________________________________________________
batch_normalization_110 (BatchN (None, 7, 7, 864)    3456        concatenate_52[0][0]             
__________________________________________________________________________________________________
activation_110 (Activation)     (None, 7, 7, 864)    0           batch_normalization_110[0][0]    
__________________________________________________________________________________________________
conv2d_110 (Conv2D)             (None, 7, 7, 128)    110720      activation_110[0][0]             
__________________________________________________________________________________________________
batch_normalization_111 (BatchN (None, 7, 7, 128)    512         conv2d_110[0][0]                 
__________________________________________________________________________________________________
activation_111 (Activation)     (None, 7, 7, 128)    0           batch_normalization_111[0][0]    
__________________________________________________________________________________________________
conv2d_111 (Conv2D)             (None, 7, 7, 32)     36896       activation_111[0][0]             
__________________________________________________________________________________________________
concatenate_53 (Concatenate)    (None, 7, 7, 896)    0           concatenate_52[0][0]             
                                                                 conv2d_111[0][0]                 
__________________________________________________________________________________________________
batch_normalization_112 (BatchN (None, 7, 7, 896)    3584        concatenate_53[0][0]             
__________________________________________________________________________________________________
activation_112 (Activation)     (None, 7, 7, 896)    0           batch_normalization_112[0][0]    
__________________________________________________________________________________________________
conv2d_112 (Conv2D)             (None, 7, 7, 128)    114816      activation_112[0][0]             
__________________________________________________________________________________________________
batch_normalization_113 (BatchN (None, 7, 7, 128)    512         conv2d_112[0][0]                 
__________________________________________________________________________________________________
activation_113 (Activation)     (None, 7, 7, 128)    0           batch_normalization_113[0][0]    
__________________________________________________________________________________________________
conv2d_113 (Conv2D)             (None, 7, 7, 32)     36896       activation_113[0][0]             
__________________________________________________________________________________________________
concatenate_54 (Concatenate)    (None, 7, 7, 928)    0           concatenate_53[0][0]             
                                                                 conv2d_113[0][0]                 
__________________________________________________________________________________________________
batch_normalization_114 (BatchN (None, 7, 7, 928)    3712        concatenate_54[0][0]             
__________________________________________________________________________________________________
activation_114 (Activation)     (None, 7, 7, 928)    0           batch_normalization_114[0][0]    
__________________________________________________________________________________________________
conv2d_114 (Conv2D)             (None, 7, 7, 128)    118912      activation_114[0][0]             
__________________________________________________________________________________________________
batch_normalization_115 (BatchN (None, 7, 7, 128)    512         conv2d_114[0][0]                 
__________________________________________________________________________________________________
activation_115 (Activation)     (None, 7, 7, 128)    0           batch_normalization_115[0][0]    
__________________________________________________________________________________________________
conv2d_115 (Conv2D)             (None, 7, 7, 32)     36896       activation_115[0][0]             
__________________________________________________________________________________________________
concatenate_55 (Concatenate)    (None, 7, 7, 960)    0           concatenate_54[0][0]             
                                                                 conv2d_115[0][0]                 
__________________________________________________________________________________________________
batch_normalization_116 (BatchN (None, 7, 7, 960)    3840        concatenate_55[0][0]             
__________________________________________________________________________________________________
activation_116 (Activation)     (None, 7, 7, 960)    0           batch_normalization_116[0][0]    
__________________________________________________________________________________________________
conv2d_116 (Conv2D)             (None, 7, 7, 128)    123008      activation_116[0][0]             
__________________________________________________________________________________________________
batch_normalization_117 (BatchN (None, 7, 7, 128)    512         conv2d_116[0][0]                 
__________________________________________________________________________________________________
activation_117 (Activation)     (None, 7, 7, 128)    0           batch_normalization_117[0][0]    
__________________________________________________________________________________________________
conv2d_117 (Conv2D)             (None, 7, 7, 32)     36896       activation_117[0][0]             
__________________________________________________________________________________________________
concatenate_56 (Concatenate)    (None, 7, 7, 992)    0           concatenate_55[0][0]             
                                                                 conv2d_117[0][0]                 
__________________________________________________________________________________________________
batch_normalization_118 (BatchN (None, 7, 7, 992)    3968        concatenate_56[0][0]             
__________________________________________________________________________________________________
activation_118 (Activation)     (None, 7, 7, 992)    0           batch_normalization_118[0][0]    
__________________________________________________________________________________________________
conv2d_118 (Conv2D)             (None, 7, 7, 128)    127104      activation_118[0][0]             
__________________________________________________________________________________________________
batch_normalization_119 (BatchN (None, 7, 7, 128)    512         conv2d_118[0][0]                 
__________________________________________________________________________________________________
activation_119 (Activation)     (None, 7, 7, 128)    0           batch_normalization_119[0][0]    
__________________________________________________________________________________________________
conv2d_119 (Conv2D)             (None, 7, 7, 32)     36896       activation_119[0][0]             
__________________________________________________________________________________________________
concatenate_57 (Concatenate)    (None, 7, 7, 1024)   0           concatenate_56[0][0]             
                                                                 conv2d_119[0][0]                 
__________________________________________________________________________________________________
average_pooling2d_3 (AveragePoo (None, 1, 1, 1024)   0           concatenate_57[0][0]             
__________________________________________________________________________________________________
dense (Dense)                   (None, 1, 1, 1000)   1025000     average_pooling2d_3[0][0]        
==================================================================================================
Total params: 8,068,404
Trainable params: 7,986,926
Non-trainable params: 81,478
__________________________________________________________________________________________________
```
</details>

---

:small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond:

---


## [9. Autoencoders Project](https://github.com/Luffy981/holbertonschool-machine_learning/tree/master/unsupervised_learning/0x04-autoencoders)
Autoencoders are an unsupervised learning technique and are a type of artificial neural network used to learn efficient codings of unlabeled data. It is essentially a learned dimensionality reduction technique, where the reduced dimensions exists in the latent vector space. The encoding is validated and refined by learning to regenerate the input from the encoding. In this project, we learned to build autoencoder models using the keras framework on the mnist dataset.

### Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| matplotlib         | ^3.4.3  |
| tensorflow         | ^2.6.0  |
| keras              | ^2.6.0  |

### Tasks

#### [Vanilla Autoencoder](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x04-autoencoders/0-vanilla.py "Valnilla Autoencoder")
A function that creates a vanilla autoencoder network.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('0-vanilla').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
np.random.seed(0)
tf.set_random_seed(0)
encoder, decoder, auto = autoencoder(784, [128, 64], 32)
auto.fit(x_train, x_train, epochs=50,batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded = encoder.predict(x_test[:10])
print(np.mean(encoded))
reconstructed = decoder.predict(encoded)

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i].reshape((28, 28)))
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i].reshape((28, 28)))
plt.show()
```

![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x04-autoencoders/images/autoencoders-0.png)

---

#### [Sparse Encoder](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x04-autoencoders/1-sparse.py "Sparse Encoder")
Creates autoencoder network with sparse activatons.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('1-sparse').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
np.random.seed(0)
tf.set_random_seed(0)
encoder, decoder, auto = autoencoder(784, [128, 64], 32, 10e-6)
auto.fit(x_train, x_train, epochs=100,batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded = encoder.predict(x_test[:10])
print(np.mean(encoded))
reconstructed = decoder.predict(encoded)

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i].reshape((28, 28)))
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i].reshape((28, 28)))
plt.show()
```
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x04-autoencoders/images/autoencoders-1.png)

---

#### [Convolutional Autoencoder](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x04-autoencoders/2-convolutional.py "Convolutional Autoencoder")
Creates convolutional autoencoder network.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('2-convolutional').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
print(x_train.shape)
print(x_test.shape)
np.random.seed(0)
tf.set_random_seed(0)
encoder, decoder, auto = autoencoder((28, 28, 1), [16, 8, 8], (4, 4, 8))
auto.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded = encoder.predict(x_test[:10])
print(np.mean(encoded))
reconstructed = decoder.predict(encoded)[:,:,:,0]

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i,:,:,0])
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i])
plt.show()
```
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x04-autoencoders/images/autoencoders-2.png)

---

#### [Variational Autoencoder](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x04-autoencoders/3-variational.py "Variational Autoencoder")
Creates variational autoencoder network.

``` python
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('3-variational').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
np.random.seed(0)
tf.set_random_seed(0)
encoder, decoder, auto = autoencoder(784, [512], 2)
auto.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded, mu, log_sig = encoder.predict(x_test[:10])
print(mu)
print(np.exp(log_sig / 2))
reconstructed = decoder.predict(encoded).reshape((-1, 28, 28))
x_test = x_test.reshape((-1, 28, 28))

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i])
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i])
plt.show()


l1 = np.linspace(-3, 3, 25)
l2 = np.linspace(-3, 3, 25)
L = np.stack(np.meshgrid(l1, l2, sparse=False, indexing='ij'), axis=2)
G = decoder.predict(L.reshape((-1, 2)), batch_size=125)

for i in range(25*25):
    ax = plt.subplot(25, 25, i + 1)
    ax.axis('off')
    plt.imshow(G[i].reshape((28, 28)))
plt.show()
```
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x04-autoencoders/images/autoencoder-3-1.png)
![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x04-autoencoders/images/autoencoder-3-2.png)

---


:small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond:

---


## [10. Markov Chains and Hidden Markov Models](https://github.com/Luffy981/holbertonschool-machine_learning/tree/master/unsupervised_learning/0x02-hmm)
Hidden Markov Model (HMM) is a statistical Markov model in which the system being modeled is assumed to be a Markov process. As part of the definition, HMM requires that there be an observable process Y whose outcomes are "influenced" by the outcomes of X in a known way. Since X cannot be observed directly, the goal is to learn about X by observing Y. A hidden markov model is modeled using transition probabilties (the probabilty of transitioning from any hidden state to any other hidden state at a given time step) and emission probabilities (the probability of being in a hidden state given an observation).

![image](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x02-hmm/images/hmm.png)


### Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |

### Tasks
The use case examples are not very intuitive, so check the documentation I provided to understand the how each function works.

#### [Markov Chain](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x02-hmm/0-markov_chain.py "Markov Chain")
Determines the probability of a markov chain being in a particular state after a specified number of iterations.
``` python
#!/usr/bin/env python3

import numpy as np
markov_chain = __import__('0-markov_chain').markov_chain

if __name__ == "__main__":
    P = np.array([[0.25, 0.2, 0.25, 0.3], [0.2, 0.3, 0.2, 0.3], [0.25, 0.25, 0.4, 0.1], [0.3, 0.3, 0.1, 0.3]])
    s = np.array([[1, 0, 0, 0]])
    print(markov_chain(P, s, 300))
```
---

#### [Regular Chains](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x02-hmm/1-regular.py "Regular Chains")
Determines the steady state probabilities of a regular markov chain.
``` python
#!/usr/bin/env python3

import numpy as np
regular = __import__('1-regular').regular

if __name__ == '__main__':
    a = np.eye(2)
    b = np.array([[0.6, 0.4],
                  [0.3, 0.7]])
    c = np.array([[0.25, 0.2, 0.25, 0.3],
                  [0.2, 0.3, 0.2, 0.3],
                  [0.25, 0.25, 0.4, 0.1],
                  [0.3, 0.3, 0.1, 0.3]])
    print(regular(a))
    print(regular(b))
    print(regular(c))
```
---

#### [Absorbing Chains](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x02-hmm/2-absorbing.py "Absorbing Chains")
Determines if a markov chain is absorbing.
``` python
#!/usr/bin/env python3

import numpy as np
absorbing = __import__('2-absorbing').absorbing

if __name__ == '__main__':
    a = np.eye(2)
    b = np.array([[0.6, 0.4],
                  [0.3, 0.7]])
    c = np.array([[0.25, 0.2, 0.25, 0.3],
                  [0.2, 0.3, 0.2, 0.3],
                  [0.25, 0.25, 0.4, 0.1],
                  [0.3, 0.3, 0.1, 0.3]])
    d = np.array([[1, 0, 0, 0, 0],
                  [0.25, 0.75, 0, 0, 0],
                  [0, 0, 0.5, 0.2, 0.3],
                  [0, 0, 0.3, 0.5, .2],
                  [0, 0, 0.2, 0.3, 0.5]])
    print(absorbing(a))
    print(absorbing(b))
    print(absorbing(c))
```
---

#### [The Forward Algorithm](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x02-hmm/3-forward.py "The Forward Algorithm")
Performs the forward algorithm for a hidden markov model.
``` python
#!/usr/bin/env python3

import numpy as np
forward = __import__('3-forward').forward

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    P, F = forward(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(F)
```
---

#### [The Viretbi Algorithm](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x02-hmm/4-viterbi.py "The Viretbi Algorithm")
Calculates the most likely sequence of hidden states for a hidden markov model.
``` python
#!/usr/bin/env python3

import numpy as np
viterbi = __import__('4-viterbi').viterbi

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    path, P = viterbi(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(path)
```
---

#### [The Backward Algorithm](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x02-hmm/5-backward.py "The Backward Algorithm")
Performs the backward algorithm for a hidden markov model.
``` python
#!/usr/bin/env python3

import numpy as np
backward = __import__('5-backward').backward

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    P, B = backward(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(B)
```
---

#### [The Baum-Welch Algorithm](https://github.com/Luffy981/holbertonschool-machine_learning/blob/master/unsupervised_learning/0x02-hmm/6-baum_welch.py "The Baum-Welch Algorithm")
Module contains function that performs the Baum-Welch algorithm for finding locally optimal transition and emission probabilities for a Hidden Markov Model.

``` python
#!/usr/bin/env python3

import numpy as np
baum_welch = __import__('6-baum_welch').baum_welch

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00],
                         [0.40, 0.50, 0.10]])
    Transition = np.array([[0.60, 0.4],
                           [0.30, 0.70]])
    Initial = np.array([0.5, 0.5])
    Hidden = [np.random.choice(2, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(2, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(3, p=Emission[s]))
    Observations = np.array(Observations)
    T_test = np.ones((2, 2)) / 2
    E_test = np.abs(np.random.randn(2, 3))
    E_test = E_test / np.sum(E_test, axis=1).reshape((-1, 1))
    T, E = baum_welch(Observations, T_test, E_test, Initial.reshape((-1, 1)))
    print(np.round(T, 2))
    print(np.round(E, 2))
```

---

:small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond: :small_blue_diamond:

---
