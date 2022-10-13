# 0x13. QA Bot
## Details
 By: Alexa Orrico, Software Engineer at Holberton School Weight: 1Project will startOct 3, 2022 12:00 AM, must end byOct 14, 2022 12:00 AMManual QA review must be done(request it when you are done with the project) ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/11/52f1c0b37c7c58d30706b7e24bb7534b5a5889db.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20221013%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221013T013545Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=038259a1f3c35ed30e3bc0454a03ae5e6b74ceae1a67f5f6dc499930918ce1bc) 

## Resources
Read or watch : 
* [Improving Language Understanding by Generative Pre-Training (2018)](https://intranet.hbtn.io/rltoken/isif62esIXAxfKrVWRFUvA) 

* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](https://intranet.hbtn.io/rltoken/tFF7DtiDKAHOB_AnliLQ-w) 

* [SQuAD 2.0](https://intranet.hbtn.io/rltoken/TWw2zYrbEv1A6Xp7Q5xeIg) 

* [Know What You Don’t Know: Unanswerable Questions for SQuAD (2018)](https://intranet.hbtn.io/rltoken/JDS8l-V-AyRMgZ-VWCR4mg) 

* [GLUE Benchmark](https://intranet.hbtn.io/rltoken/G5RYcXocPNTEGSeiWFfT9Q) 

* [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding (2019)](https://intranet.hbtn.io/rltoken/58OooeuXuihK66vnEQL5og) 

* [Speech-transformer: A no-recurrence sequence-to-sequence model for speech recognition (2018)](https://intranet.hbtn.io/rltoken/t0l-uUrcWnFizy9y6ySnOg) 

More recent papers in NLP:
* [Generating Long Sequences with Sparse Transformers (2019)](https://intranet.hbtn.io/rltoken/fLho-pVtLacRpYcoeKQlbw) 

* [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (2019)](https://intranet.hbtn.io/rltoken/DNjHiFbB_cOVhIB7usnx8A) 

* [XLNet: Generalized Autoregressive Pretraining for Language Understanding (2019)](https://intranet.hbtn.io/rltoken/fr6iKOeQ2J2I3m-2mnVk5w) 

* [Language Models are Unsupervised Multitask Learners (GPT-2, 2019)](https://intranet.hbtn.io/rltoken/4eTrqxWdepAKneRJsjaD8g) 

* [Language Models are Few-Shot Learners (GPT-3, 2020)](https://intranet.hbtn.io/rltoken/-LbOp0yRvuSn1Hqzd2OF2Q) 

* [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations (2020)](https://intranet.hbtn.io/rltoken/s4A59Pf2b2QjWT8NjgWB8g) 

To keep up with the newest papers and their code bases go to  [paperswithcode.com](https://intranet.hbtn.io/rltoken/Glv09z1yOkMl0Gsl0zBHjg) 
 . For example, check out the  [raked list of state of the art models for Language Modelling on Penn Treebank](https://intranet.hbtn.io/rltoken/DZKmWejiXC0czbR9j32N_g) 
 .
## Learning Objectives
At the end of this project, you are expected to be able to  [explain to anyone](https://intranet.hbtn.io/rltoken/zhTSATRc8MjZEpBbdNxggQ) 
 ,  without the help of Google :
### General
* What is Question-Answering?
* What is Semantic Search?
* What is BERT?
* How to develop a QA chatbot
* How to use the  ` transformers `  library
* How to use the  ` tensorflow-hub `  library
## Requirements
### General
* Allowed editors:  ` vi ` ,  ` vim ` ,  ` emacs ` 
* All your files will be interpreted/compiled on Ubuntu 16.04 LTS using  ` python3 `  (version 3.6.12)
* Your files will be executed with  ` numpy `  (version 1.18) and  ` tensorflow `  (version 2.3)
* All your files should end with a new line
* The first line of all your files should be exactly  ` #!/usr/bin/env python3 ` 
* All of your files must be executable
* A  ` README.md `  file, at the root of the folder of the project, is mandatory
* Your code should follow the  ` pycodestyle `  style (version 2.4)
* All your modules should have documentation ( ` python3 -c 'print(__import__("my_module").__doc__)' ` )
* All your classes should have documentation ( ` python3 -c 'print(__import__("my_module").MyClass.__doc__)' ` )
* All your functions (inside and outside a class) should have documentation ( ` python3 -c 'print(__import__("my_module").my_function.__doc__)' `  and  ` python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)' ` )
## Upgrade to Tensorflow 2.3
 ` pip install --user tensorflow==2.3 ` 
## Install Tensorflow Hub
 ` pip install --user tensorflow-hub ` 
## Install Transformers
 ` pip install --user transformers ` 
## Zendesk Articles
For this project, we will be using a collection of Holberton USA Zendesk Articles,  [ZendeskArticles.zip](https://holbertonintranet.s3.amazonaws.com/uploads/misc/2020/11/c15a067b44a328c7d5a03c79070b7865f444d1e3.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20221013%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221013T013545Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=b98e4bf4b4e409e8bb00ef19e59ec910ea9f93dc2500b6bc0688dfcc639c21aa) 
 .
## Tasks
### 0. Question Answering
          mandatory         Progress vs Score  Task Body Write a function   ` def question_answer(question, reference): `   that finds a snippet of text within a reference document to answer a question:
*  ` question `  is a string containing the question to answer
*  ` reference `  is a string containing the reference document from which to find the answer
* Returns: a string containing the answer
* If no answer is found, return  ` None ` 
* Your function should use the   ` bert-uncased-tf2-qa `  model from the  ` tensorflow-hub `  library
* Your function should use the pre-trained  ` BertTokenizer ` ,  ` bert-large-uncased-whole-word-masking-finetuned-squad ` , from the  ` transformers `  library
```bash
$ cat 0-main.py
#!/usr/bin/env python3

question_answer = __import__('0-qa').question_answer

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

print(question_answer('When are PLDs?', reference))
$ ./0-main.py
on - site days from 9 : 00 am to 3 : 00 pm
$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x13-qa_bot ` 
* File:  ` 0-qa.py ` 
 Self-paced manual review  Panel footer - Controls 
### 1. Create the loop
          mandatory         Progress vs Score  Task Body Create a script that takes in input from the user with the prompt   ` Q: `   and prints   ` A: `   as a response. If the user inputs   ` exit `  ,   ` quit `  ,   ` goodbye `  , or   ` bye `  , case insensitive, print   ` A: Goodbye `   and exit.
 ` $ ./1-loop.py
Q: Hello
A:
Q: How are you?
A:
Q: BYE
A: Goodbye
$
 `  Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` 0x13-qa_bot ` 
* File:  ` 1-loop.py ` 
 Self-paced manual review  Panel footer - Controls 
### 2. Answer Questions
          mandatory         Progress vs Score  Task Body Based on the previous tasks, write a function   ` def answer_loop(reference): `   that answers questions from a reference text:
*  ` reference `  is the reference text
* If the answer cannot be found in the reference text, respond with  ` Sorry, I do not understand your question. ` 
```bash
$ cat 2-main.py
#!/usr/bin/env python3

answer_loop = __import__('2-qa').answer_loop

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

answer_loop(reference)
$ ./2-main.py
Q: When are PLDs?
A: from 9 : 00 am to 3 : 00 pm
Q: What are Mock Interviews?
A: Sorry, I do not understand your question.
Q: What does PLD stand for?
A: peer learning days
Q: EXIT
A: Goodbye
$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x13-qa_bot ` 
* File:  ` 2-qa.py ` 
 Self-paced manual review  Panel footer - Controls 
### 3. Semantic Search
          mandatory         Progress vs Score  Task Body Write a function   ` def semantic_search(corpus_path, sentence): `   that performs semantic search on a corpus of documents:
*  ` corpus_path `  is the path to the corpus of reference documents on which to perform semantic search
*  ` sentence `  is the sentence from which to perform semantic search
* Returns: the reference text of the document most similar to  ` sentence ` 
```bash
$ cat 3-main.py
#!/usr/bin/env python3

semantic_search = __import__('3-semantic_search').semantic_search

print(semantic_search('ZendeskArticles', 'When are PLDs?'))
$ ./ 3-main.py
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
$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x13-qa_bot ` 
* File:  ` 3-semantic_search.py ` 
 Self-paced manual review  Panel footer - Controls 
### 4. Multi-reference Question Answering
          mandatory         Progress vs Score  Task Body Based on the previous tasks, write a function   ` def question_answer(corpus_path): `   that answers questions from multiple reference texts:
*  ` corpus_path `  is the path to the corpus of reference documents
```bash
$ cat 4-main.py
#!/usr/bin/env python3

question_answer = __import__('4-qa').question_answer

question_answer('ZendeskArticles')
$ ./4-main.py
Q: When are PLDs?
A: on - site days from 9 : 00 am to 3 : 00 pm
Q: What are Mock Interviews?
A: help you train for technical interviews
Q: What does PLD stand for?
A: peer learning days
Q: goodbye
A: Goodbye
$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x13-qa_bot ` 
* File:  ` 4-qa.py ` 
 Self-paced manual review  Panel footer - Controls 
Ready for a  manual review
