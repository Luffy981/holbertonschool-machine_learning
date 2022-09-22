# 0x10. Natural Language Processing - Evaluation Metrics
## Details
 By: Alexa Orrico, Software Engineer at Holberton School Weight: 1Ongoing second chance project - startedSep 17, 2022 12:00 AM, must end bySep 22, 2022 12:00 AM An auto review will be launched at the deadline#### In a nutshell…
* Auto QA review:          0.0/21 mandatory      
* Altogether:         0.0%* Mandatory: 0.0%
* Optional: no optional tasks

## Resources
Read or watch:
* [7 Applications of Deep Learning for Natural Language Processing](https://intranet.hbtn.io/rltoken/EFeppnrszrEGza6nrymxgQ) 

* [10 Applications of Artificial Neural Networks in Natural Language Processing](https://intranet.hbtn.io/rltoken/1COfka_urNOmhQuqmeZWkA) 

* [A Gentle Introduction to Calculating the BLEU Score for Text in Python](https://intranet.hbtn.io/rltoken/lC85P6iX492bGuBncUNwiw) 

* [Bleu Score](https://intranet.hbtn.io/rltoken/lT-MBM6w7AjXPIiZoPKR_A) 

* [Evaluating Text Output in NLP: BLEU at your own risk](https://intranet.hbtn.io/rltoken/WkIjrg9GxphCmjzOYG1vzw) 

* [ROUGE metric](https://intranet.hbtn.io/rltoken/_-kqsn4KiHgRAL1Cz3jqOw) 

* [Evaluation and Perplexity](https://intranet.hbtn.io/rltoken/Mgyoxa8c6WvpFJaHFxqlQQ) 

Definitions to skim
* [BLEU](https://intranet.hbtn.io/rltoken/njmmpbMuP0cPnnWwbFpj3A) 

* [ROUGE](https://intranet.hbtn.io/rltoken/BJK2tEo1kVYXytMDoVF9fQ) 

* [Perplexity](https://intranet.hbtn.io/rltoken/MayHONfLeczBB8qWvaDrkQ) 

References:
* [BLEU: a Method for Automatic Evaluation of Machine Translation (2002)](https://intranet.hbtn.io/rltoken/EsAnXupX-J-y6YwoH6VUTw) 

* [ROUGE: A Package for Automatic Evaluation of Summaries (2004)](https://intranet.hbtn.io/rltoken/A8PhjII-AIn5JCQzhXNQ2A) 

## Learning Objectives
At the end of this project, you are expected to be able to  [explain to anyone](https://intranet.hbtn.io/rltoken/wObQEQrYx4Kuy6-OJHXLBw) 
 ,  without the help of Google :
### General
* What are the applications of natural language processing?
* What is a BLEU score?
* What is a ROUGE score?
* What is perplexity?
* When should you use one evaluation metric over another?
## Requirements
### General
* Allowed editors:  ` vi ` ,  ` vim ` ,  ` emacs ` 
* All your files will be interpreted/compiled on Ubuntu 16.04 LTS using  ` python3 `  (version 3.5)
* Your files will be executed with  ` numpy `  (version 1.15)
* All your files should end with a new line
* The first line of all your files should be exactly  ` #!/usr/bin/env python3 ` 
* All of your files must be executable
* A  ` README.md `  file, at the root of the folder of the project, is mandatory
* Your code should follow the  ` pycodestyle `  style (version 2.4)
* All your modules should have documentation ( ` python3 -c 'print(__import__("my_module").__doc__)' ` )
* All your classes should have documentation ( ` python3 -c 'print(__import__("my_module").MyClass.__doc__)' ` )
* All your functions (inside and outside a class) should have documentation ( ` python3 -c 'print(__import__("my_module").my_function.__doc__)' `  and  ` python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)' ` )
* You are not allowed to use the  ` nltk `  module
### Quiz questions
Great!          You've completed the quiz successfully! Keep going!          (Show quiz)#### 
        
        Question #0
    
 Quiz question Body The BLEU score measures:
 Quiz question Answers * A model’s accuracy

* A model’s precision

* A model’s recall

* A model’s perplexity

 Quiz question Tips #### 
        
        Question #1
    
 Quiz question Body The ROUGE score measures:
 Quiz question Answers * A model’s accuracy

* A model’s precision

* A model’s recall

* A model’s perplexity

 Quiz question Tips #### 
        
        Question #2
    
 Quiz question Body Perplexity measures:
 Quiz question Answers * The accuracy of a prediction

* The branching factor of a prediction

* A prediction’s recall

* A prediction’s accuracy

 Quiz question Tips #### 
        
        Question #3
    
 Quiz question Body The BLEU score was designed for:
 Quiz question Answers * Sentiment Analysis

* Machine Translation

* Question-Answering

* Document Summarization

 Quiz question Tips #### 
        
        Question #4
    
 Quiz question Body What are the shortcomings of the BLEU score?
 Quiz question Answers * It cannot judge grammatical accuracy

* It cannot judge meaning

* It does not work with languages that lack word boundaries

* A higher score is not necessarily indicative of a better translation

 Quiz question Tips ## Tasks
### 0. Unigram BLEU score
          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write the function   ` def uni_bleu(references, sentence): `   that calculates the unigram BLEU score for a sentence:
*  ` references `  is a list of reference translations* each reference translation is a list of the words in the translation

*  ` sentence `  is a list containing the model proposed sentence
* Returns: the unigram BLEU score
```bash
$ cat 0-main.py
#!/usr/bin/env python3

uni_bleu = __import__('0-uni_bleu').uni_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(uni_bleu(references, sentence))
$ ./0-main.py
0.6549846024623855
$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x10-nlp_metrics ` 
* File:  ` 0-uni_bleu.py ` 
 Self-paced manual review  Panel footer - Controls 
### 1. N-gram BLEU score
          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write the function   ` def ngram_bleu(references, sentence, n): `   that calculates the n-gram BLEU score for a sentence:
*  ` references `  is a list of reference translations* each reference translation is a list of the words in the translation

*  ` sentence `  is a list containing the model proposed sentence
*  ` n `  is the size of the n-gram to use for evaluation
* Returns: the n-gram BLEU score
```bash
$ cat 1-main.py
#!/usr/bin/env python3

ngram_bleu = __import__('1-ngram_bleu').ngram_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(ngram_bleu(references, sentence, 2))
$ ./1-main.py
0.6140480648084865
$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x10-nlp_metrics ` 
* File:  ` 1-ngram_bleu.py ` 
 Self-paced manual review  Panel footer - Controls 
### 2. Cumulative N-gram BLEU score
          mandatory         Progress vs Score           Score: 0.00% (Checks completed: 0.00%)         Task Body Write the function   ` def cumulative_bleu(references, sentence, n): `   that calculates the cumulative n-gram BLEU score for a sentence:
*  ` references `  is a list of reference translations* each reference translation is a list of the words in the translation

*  ` sentence `  is a list containing the model proposed sentence
*  ` n `  is the size of the largest n-gram to use for evaluation
* All n-gram scores should be weighted evenly
* Returns: the cumulative n-gram BLEU score
```bash
$ cat 2-main.py
#!/usr/bin/env python3

cumulative_bleu = __import__('2-cumulative_bleu').cumulative_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(cumulative_bleu(references, sentence, 4))
$ ./2-main.py
0.5475182535069453
$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` supervised_learning/0x10-nlp_metrics ` 
* File:  ` 2-cumulative_bleu.py ` 
 Self-paced manual review  Panel footer - Controls 
