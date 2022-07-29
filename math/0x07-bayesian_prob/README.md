# 0x07. Bayesian Probability
## Details
      By Alexa Orrico, Software Engineer at Holberton School          Weight: 1                Ongoing project - started Jul 28, 2022 , must end by Jul 29, 2022           - you're done with 0% of tasks.              Checker was released at Jul 28, 2022 12:00 PM        An auto review will be launched at the deadline       ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/8/8358e1144bbb1fcc51b4.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220728%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220728T231527Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e279b7b0c26b639c3acd19dcd79c71c4ebf79a4b22694e0e6dde1098b180b259) 

## Resources
Read or watch :
* [Bayesian probability](https://intranet.hbtn.io/rltoken/sTaD6jnhKs_TTfQZzJ3yhQ) 

* [Bayesian statistics](https://intranet.hbtn.io/rltoken/1v-8Ekg3h0raamUXPhY2QQ) 

* [Bayes’ Theorem - The Simplest Case](https://intranet.hbtn.io/rltoken/VqWGh8Z-0EAiTxGbxOR3gw) 

* [A visual guide to Bayesian thinking](https://intranet.hbtn.io/rltoken/oO_89xTL9ijXyB6d6TJUPg) 

* [Base Rates](https://intranet.hbtn.io/rltoken/JheEb1W71ompqRatlXHIzw) 

* [Bayesian statistics: a comprehensive course](https://intranet.hbtn.io/rltoken/Kmv4IuCD4b2C1et6zDPHGA) 
* [Bayes’ rule - an intuitive explanation](https://intranet.hbtn.io/rltoken/wVw3Sust10jQDa3-BDDzUA) 

* [Bayes’ rule in statistics](https://intranet.hbtn.io/rltoken/wUhrdfFq0be4VH4strzaXQ) 

* [Bayes’ rule in inference - likelihood](https://intranet.hbtn.io/rltoken/EhC5nfFrqlMxRG6a8YC3dw) 

* [Bayes’ rule in inference - the prior and denominator](https://intranet.hbtn.io/rltoken/76IgPqJyHwanrMbxPld4qg) 

* [Bayes’ rule denominator: discrete and continuous](https://intranet.hbtn.io/rltoken/vO953V4kzEr6izhjVy2zqg) 

* [Bayes’ rule: why likelihood is not a probability](https://intranet.hbtn.io/rltoken/UGHHljv4xEmsSkF9r5h4wQ) 


## Learning Objectives
* What is Bayesian Probability?
* What is Bayes’ rule and how do you use it?
* What is a base rate?
* What is a prior?
* What is a posterior?
* What is a likelihood?
## Requirements
### General
* Allowed editors:  ` vi ` ,  ` vim ` ,  ` emacs ` 
* All your files will be interpreted/compiled on Ubuntu 16.04 LTS using  ` python3 `  (version 3.5)
* Your files will be executed with  ` numpy `  (version 1.15)
* All your files should end with a new line
* The first line of all your files should be exactly  ` #!/usr/bin/env python3 ` 
* A  ` README.md `  file, at the root of the folder of the project, is mandatory
* Your code should use the  ` pycodestyle `  style (version 2.5)
* All your modules should have documentation ( ` python3 -c 'print(__import__("my_module").__doc__)' ` )
* All your classes should have documentation ( ` python3 -c 'print(__import__("my_module").MyClass.__doc__)' ` )
* All your functions (inside and outside a class) should have documentation ( ` python3 -c 'print(__import__("my_module").my_function.__doc__)' `  and  ` python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)' ` )
* Unless otherwise noted, you are not allowed to import any module except  ` import numpy as np ` 
* All your files must be executable
* The length of your files will be tested using  ` wc ` 
### Quiz questions
Great!          You've completed the quiz successfully! Keep going!          (Show quiz)#### 
        
        Question #0
    
 Quiz question Body Bayes’ rule states that   ` P(A | B) = P(B | A) * P(A) / P(B) ` 
What is   ` P(A | B) `  ?
 Quiz question Answers * Likelihood

* Marginal probability

* Posterior probability

* Prior probability

 Quiz question Tips #### 
        
        Question #1
    
 Quiz question Body Bayes’ rule states that   ` P(A | B) = P(B | A) * P(A) / P(B) ` 
What is   ` P(B | A) `  ?
 Quiz question Answers * Likelihood

* Marginal probability

* Posterior probability

* Prior probability

 Quiz question Tips #### 
        
        Question #2
    
 Quiz question Body Bayes’ rule states that   ` P(A | B) = P(B | A) * P(A) / P(B) ` 
What is   ` P(A) `  ?
 Quiz question Answers * Likelihood

* Marginal probability

* Posterior probability

* Prior probability

 Quiz question Tips #### 
        
        Question #3
    
 Quiz question Body Bayes’ rule states that   ` P(A | B) = P(B | A) * P(A) / P(B) ` 
What is   ` P(B) `  ?
 Quiz question Answers * Likelihood

* Marginal probability

* Posterior probability

* Prior probability

 Quiz question Tips ## Tasks
### 0. Likelihood
          mandatory         Progress vs Score  Task Body You are conducting a study on a revolutionary cancer drug and are looking to find the probability that a patient who takes this drug will develop severe side effects. During your trials,   ` n `   patients take the drug and   ` x `   patients develop severe side effects. You can assume that   ` x `   follows a binomial distribution.
Write a function   ` def likelihood(x, n, P): `   that calculates the likelihood of obtaining this data given various hypothetical probabilities of developing severe side effects:
*  ` x `  is the number of patients that develop severe side effects
*  ` n `  is the total number of patients observed
*  ` P `  is a 1D  ` numpy.ndarray `  containing the various hypothetical probabilities of developing severe side effects
* If  ` n `  is not a positive integer, raise a  ` ValueError `  with the message  ` n must be a positive integer ` 
* If  ` x `  is not an integer that is greater than or equal to  ` 0 ` , raise a  ` ValueError `  with the message  ` x must be an integer that is greater than or equal to 0 ` 
* If  ` x `  is greater than  ` n ` , raise a  ` ValueError `  with the message  ` x cannot be greater than n ` 
* If  ` P `  is not a 1D  ` numpy.ndarray ` , raise a  ` TypeError `  with the message  ` P must be a 1D numpy.ndarray ` 
* If any value in  ` P `  is not in the range  ` [0, 1] ` , raise a  ` ValueError `  with the message  ` All values in P must be in the range [0, 1] ` 
* Returns: a 1D  ` numpy.ndarray `  containing the likelihood of obtaining the data,  ` x `  and  ` n ` , for each probability in  ` P ` , respectively
```bash
alexa@ubuntu-xenial:0x07-bayesian_prob$ cat 0-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    likelihood = __import__('0-likelihood').likelihood

    P = np.linspace(0, 1, 11) # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(likelihood(26, 130, P))
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./0-main.py 
[0.00000000e+00 2.71330957e-04 8.71800070e-02 3.07345706e-03
 5.93701546e-07 1.14387595e-12 1.09257177e-20 6.10151799e-32
 9.54415702e-49 1.00596671e-78 0.00000000e+00]
alexa@ubuntu-xenial:0x07-bayesian_prob$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x07-bayesian_prob ` 
* File:  ` 0-likelihood.py ` 
 Self-paced manual review  Panel footer - Controls 
### 1. Intersection
          mandatory         Progress vs Score  Task Body Based on   ` 0-likelihood.py `  , write a function   ` def intersection(x, n, P, Pr): `   that calculates the intersection of obtaining this data with the various hypothetical probabilities:
*  ` x `  is the number of patients that develop severe side effects
*  ` n `  is the total number of patients observed
*  ` P `  is a 1D  ` numpy.ndarray `  containing the various hypothetical probabilities of developing severe side effects
*  ` Pr `  is a 1D  ` numpy.ndarray `  containing the prior beliefs of  ` P ` 
* If  ` n `  is not a positive integer, raise a  ` ValueError `  with the message  ` n must be a positive integer ` 
* If  ` x `  is not an integer that is greater than or equal to  ` 0 ` , raise a  ` ValueError `  with the message  ` x must be an integer that is greater than or equal to 0 ` 
* If  ` x `  is greater than  ` n ` , raise a  ` ValueError `  with the message  ` x cannot be greater than n ` 
* If  ` P `  is not a 1D  ` numpy.ndarray ` , raise a  ` TypeError `  with the message  ` P must be a 1D numpy.ndarray ` 
* If  ` Pr `  is not a  ` numpy.ndarray `  with the same shape as  ` P ` , raise a  ` TypeError `  with the message  ` Pr must be a numpy.ndarray with the same shape as P ` 
* If any value in  ` P `  or  ` Pr `  is not in the range  ` [0, 1] ` , raise a  ` ValueError `  with the message  ` All values in {P} must be in the range [0, 1] `  where  ` {P} `  is the incorrect variable
* If  ` Pr `  does not sum to  ` 1 ` , raise a  ` ValueError `  with the message  ` Pr must sum to 1 ` Hint: use [numpy.isclose](https://intranet.hbtn.io/rltoken/7pptg2vy0_-c0qQ9MnZu1w) 

* All exceptions should be raised in the above order
* Returns: a 1D  ` numpy.ndarray `  containing the intersection of obtaining  ` x `  and  ` n `  with each probability in  ` P ` , respectively
```bash
alexa@ubuntu-xenial:0x07-bayesian_prob$ cat 1-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    intersection = __import__('1-intersection').intersection

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11 # this prior assumes that everything is equally as likely
    print(intersection(26, 130, P, Pr))
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./1-main.py 
[0.00000000e+00 2.46664506e-05 7.92545518e-03 2.79405187e-04
 5.39728678e-08 1.03988723e-13 9.93247059e-22 5.54683454e-33
 8.67650639e-50 9.14515194e-80 0.00000000e+00]
alexa@ubuntu-xenial:0x07-bayesian_prob$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x07-bayesian_prob ` 
* File:  ` 1-intersection.py ` 
 Self-paced manual review  Panel footer - Controls 
### 2. Marginal Probability
          mandatory         Progress vs Score  Task Body Based on   ` 1-intersection.py `  , write a function   ` def marginal(x, n, P, Pr): `   that calculates the marginal probability of obtaining the data:
*  ` x `  is the number of patients that develop severe side effects
*  ` n `  is the total number of patients observed
*  ` P `  is a 1D  ` numpy.ndarray `  containing the various hypothetical probabilities of patients developing severe side effects
*  ` Pr `  is a 1D  ` numpy.ndarray `  containing the prior beliefs about  ` P ` 
* If  ` n `  is not a positive integer, raise a  ` ValueError `  with the message  ` n must be a positive integer ` 
* If  ` x `  is not an integer that is greater than or equal to  ` 0 ` , raise a  ` ValueError `  with the message  ` x must be an integer that is greater than or equal to 0 ` 
* If  ` x `  is greater than  ` n ` , raise a  ` ValueError `  with the message  ` x cannot be greater than n ` 
* If  ` P `  is not a 1D  ` numpy.ndarray ` , raise a  ` TypeError `  with the message  ` P must be a 1D numpy.ndarray ` 
* If  ` Pr `  is not a  ` numpy.ndarray `  with the same shape as  ` P ` , raise a  ` TypeError `  with the message  ` Pr must be a numpy.ndarray with the same shape as P ` 
* If any value in  ` P `  or  ` Pr `  is not in the range  ` [0, 1] ` , raise a  ` ValueError `  with the message  ` All values in {P} must be in the range [0, 1] `  where  ` {P} `  is the incorrect variable
* If  ` Pr `  does not sum to  ` 1 ` , raise a  ` ValueError `  with the message  ` Pr must sum to 1 ` 
* All exceptions should be raised in the above order
* Returns: the marginal probability of obtaining  ` x `  and  ` n ` 
```bash
alexa@ubuntu-xenial:0x07-bayesian_prob$ cat 2-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    marginal = __import__('2-marginal').marginal

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(marginal(26, 130, P, Pr))
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./2-main.py 
0.008229580791426582
alexa@ubuntu-xenial:0x07-bayesian_prob$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x07-bayesian_prob ` 
* File:  ` 2-marginal.py ` 
 Self-paced manual review  Panel footer - Controls 
### 3. Posterior
          mandatory         Progress vs Score  Task Body Based on   ` 2-marginal.py `  , write a function   ` def posterior(x, n, P, Pr): `   that calculates the posterior probability for the various hypothetical probabilities of developing severe side effects given the data:
*  ` x `  is the number of patients that develop severe side effects
*  ` n `  is the total number of patients observed
*  ` P `  is a 1D  ` numpy.ndarray `  containing the various hypothetical probabilities of developing severe side effects
*  ` Pr `  is a 1D  ` numpy.ndarray `  containing the prior beliefs of  ` P ` 
* If  ` n `  is not a positive integer, raise a  ` ValueError `  with the message  ` n must be a positive integer ` 
* If  ` x `  is not an integer that is greater than or equal to  ` 0 ` , raise a  ` ValueError `  with the message  ` x must be an integer that is greater than or equal to 0 ` 
* If  ` x `  is greater than  ` n ` , raise a  ` ValueError `  with the message  ` x cannot be greater than n ` 
* If  ` P `  is not a 1D  ` numpy.ndarray ` , raise a  ` TypeError `  with the message  ` P must be a 1D numpy.ndarray ` 
* If  ` Pr `  is not a  ` numpy.ndarray `  with the same shape as  ` P ` , raise a  ` TypeError `  with the message  ` Pr must be a numpy.ndarray with the same shape as P ` 
* If any value in  ` P `  or  ` Pr `  is not in the range  ` [0, 1] ` , raise a  ` ValueError `  with the message  ` All values in {P} must be in the range [0, 1] `  where  ` {P} `  is the incorrect variable
* If  ` Pr `  does not sum to  ` 1 ` , raise a  ` ValueError `  with the message  ` Pr must sum to 1 ` 
* All exceptions should be raised in the above order
* Returns: the posterior probability of each probability in  ` P `  given  ` x `  and  ` n ` , respectively
```bash
alexa@ubuntu-xenial:0x07-bayesian_prob$ cat 3-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    posterior = __import__('3-posterior').posterior

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(posterior(26, 130, P, Pr))
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./3-main.py 
[0.00000000e+00 2.99729127e-03 9.63044824e-01 3.39513268e-02
 6.55839819e-06 1.26359684e-11 1.20692303e-19 6.74011797e-31
 1.05430721e-47 1.11125368e-77 0.00000000e+00]
alexa@ubuntu-xenial:0x07-bayesian_prob$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x07-bayesian_prob ` 
* File:  ` 3-posterior.py ` 
 Self-paced manual review  Panel footer - Controls 
### 4. Continuous Posterior
          #advanced         Progress vs Score  Task Body Based on   ` 3-posterior.py `  , write a function   ` def posterior(x, n, p1, p2): `   that calculates the posterior probability that the probability of developing severe side effects falls within a specific range given the data:
*  ` x `  is the number of patients that develop severe side effects
*  ` n `  is the total number of patients observed
*  ` p1 `  is the lower bound on the range
*  ` p2 `  is the upper bound on the range
* You can assume the prior beliefs of  ` p `  follow a uniform distribution
* If  ` n `  is not a positive integer, raise a  ` ValueError `  with the message  ` n must be a positive integer ` 
* If  ` x `  is not an integer that is greater than or equal to  ` 0 ` , raise a  ` ValueError `  with the message  ` x must be an integer that is greater than or equal to 0 ` 
* If  ` x `  is greater than  ` n ` , raise a  ` ValueError `  with the message  ` x cannot be greater than n ` 
* If  ` p1 `  or  ` p2 `  are not floats within the range  ` [0, 1] ` , raise a ` ValueError `  with the message  ` {p} must be a float in the range [0, 1] `  where  ` {p} `  is the corresponding variable
* if  ` p2 `  <=  ` p1 ` , raise a  ` ValueError `  with the message  ` p2 must be greater than p1 ` 
* The only import you are allowed to use is  ` from scipy import special ` 
* Returns: the posterior probability that  ` p `  is within the range  ` [p1, p2] `  given  ` x `  and  ` n ` 
Hint: See [Binomial Distribution](https://intranet.hbtn.io/rltoken/gbQrCFCrmoyIMATa-OI__Q) 
 and  [Beta Distribution](https://intranet.hbtn.io/rltoken/0qYB6d2hO_4Yadao2mtBRg) 

```bash
alexa@ubuntu-xenial:0x07-bayesian_prob$ cat 100-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    posterior = __import__('100-continuous').posterior

    print(posterior(26, 130, 0.17, 0.23))
alexa@ubuntu-xenial:0x07-bayesian_prob$ ./100-main.py 
0.6098093274896035
alexa@ubuntu-xenial:0x07-bayesian_prob$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x07-bayesian_prob ` 
* File:  ` 100-continuous.py ` 
 Self-paced manual review  Panel footer - Controls 
