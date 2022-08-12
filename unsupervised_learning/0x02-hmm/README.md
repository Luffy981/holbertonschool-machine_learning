# 0x02. Hidden Markov Models
## Details
 By: Alexa Orrico, Software Engineer at Holberton School Weight: 5Project will startAug 8, 2022 12:00 AM, must end byAug 12, 2022 12:00 AMwas released atAug 10, 2022 12:00 AM An auto review will be launched at the deadline ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/1/027d4a67aea17e6fa181.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220812%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220812T033912Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=7ab71b3aed05878be0763ffea93d985118bc013dae595bb77de743ec06d1cee6) 

## Resources
Read or watch :
* [Markov property](https://intranet.hbtn.io/rltoken/F7v-6UX8GSo7tcrLuj3pTg) 

* [Markov Chain](https://intranet.hbtn.io/rltoken/pJySWk8zYyiFBbXha1v9Uw) 

* [Properties of Markov Chains](https://intranet.hbtn.io/rltoken/tJPuYPGZmTCCiajHOHzHPg) 

* [Markov Chains](https://intranet.hbtn.io/rltoken/ek3QosV9fS9Ep7hF7Z8UNA) 

* [Markov Matrices](https://intranet.hbtn.io/rltoken/ismECln2KQ_NWqlhDi4SOA) 

* [1.3 Convergence of Regular Markov Chains](https://intranet.hbtn.io/rltoken/-P79YH94sPDmW3witwXEgA) 

* [Markov Chains, Part 1](https://intranet.hbtn.io/rltoken/Gphacn9fdFCQFGMeMyYxlg) 

* [Markov Chains, Part 2](https://intranet.hbtn.io/rltoken/flDg5iw0va1FhUjsMFHgdg) 

* [Markov Chains, Part 3](https://intranet.hbtn.io/rltoken/zRg0ddD8arH7F1hiOlaNiA) 

* [Markov Chains, Part 4](https://intranet.hbtn.io/rltoken/AD3VcrR0vmdPkLIHFCWd2Q) 

* [Markov Chains, Part 5](https://intranet.hbtn.io/rltoken/V7XdIdjg5NJpuWgV_tVk3A) 

* [Markov Chains, Part 7](https://intranet.hbtn.io/rltoken/Iyup5UA69u1UYzIsgcn4Fg) 

* [Markov Chains, Part 8](https://intranet.hbtn.io/rltoken/wXvkFVOTl3NOKWgT63odOw) 

* [Markov Chains, Part 9](https://intranet.hbtn.io/rltoken/UC94QIzIwcX280YAvJTJUA) 

* [Hidden Markov model](https://intranet.hbtn.io/rltoken/Qg8C9pzP1Yr4P8bxECb7pQ) 

* [Hidden Markov Models](https://intranet.hbtn.io/rltoken/D4kPhrRbShrDWSANnlJdkQ) 

* [(ML 14.1) Markov models - motivating examples](https://intranet.hbtn.io/rltoken/CpcwO0SbMD05S7IOfc3jeA) 

* [(ML 14.2) Markov chains (discrete-time) (part 1)](https://intranet.hbtn.io/rltoken/C-TgJ6CKgBUbL3yxfvJHqA) 

* [(ML 14.3) Markov chains (discrete-time) (part 2)](https://intranet.hbtn.io/rltoken/zMjTTG-qtP0QfcbYXFujUg) 

* [(ML 14.4) Hidden Markov models (HMMs) (part 1)](https://intranet.hbtn.io/rltoken/tMsk_K-n0mYOtsthhBrQcg) 

* [(ML 14.5) Hidden Markov models (HMMs) (part 2)](https://intranet.hbtn.io/rltoken/2k8q4yyclHlMoE83WhKf8g) 

* [(ML 14.6) Forward-Backward algorithm for HMMs](https://intranet.hbtn.io/rltoken/Qljf3X5iH7oaKWuF2I165A) 

* [(ML 14.7) Forward algorithm (part 1)](https://intranet.hbtn.io/rltoken/Tc6D_BMgvdxMWGoBtvo-Nw) 

* [(ML 14.8) Forward algorithm (part 2)](https://intranet.hbtn.io/rltoken/AMUSX-wBTAeTsvJKFlOiIQ) 

* [(ML 14.9) Backward algorithm](https://intranet.hbtn.io/rltoken/GuKHZZ4HNUS-xnbwBf8YsQ) 

* [(ML 14.10) Underflow and the log-sum-exp trick](https://intranet.hbtn.io/rltoken/uZ3KdzsuS0YmbvxDD2G-NQ) 

* [(ML 14.11) Viterbi algorithm (part 1)](https://intranet.hbtn.io/rltoken/UAmz_LJdG5w3sS_8xSAsGg) 

* [(ML 14.12) Viterbi algorithm (part 2)](https://intranet.hbtn.io/rltoken/c0LxuyQ8HeprSObqEVkTQA) 

## Learning Objectives
* What is the Markov property?
* What is a Markov chain?
* What is a state?
* What is a transition probability/matrix?
* What is a stationary state?
* What is a regular Markov chain?
* How to determine if a transition matrix is regular
* What is an absorbing state?
* What is a transient state?
* What is a recurrent state?
* What is an absorbing Markov chain?
* What is a Hidden Markov Model?
* What is a hidden state?
* What is an observation?
* What is an emission probability/matrix?
* What is a Trellis diagram?
* What is the Forward algorithm and how do you implement it?
* What is decoding?
* What is the Viterbi algorithm and how do you implement it?
* What is the Forward-Backward algorithm and how do you implement it?
* What is the Baum-Welch algorithm and how do you implement it?
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
## Tasks
### 0. Markov Chain
          mandatory         Progress vs Score  Task Body Write the function   ` def markov_chain(P, s, t=1): `   that determines the probability of a markov chain being in a particular state after a specified number of iterations:
*  ` P `  is a square 2D  ` numpy.ndarray `  of shape  ` (n, n) `  representing the transition matrix*  ` P[i, j] `  is the probability of transitioning from state  ` i `  to state  ` j ` 
*  ` n `  is the number of states in the markov chain

*  ` s `  is a  ` numpy.ndarray `  of shape  ` (1, n) `  representing the probability of starting in each state
*  ` t `  is the number of iterations that the markov chain has been through
* Returns: a  ` numpy.ndarray `  of shape  ` (1, n) `  representing the probability of being in a specific state after  ` t `  iterations, or  ` None `  on failure
```bash
alexa@ubuntu-xenial:0x02-hmm$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np
markov_chain = __import__('0-markov_chain').markov_chain

if __name__ == "__main__":
    P = np.array([[0.25, 0.2, 0.25, 0.3], [0.2, 0.3, 0.2, 0.3], [0.25, 0.25, 0.4, 0.1], [0.3, 0.3, 0.1, 0.3]])
    s = np.array([[1, 0, 0, 0]])
    print(markov_chain(P, s, 300))
alexa@ubuntu-xenial:0x02-hmm$ ./0-main.py
[[0.2494929  0.26335362 0.23394185 0.25321163]]
alexa@ubuntu-xenial:0x02-hmm$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` unsupervised_learning/0x02-hmm ` 
* File:  ` 0-markov_chain.py ` 
 Self-paced manual review  Panel footer - Controls 
### 1. Regular Chains
          mandatory         Progress vs Score  Task Body Write the function   ` def regular(P): `   that determines the steady state probabilities of a regular markov chain:
*  ` P `  is a is a square 2D  ` numpy.ndarray `  of shape  ` (n, n) `  representing the transition matrix*  ` P[i, j] `  is the probability of transitioning from state  ` i `  to state  ` j ` 
*  ` n `  is the number of states in the markov chain

* Returns: a  ` numpy.ndarray `  of shape  ` (1, n) `  containing the steady state probabilities, or  ` None `  on failure
```bash
alexa@ubuntu-xenial:0x02-hmm$ cat 1-main.py
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
    d = np.array([[0.8, 0.2, 0, 0, 0],
                [0.25, 0.75, 0, 0, 0],
                [0, 0, 0.5, 0.2, 0.3],
                [0, 0, 0.3, 0.5, .2],
                [0, 0, 0.2, 0.3, 0.5]])
    e = np.array([[1, 0.25, 0, 0, 0],
                [0.25, 0.75, 0, 0, 0],
                [0, 0.1, 0.5, 0.2, 0.2],
                [0, 0.1, 0.2, 0.5, .2],
                [0, 0.1, 0.2, 0.2, 0.5]])
    print(regular(a))
    print(regular(b))
    print(regular(c))
    print(regular(d))
    print(regular(e))
alexa@ubuntu-xenial:0x02-hmm$ ./1-main.py
None
[[0.42857143 0.57142857]]
[[0.2494929  0.26335362 0.23394185 0.25321163]]
None
None
alexa@ubuntu-xenial:0x02-hmm$ 

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` unsupervised_learning/0x02-hmm ` 
* File:  ` 1-regular.py ` 
 Self-paced manual review  Panel footer - Controls 
### 2. Absorbing Chains
          mandatory         Progress vs Score  Task Body Write the function   ` def absorbing(P): `   that determines if a markov chain is absorbing:
* P is a is a square 2D  ` numpy.ndarray `  of shape  ` (n, n) `  representing the standard transition matrix*  ` P[i, j] `  is the probability of transitioning from state  ` i `  to state  ` j ` 
*  ` n `  is the number of states in the markov chain

* Returns:  ` True `  if it is absorbing, or  ` False `  on failure
```bash
alexa@ubuntu-xenial:0x02-hmm$ cat 2-main.py
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
    e = np.array([[1, 0, 0, 0, 0],
                  [0.25, 0.75, 0, 0, 0],
                  [0, 0.1, 0.5, 0.2, 0.2],
                  [0, 0.1, 0.2, 0.5, .2],
                  [0, 0.1, 0.2, 0.2, 0.5]])
    f = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0.5, 0.5],
                  [0, 0.5, 0.5, 0]])
    print(absorbing(a))
    print(absorbing(b))
    print(absorbing(c))
    print(absorbing(d))
    print(absorbing(e))
    print(absorbing(f))
alexa@ubuntu-xenial:0x02-hmm$ ./2-main.py
True
False
False
False
True
True
alexa@ubuntu-xenial:0x02-hmm$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` unsupervised_learning/0x02-hmm ` 
* File:  ` 2-absorbing.py ` 
 Self-paced manual review  Panel footer - Controls 
### 3. The Forward Algorithm
          mandatory         Progress vs Score  Task Body  ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/1/a4a616525a089952d29f.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220812%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220812T033912Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4d1b001db11aba83963ab7897a348bed4e70a50b2326eb7e035b88fefb9746da) 

 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/1/f847db61fbc52eda75d9.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220812%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220812T033912Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=14e80ae83cd26484bce230b0b63f42f8c4f039fed25f297e3f3b83214dae60db) 

Write the function   ` def forward(Observation, Emission, Transition, Initial): `   that performs the forward algorithm for a hidden markov model:
*  ` Observation `  is a  ` numpy.ndarray `  of shape  ` (T,) `  that contains the index of the observation*  ` T `  is the number of observations

*  ` Emission `  is a  ` numpy.ndarray `  of shape  ` (N, M) `  containing the emission probability of a specific observation given a hidden state*  ` Emission[i, j] `  is the probability of observing  ` j `  given the hidden state  ` i ` 
*  ` N `  is the number of hidden states
*  ` M `  is the number of all possible observations

*  ` Transition `  is a 2D  ` numpy.ndarray `  of shape  ` (N, N) `  containing the transition probabilities*  ` Transition[i, j] `  is the probability of transitioning from the hidden state  ` i `  to  ` j ` 

*  ` Initial `  a  ` numpy.ndarray `  of shape  ` (N, 1) `  containing the probability of starting in a particular hidden state
* Returns:  ` P, F ` , or  ` None, None `  on failure*  ` P `  is the likelihood of the observations given the model
*  ` F `  is a  ` numpy.ndarray `  of shape  ` (N, T) `  containing  the forward path probabilities*  ` F[i, j] `  is the probability of being in hidden state  ` i `  at time  ` j `  given the previous observations


```bash
alexa@ubuntu-xenial:0x02-hmm$ cat 3-main.py
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
alexa@ubuntu-xenial:0x02-hmm$ ./3-main.py
1.7080966131859584e-214
[[0.00000000e+000 0.00000000e+000 2.98125000e-004 ... 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [2.00000000e-002 0.00000000e+000 3.18000000e-003 ... 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [2.50000000e-001 3.31250000e-002 0.00000000e+000 ... 2.13885975e-214
  1.17844112e-214 0.00000000e+000]
 [1.00000000e-002 4.69000000e-002 0.00000000e+000 ... 2.41642482e-213
  1.27375484e-213 9.57568349e-215]
 [0.00000000e+000 8.00000000e-004 0.00000000e+000 ... 1.96973759e-214
  9.65573676e-215 7.50528264e-215]]
alexa@ubuntu-xenial:0x02-hmm$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` unsupervised_learning/0x02-hmm ` 
* File:  ` 3-forward.py ` 
 Self-paced manual review  Panel footer - Controls 
### 4. The Viretbi Algorithm
          mandatory         Progress vs Score  Task Body Write the function   ` def viterbi(Observation, Emission, Transition, Initial): `   that calculates the most likely sequence of hidden states for a hidden markov model:
*  ` Observation `  is a  ` numpy.ndarray `  of shape  ` (T,) `  that contains the index of the observation*  ` T `  is the number of observations

*  ` Emission `  is a  ` numpy.ndarray `  of shape  ` (N, M) `  containing the emission probability of a specific observation given a hidden state*  ` Emission[i, j] `  is the probability of observing  ` j `  given the hidden state  ` i ` 
*  ` N `  is the number of hidden states
*  ` M `  is the number of all possible observations

*  ` Transition `  is a 2D  ` numpy.ndarray `  of shape  ` (N, N) `  containing the transition probabilities*  ` Transition[i, j] `  is the probability of transitioning from the hidden state  ` i `  to  ` j ` 

*  ` Initial `  a  ` numpy.ndarray `  of shape  ` (N, 1) `  containing the probability of starting in a particular hidden state
* Returns:  ` path, P ` , or  ` None, None `  on failure*  ` path `  is the a list of length  ` T `  containing the most likely sequence of hidden states
*  ` P `  is the probability of obtaining the  ` path `  sequence

```bash
alexa@ubuntu-xenial:0x02-hmm$ cat 4-main.py
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
alexa@ubuntu-xenial:0x02-hmm$ ./4-main.py
4.701733355108224e-252
[2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 0, 0, 1, 2, 2, 2, 3, 3, 3, 2, 1, 2, 1, 1, 2, 2, 2, 3, 3, 2, 2, 3, 4, 4, 3, 3, 2, 2, 3, 3, 3, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 2, 3, 3, 2, 1, 2, 1, 1, 1, 2, 2, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 3, 2, 2, 3, 2, 2, 3, 4, 4, 4, 3, 2, 1, 0, 0, 0, 1, 2, 2, 1, 1, 2, 3, 3, 2, 1, 1, 1, 2, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 3, 3, 3, 2, 1, 1, 1, 1, 2, 1, 0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 4, 4, 4, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 2, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 2, 1, 1, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3]
alexa@ubuntu-xenial:0x02-hmm$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` unsupervised_learning/0x02-hmm ` 
* File:  ` 4-viterbi.py ` 
 Self-paced manual review  Panel footer - Controls 
### 5. The Backward Algorithm
          mandatory         Progress vs Score  Task Body Write the function   ` def backward(Observation, Emission, Transition, Initial): `   that performs the backward algorithm for a hidden markov model:
*  ` Observation `  is a  ` numpy.ndarray `  of shape  ` (T,) `  that contains the index of the observation*  ` T `  is the number of observations

*  ` Emission `  is a  ` numpy.ndarray `  of shape  ` (N, M) `  containing the emission probability of a specific observation given a hidden state*  ` Emission[i, j] `  is the probability of observing  ` j `  given the hidden state  ` i ` 
*  ` N `  is the number of hidden states
*  ` M `  is the number of all possible observations

*  ` Transition `  is a 2D  ` numpy.ndarray `  of shape  ` (N, N) `  containing the transition probabilities*  ` Transition[i, j] `  is the probability of transitioning from the hidden state  ` i `  to  ` j ` 

*  ` Initial `  a  ` numpy.ndarray `  of shape  ` (N, 1) `  containing the probability of starting in a particular hidden state
* Returns:  ` P, B ` , or  ` None, None `  on failure*  ` P ` is the likelihood of the observations given the model
*  ` B `  is a  ` numpy.ndarray `  of shape  ` (N, T) `  containing  the backward path probabilities*  ` B[i, j] `  is the probability of generating the future observations from hidden state  ` i `  at time  ` j ` 


```bash
alexa@ubuntu-xenial:0x02-hmm$ cat 5-main.py
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
alexa@ubuntu-xenial:0x02-hmm$ ./5-main.py
1.7080966131859631e-214
[[1.28912952e-215 6.12087935e-212 1.00555701e-211 ... 6.75000000e-005
  0.00000000e+000 1.00000000e+000]
 [3.86738856e-214 2.69573528e-212 4.42866330e-212 ... 2.02500000e-003
  0.00000000e+000 1.00000000e+000]
 [6.44564760e-214 5.15651808e-213 8.47145100e-213 ... 2.31330000e-002
  2.70000000e-002 1.00000000e+000]
 [1.93369428e-214 0.00000000e+000 0.00000000e+000 ... 6.39325000e-002
  1.15000000e-001 1.00000000e+000]
 [1.28912952e-215 0.00000000e+000 0.00000000e+000 ... 5.77425000e-002
  2.19000000e-001 1.00000000e+000]]
alexa@ubuntu-xenial:0x02-hmm$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` unsupervised_learning/0x02-hmm ` 
* File:  ` 5-backward.py ` 
 Self-paced manual review  Panel footer - Controls 
### 6. The Baum-Welch Algorithm
          mandatory         Progress vs Score  Task Body Write the function   ` def baum_welch(Observations, Transition, Emission, Initial, iterations=1000): `   that performs the Baum-Welch algorithm for a hidden markov model:
*  ` Observations `  is a  ` numpy.ndarray `  of shape  ` (T,) `  that contains the index of the observation*  ` T `  is the number of observations

*  ` Transition `  is a  ` numpy.ndarray `  of shape  ` (M, M) `  that contains the initialized transition probabilities*  ` M `  is the number of hidden states

*  ` Emission `  is a  ` numpy.ndarray `  of shape  ` (M, N) `  that contains the initialized emission probabilities*  ` N `  is the number of output states

*  ` Initial `  is a  ` numpy.ndarray `  of shape  ` (M, 1) `  that contains the initialized starting probabilities
*  ` iterations `  is the number of times expectation-maximization should be performed
* Returns: the converged  ` Transition, Emission ` , or  ` None, None `  on failure
```bash
alexa@ubuntu-xenial:0x02-hmm$ cat 6-main.py
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
alexa@ubuntu-xenial:0x02-hmm$ ./6-main.py
[[0.81 0.19]
 [0.28 0.72]]
[[0.82 0.18 0.  ]
 [0.26 0.58 0.16]]
alexa@ubuntu-xenial:0x02-hmm$

```
With very little data (only 365 observations), we have been able to get a pretty good estimate of the transition and emission probabilities. We have not used a larger sample size in this example because our implementation does not utilize logarithms to handle values approaching 0 with the increased sequence length
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` unsupervised_learning/0x02-hmm ` 
* File:  ` 6-baum_welch.py ` 
 Self-paced manual review  Panel footer - Controls 
