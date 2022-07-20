# 0x05. Advanced Linear Algebra
## Details
      By Alexa Orrico, Software Engineer at Holberton School          Weight: 4                Ongoing project - started Jul 18, 2022 , must end by Jul 22, 2022           - you're done with 0% of tasks.              Checker will be released at Jul 20, 2022 12:00 AM        An auto review will be launched at the deadline       ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/dfef9b5a1411d49808c1.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220720%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220720T023206Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=9f294d9efd5a2d519ab35d2eb6c0d7728eb99cbd4ea81d56642e38f6c8db8a64) 

## Resources
Read or watch :
* [The determinant | Essence of linear algebra](https://intranet.hbtn.io/rltoken/F61LWMDnv216TI4VURIavQ) 

* [Determinant of a Matrix](https://intranet.hbtn.io/rltoken/d_uY4uoRqk7EU4Pg9qz7Ig) 

* [Determinant](https://intranet.hbtn.io/rltoken/eLZ_J1BAltkAFuNsOhutKQ) 

* [Determinant of an empty matrix](https://intranet.hbtn.io/rltoken/5REQT4orNW4FNpoDdy0RLg) 

* [Inverse matrices, column space and null space](https://intranet.hbtn.io/rltoken/5alpJ5Uyeb8QWaWmKeIFcA) 

* [Inverse of a Matrix using Minors, Cofactors and Adjugate](https://intranet.hbtn.io/rltoken/nAtBpV2gGB1PhMtfw8JWww) 

* [Minor](https://intranet.hbtn.io/rltoken/IAjFHGF9pgUlajyVY0ytaA) 

* [Cofactor](https://intranet.hbtn.io/rltoken/KykXUKmSVLjYmT7z_Nc07g) 

* [Adjugate matrix](https://intranet.hbtn.io/rltoken/30NiZqFhPtoe61DUmXkUCA) 

* [Singular Matrix](https://intranet.hbtn.io/rltoken/Wdpp4xtLOgxFq5kGyBpZsw) 

* [Elementary Matrix Operations](https://intranet.hbtn.io/rltoken/fAyfVsVuokMPK-z9FF7bjw) 

* [Gaussian Elimination](https://intranet.hbtn.io/rltoken/VTo7Q09EqloHCv-q6XGw3g) 

* [Gauss-Jordan Elimination](https://intranet.hbtn.io/rltoken/CMLsBAfm0W4uqe2r_Oezkw) 

* [Matrix Inverse](https://intranet.hbtn.io/rltoken/2Oz0LITPFPX9PTNn9zqrwQ) 

* [Eigenvectors and eigenvalues | Essence of linear algebra](https://intranet.hbtn.io/rltoken/E8_Gy9NO7Gjb4LxN0jUFWQ) 

* [Eigenvalues and eigenvectors](https://intranet.hbtn.io/rltoken/n1w31RZHRl8_tNhUY1-TRA) 

* [Eigenvalues and Eigenvectors](https://intranet.hbtn.io/rltoken/ZSkHYcwTonsySFhCBrM8uA) 

* [Definiteness of a matrix](https://intranet.hbtn.io/rltoken/qODLYtIIrUMQLsEENialNQ) 
Up to Eigenvalues
* [Definite, Semi-Definite and Indefinite Matrices](https://intranet.hbtn.io/rltoken/cuiQxdTiqu_4T4k2kVVelQ) 
Ignore Hessian Matrices
* [Tests for Positive Definiteness of a Matrix](https://intranet.hbtn.io/rltoken/rYtty7OT1dlzKac_-QGPhA) 

* [Positive Definite Matrices and Minima](https://intranet.hbtn.io/rltoken/S-Yndhr3Pr-rTmzGgiIHTQ) 

* [Positive Definite Matrices](https://intranet.hbtn.io/rltoken/j-ejEoIpd7Cg6SxMlChpew) 

As references :
* [numpy.linalg.eig](https://intranet.hbtn.io/rltoken/AiJWrZH4RCkUydqzEeut-A) 

## Learning Objectives
At the end of this project, you are expected to be able to  [explain to anyone](https://intranet.hbtn.io/rltoken/zW40Gzvr0rGN8TBZ33Cqzw) 
 ,  without the help of Google :
### General
* What is a determinant? How would you calculate it?
* What is a minor, cofactor, adjugate? How would calculate them?
* What is an inverse? How would you calculate it?
* What are eigenvalues and eigenvectors? How would you calculate them?
* What is definiteness of a matrix? How would you determine a matrix’s definiteness?
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
* Unless otherwise noted, you are not allowed to import any module
* All your files must be executable
* The length of your files will be tested using  ` wc ` 
### Quiz questions
Great!          You've completed the quiz successfully! Keep going!          (Show quiz)#### 
        
        Question #0
    
 Quiz question Body What is the determinant of the following matrix?
[[  -7,   0,   6 ]   [   5,  -2, -10 ]   [   4,   3,   2 ]]
 Quiz question Answers * -44

* 44

* 14

* -14

 Quiz question Tips #### 
        
        Question #1
    
 Quiz question Body What is the minor of the following matrix?
[[  -7,   0,   6 ]   [   5,  -2, -10 ]   [   4,   3,   2 ]]
 Quiz question Answers * [[ 26, 50, 23 ]        [ -18, -38, -21 ]        [ 12, 40, 15 ]]

* [[ 26, 50, 23 ]        [ -18, -38, -21 ]        [ 12, 40, 14 ]]

* [[ 26, 50, 23 ]        [ -18, -39, -21 ]        [ 12, 40, 14 ]]

* [[ 26, 50, 23 ]        [ -18, -39, -21 ]        [ 12, 40, 15 ]]

 Quiz question Tips #### 
        
        Question #2
    
 Quiz question Body What is the cofactor of the following matrix?
[[ 6, -9, 9 ],   [ 7, 5, 0 ],   [ 4, 3, -8 ]]
 Quiz question Answers * [[ -40, 56, 1 ],        [ -45, -84, -54 ],        [ -45, 64, 93 ]]

* [[ -40, 56, 1 ],         [ -44, -84, -54 ],         [ -45, 64, 93 ]]

* [[ -40, 56, 1 ],         [ -44, -84, -54 ],         [ -45, 63, 93 ]]

* [[ -40, 56, 1 ],         [ -45, -84, -54 ],         [ -45, 63, 93 ]]

 Quiz question Tips #### 
        
        Question #3
    
 Quiz question Body What is the adjugate of the following matrix?
[[ -4, 1, 9 ],   [ -9, -8, -5 ],   [ -3, 8, 10 ]]
 Quiz question Answers * [[ -40, 62, 67 ],         [ 105, -13, -101 ],         [ -97, 29, 41 ]]

* [[ -40, 62, 67 ],         [ 105, -14, -101 ],         [ -97, 29, 41 ]]

* [[ -40, 62, 67 ],        [ 105, -13, -101 ],        [ -96, 29, 41 ]]

* [[ -40, 62, 67 ],         [ 105, -14, -101 ],         [ -96, 29, 41 ]]

 Quiz question Tips #### 
        
        Question #4
    
 Quiz question Body Is the following matrix invertible? If so, what is its inverse?
[[  1,  0,  1 ]   [  2,  1,  2 ]   [  1,  0, -1 ]]
 Quiz question Answers * [[ 0.5, 0,   0.5 ]        [    0,  1,     2  ]        [ 0.5,  0,  0.5 ]]

* [[ 0.5, 0,   0.5 ]        [   -2,  1,     0  ]        [ 0.5,  0, -0.5 ]]

* [[ 0.5, 0,   0.5 ]        [    2,  1,     0  ]        [ 0.5,  0,  0.5 ]]

* It is singular

 Quiz question Tips #### 
        
        Question #5
    
 Quiz question Body Is the following matrix invertible? If so, what is its inverse?
[[ 2, 1, 2 ]   [ 1, 0, 1 ]   [ 4, 1, 4 ]]
 Quiz question Answers * [[ 4, 1, 2 ]        [ 1, 0, 1 ]        [ 4, 1, 2 ]]

* [[ 2, 1, 4 ]        [ 1, 0, 1 ]        [ 2, 1, 4 ]]

* [[ 4, 1, 4 ]        [ 1, 0, 1 ]         [ 2, 1, 2 ]]

* It is singular

 Quiz question Tips #### 
        
        Question #6
    
 Quiz question Body Given  ` A `   = [[-2, -4, 2],           [-2, 1, 2],           [4, 2, 5]]  ` v `   = [[2], [-3], [-1]] Where   ` v `   is an eigenvector of   ` A `  , calculate   ` A `  10  ` v ` 
 Quiz question Answers * [[118098], [-177147], [-59049]]

* [[2097152], [-3145728], [-1048576]]

* [[2048], [-3072], [-1024]]

* None of the above

 Quiz question Tips #### 
        
        Question #7
    
 Quiz question Body Which of the following are also eigenvalues (  ` λ `  ) and eigenvectors (  ` v `  ) of   ` A `   where  ` A `   = [[-2, -4, 2],           [-2, 1, 2],           [4, 2, 5]]
 Quiz question Answers *  ` λ `   = 5;   ` v `   = [[2], [1], [1]] 

*  ` λ `   = -5;   ` v `   = [[-2], [-1], [1]] 

*  ` λ `   = -3;   ` v `   = [[4], [-2], [3]] 

*  ` λ `   = 6;   ` v `   = [[1], [6], [16]] 

 Quiz question Tips #### 
        
        Question #8
    
 Quiz question Body What is the definiteness of the following matrix:
[[ -1, 2, 0 ]   [ 2, -5, 2 ]   [ 0, 2, -6 ]]
 Quiz question Answers * Positive definite

* Positive semi-definite

* Negative semi-definite

* Negative definite

* Indefinite

 Quiz question Tips #### 
        
        Question #9
    
 Quiz question Body What is the definiteness of the following matrix:
[[ 2, 2, 1 ]   [ 2, 1, 3 ]   [ 1, 3, 8 ]]
 Quiz question Answers * Positive definite

* Positive semi-definite

* Negative semi-definite

* Negative definite

* Indefinite

 Quiz question Tips #### 
        
        Question #10
    
 Quiz question Body What is the definiteness of the following matrix:
[[ 2, 1, 1 ]   [ 1, 2, -1 ]   [ 1, -1, 2 ]]
 Quiz question Answers * Positive definite

* Positive semi-definite

* Negative semi-definite

* Negative definite

* Indefinite

 Quiz question Tips ## Tasks
### 0. Determinant
          mandatory         Progress vs Score  Task Body Write a function   ` def determinant(matrix): `   that calculates the determinant of a matrix:
*  ` matrix `  is a list of lists whose determinant should be calculated
* If  ` matrix `  is not a list of lists, raise a  ` TypeError `  with the message  ` matrix must be a list of lists ` 
* If  ` matrix `  is not square, raise a  ` ValueError `  with the message  ` matrix must be a square matrix ` 
* The list  ` [[]] `  represents a  ` 0x0 `  matrix
* Returns: the determinant of  ` matrix ` 
```bash
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ cat 0-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    determinant = __import__('0-determinant').determinant

    mat0 = [[]]
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(determinant(mat0))
    print(determinant(mat1))
    print(determinant(mat2))
    print(determinant(mat3))
    print(determinant(mat4))
    try:
        determinant(mat5)
    except Exception as e:
        print(e)
    try:
        determinant(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./0-main.py 
1
5
-2
0
192
matrix must be a list of lists
matrix must be a square matrix
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x05-advanced_linear_algebra ` 
* File:  ` 0-determinant.py ` 
 Self-paced manual review  Panel footer - Controls 
### 1. Minor
          mandatory         Progress vs Score  Task Body Write a function   ` def minor(matrix): `   that calculates the minor matrix of a matrix:
*  ` matrix `  is a list of lists whose minor matrix should be calculated
* If  ` matrix `  is not a list of lists, raise a  ` TypeError `  with the message  ` matrix must be a list of lists ` 
* If  ` matrix `  is not square or is empty, raise a  ` ValueError `  with the message  ` matrix must be a non-empty square matrix ` 
* Returns: the minor matrix of  ` matrix ` 
```bash
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ cat 1-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    minor = __import__('1-minor').minor

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(minor(mat1))
    print(minor(mat2))
    print(minor(mat3))
    print(minor(mat4))
    try:
        minor(mat5)
    except Exception as e:
        print(e)
    try:
        minor(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./1-main.py 
[[1]]
[[4, 3], [2, 1]]
[[1, 1], [1, 1]]
[[-12, -36, 0], [10, -34, -32], [47, 13, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x05-advanced_linear_algebra ` 
* File:  ` 1-minor.py ` 
 Self-paced manual review  Panel footer - Controls 
### 2. Cofactor
          mandatory         Progress vs Score  Task Body Write a function   ` def cofactor(matrix): `   that calculates the cofactor matrix of a matrix:
*  ` matrix `  is a list of lists whose cofactor matrix should be calculated
* If  ` matrix `  is not a list of lists, raise a  ` TypeError `  with the message  ` matrix must be a list of lists ` 
* If  ` matrix `  is not square or is empty, raise a  ` ValueError `  with the message  ` matrix must be a non-empty square matrix ` 
* Returns: the cofactor matrix of  ` matrix ` 
```bash
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ cat 2-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    cofactor = __import__('2-cofactor').cofactor

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(cofactor(mat1))
    print(cofactor(mat2))
    print(cofactor(mat3))
    print(cofactor(mat4))
    try:
        cofactor(mat5)
    except Exception as e:
        print(e)
    try:
        cofactor(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./2-main.py 
[[1]]
[[4, -3], [-2, 1]]
[[1, -1], [-1, 1]]
[[-12, 36, 0], [-10, -34, 32], [47, -13, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x05-advanced_linear_algebra ` 
* File:  ` 2-cofactor.py ` 
 Self-paced manual review  Panel footer - Controls 
### 3. Adjugate
          mandatory         Progress vs Score  Task Body Write a function   ` def adjugate(matrix): `   that calculates the adjugate matrix of a matrix:
*  ` matrix `  is a list of lists whose adjugate matrix should be calculated
* If  ` matrix `  is not a list of lists, raise a  ` TypeError `  with the message  ` matrix must be a list of lists ` 
* If  ` matrix `  is not square or is empty, raise a  ` ValueError `  with the message  ` matrix must be a non-empty square matrix ` 
* Returns: the adjugate matrix of  ` matrix ` 
```bash
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ cat 3-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    adjugate = __import__('3-adjugate').adjugate

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(adjugate(mat1))
    print(adjugate(mat2))
    print(adjugate(mat3))
    print(adjugate(mat4))
    try:
        adjugate(mat5)
    except Exception as e:
        print(e)
    try:
        adjugate(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./3-main.py 
[[1]]
[[4, -2], [-3, 1]]
[[1, -1], [-1, 1]]
[[-12, -10, 47], [36, -34, -13], [0, 32, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x05-advanced_linear_algebra ` 
* File:  ` 3-adjugate.py ` 
 Self-paced manual review  Panel footer - Controls 
### 4. Inverse
          mandatory         Progress vs Score  Task Body Write a function   ` def inverse(matrix): `   that calculates the inverse of a matrix:
*  ` matrix `  is a list of lists whose inverse should be calculated
* If  ` matrix `  is not a list of lists, raise a  ` TypeError `  with the message  ` matrix must be a list of lists ` 
* If  ` matrix `  is not square or is empty, raise a  ` ValueError `  with the message  ` matrix must be a non-empty square matrix ` 
* Returns: the inverse of  ` matrix ` , or  ` None `  if  ` matrix `  is singular
```bash
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ cat 4-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    inverse = __import__('4-inverse').inverse

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(inverse(mat1))
    print(inverse(mat2))
    print(inverse(mat3))
    print(inverse(mat4))
    try:
        inverse(mat5)
    except Exception as e:
        print(e)
    try:
        inverse(mat6)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./4-main.py 
[[0.2]]
[[-2.0, 1.0], [1.5, -0.5]]
None
[[-0.0625, -0.052083333333333336, 0.24479166666666666], [0.1875, -0.17708333333333334, -0.06770833333333333], [0.0, 0.16666666666666666, -0.08333333333333333]]
matrix must be a list of lists
matrix must be a non-empty square matrix
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x05-advanced_linear_algebra ` 
* File:  ` 4-inverse.py ` 
 Self-paced manual review  Panel footer - Controls 
### 5. Definiteness
          mandatory         Progress vs Score  Task Body Write a function   ` def definiteness(matrix): `   that calculates the definiteness of a matrix:
*  ` matrix `  is a  ` numpy.ndarray `  of shape  ` (n, n) `  whose definiteness should be calculated
* If  ` matrix `  is not a  ` numpy.ndarray ` , raise a  ` TypeError `  with the message  ` matrix must be a numpy.ndarray ` 
* If  ` matrix `  is not a valid matrix, return  ` None ` 
* Return: the string  ` Positive definite ` ,  ` Positive semi-definite ` ,  ` Negative semi-definite ` ,  ` Negative definite ` , or  ` Indefinite `  if the matrix is positive definite, positive semi-definite, negative semi-definite, negative definite of indefinite, respectively
* If  ` matrix `  does not fit any of the above categories, return  ` None ` 
* You may  ` import numpy as np ` 
```bash
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ cat 5-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    definiteness = __import__('5-definiteness').definiteness
    import numpy as np

    mat1 = np.array([[5, 1], [1, 1]])
    mat2 = np.array([[2, 4], [4, 8]])
    mat3 = np.array([[-1, 1], [1, -1]])
    mat4 = np.array([[-2, 4], [4, -9]])
    mat5 = np.array([[1, 2], [2, 1]])
    mat6 = np.array([])
    mat7 = np.array([[1, 2, 3], [4, 5, 6]])
    mat8 = [[1, 2], [1, 2]]

    print(definiteness(mat1))
    print(definiteness(mat2))
    print(definiteness(mat3))
    print(definiteness(mat4))
    print(definiteness(mat5))
    print(definiteness(mat6))
    print(definiteness(mat7))
    try:
        definiteness(mat8)
    except Exception as e:
        print(e)
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./5-main.py 
Positive definite
Positive semi-definite
Negative semi-definite
Negative definite
Indefinite
None
None
matrix must be a numpy.ndarray
alexa@ubuntu-xenial:0x05-advanced_linear_algebra$

```
 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x05-advanced_linear_algebra ` 
* File:  ` 5-definiteness.py ` 
 Self-paced manual review  Panel footer - Controls 
