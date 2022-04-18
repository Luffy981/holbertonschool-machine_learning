# 0x01. Plotting
## Details
      By Alexa Orrico, Software Engineer at Holberton School          Weight: 1                Ongoing project - started Apr 18, 2022 , must end by Apr 19, 2022           - you're done with 0% of tasks.      Manual QA review must be done          (request it when you are done with the project)       ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/b4601426ad02130836f9.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T135124Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=6ec3f0464b898a04a00a599584d1bf31067c67a6c8d511886cae62eca45dd85d) 

## Resources
Read or watch :
* [Plot (graphics)](https://intranet.hbtn.io/rltoken/U-55m7o6No-_W4OJP-oTCg) 

* [Scatter plot](https://intranet.hbtn.io/rltoken/ewQvwktgrnrccqp9PInBpQ) 

* [Line chart](https://intranet.hbtn.io/rltoken/nUnDxiEeIAMxoV0Vk9dsOg) 

* [Bar chart](https://intranet.hbtn.io/rltoken/YZcEmsWNuQcQXYqzfyBfPg) 

* [Histogram](https://intranet.hbtn.io/rltoken/7icFpl6tgO6OvwSvee0S2Q) 

References :
* [Pyplot tutorial](https://intranet.hbtn.io/rltoken/9GES4KAFhBUOKYj9BI9vgQ) 

* [matplotlib.pyplot](https://intranet.hbtn.io/rltoken/GaHr4hgXE3LE3skZDGH2pQ) 

* [matplotlib.pyplot.plot](https://intranet.hbtn.io/rltoken/IUhQVdCg4MaCdUFEOuaXig) 

* [matplotlib.pyplot.scatter](https://intranet.hbtn.io/rltoken/oZ9O1frltXpknQLJGalGPg) 

* [matplotlib.pyplot.bar](https://intranet.hbtn.io/rltoken/gqW7RjVdB5G3WtuzJTcdew) 

* [matplotlib.pyplot.hist](https://intranet.hbtn.io/rltoken/K-yG7lADPJCb_FSWyOGerA) 

* [matplotlib.pyplot.xlabel](https://intranet.hbtn.io/rltoken/jhcagbtOr5Xq98SmXs8WTQ) 

* [matplotlib.pyplot.ylabel](https://intranet.hbtn.io/rltoken/jxrkMnJZTqhaRuvfIal5hQ) 

* [matplotlib.pyplot.title](https://intranet.hbtn.io/rltoken/5yPCtvA_2CSecHenfen8cQ) 

* [matplotlib.pyplot.subplot](https://intranet.hbtn.io/rltoken/ex_hmQCXTo2gHAbUFfPTyw) 

* [matplotlib.pyplot.subplots](https://intranet.hbtn.io/rltoken/3465mnzNsJp36kpDEd7tCA) 

* [matplotlib.pyplot.subplot2grid](https://intranet.hbtn.io/rltoken/6AIYCbwzqy67xdvhSzj1Aw) 

* [matplotlib.pyplot.suptitle](https://intranet.hbtn.io/rltoken/S5YwnEoLjpTYGDz5VryX6w) 

* [matplotlib.pyplot.xscale](https://intranet.hbtn.io/rltoken/Gy6aJCznMv4uSNn2LWS6rg) 

* [matplotlib.pyplot.yscale](https://intranet.hbtn.io/rltoken/XmLFrfjIS2WnwnjumbHLrg) 

* [matplotlib.pyplot.xlim](https://intranet.hbtn.io/rltoken/1zKdiptFjaMmbv8iqBVY1Q) 

* [matplotlib.pyplot.ylim](https://intranet.hbtn.io/rltoken/NDvu8opoi1B_uhJjB8SA0g) 

* [mplot3d tutorial](https://intranet.hbtn.io/rltoken/ENFsqb4q1lbSwCEUgTAt0Q) 

* [additional tutorials](https://intranet.hbtn.io/rltoken/-4sdqeB5ey_3u3htSZZQpw) 

## Learning Objectives
At the end of this project, you are expected to be able to  [explain to anyone](https://intranet.hbtn.io/rltoken/6I2Jz9J1x6X6NndNB-tT8Q) 
 ,  without the help of Google :
### General
* What is a plot?
* What is a scatter plot? line graph? bar graph? histogram?
* What is  ` matplotlib ` ?
* How to plot data with  ` matplotlib ` 
* How to label a plot
* How to scale an axis
* How to plot multiple sets of data at the same time
## Requirements
### General
* Allowed editors:  ` vi ` ,  ` vim ` ,  ` emacs ` 
* All your files will be interpreted/compiled on Ubuntu 20.04 LTS using  ` python3 `  (version 3.8)
* Your files will be executed with  ` numpy `  (version 1.19.2) and  ` matplotlib `  (version 3.3.4)
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
## More Info
### Installing Matplotlib 3.3.4
 ` pip install --user matplotlib==3.3.4
pip install --user Pillow
sudo apt-get install python3-tk
 ` To check that it has been successfully downloaded, use   ` pip list `  .
### Configure X11 Forwarding
Update your   ` Vagrantfile `   to include the following:
 ` Vagrant.configure(2) do |config|
  ...
  config.ssh.forward_x11 = true
end
 ` If you are running   ` vagrant `   on a Mac, you will have to install  [XQuartz](https://intranet.hbtn.io/rltoken/OVdbL0nPcj2IXiTQoIBwAw) 
  and restart your computer.
If you are running   ` vagrant `   on a Windows computer, you may have to follow  [these instructions](https://intranet.hbtn.io/rltoken/ZGU33rI2v1sPC_WvoXukkg) 
 .
Once complete, you should simply be able to   ` vagrant ssh `   to log into your VM and then any GUI application should forward to your local machine.
Hint for  ` emacs `  users: you will have to use  ` emacs -nw `  to prevent it from launching its GUI.
## Tasks
### 0. Line Graph
          mandatory         Progress vs Score  Task Body Complete the following source code to plot   ` y `   as a line graph:
*  ` y `  should be plotted as a solid red line
* The x-axis should range from 0 to 10
```bash
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# your code here

```
 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/664b2543b48ef4918687.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T135124Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=cbfea9ea9fe093ec7d42e5f521e529ee80c9fbdd0b62451efdf099e8064c5a65) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x01-plotting ` 
* File:  ` 0-line.py ` 
 Self-paced manual review  Panel footer - Controls 
### 1. Scatter
          mandatory         Progress vs Score  Task Body Complete the following source code to plot   ` x ↦ y `   as a scatter plot:
* The x-axis should be labeled  ` Height (in) ` 
* The y-axis should be labeled  ` Weight (lbs) ` 
* The title should be  ` Men's Height vs Weight ` 
* The data should be plotted as magenta points
```bash
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

# your code here

```
 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/1b143961d254e65afa2c.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T135124Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e6ec76cf171648d29bae960c6d827a7138ddaf5a2b884b76187e27586ea81cc6) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x01-plotting ` 
* File:  ` 1-scatter.py ` 
 Self-paced manual review  Panel footer - Controls 
### 2. Change of scale
          mandatory         Progress vs Score  Task Body Complete the following source code to plot   ` x ↦ y `   as a line graph:
* The x-axis should be labeled  ` Time (years) ` 
* The y-axis should be labeled  ` Fraction Remaining ` 
* The title should be  ` Exponential Decay of C-14 ` 
* The y-axis should be logarithmically scaled
* The x-axis should range from 0 to 28650
```bash
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# your code here

```
 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/2b6334feb069ae1b6014.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T135124Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=752bba9eac18f7ddc576c2a7e7173be8ba275b85789717fcde609aee61e7af51) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x01-plotting ` 
* File:  ` 2-change_scale.py ` 
 Self-paced manual review  Panel footer - Controls 
### 3. Two is better than one
          mandatory         Progress vs Score  Task Body Complete the following source code to plot   ` x ↦ y1 `   and   ` x ↦ y2 `   as line graphs:
* The x-axis should be labeled  ` Time (years) ` 
* The y-axis should be labeled  ` Fraction Remaining ` 
* The title should be  ` Exponential Decay of Radioactive Elements ` 
* The x-axis should range from 0 to 20,000
* The y-axis should range from 0 to 1
*  ` x ↦ y1 `  should be plotted with a dashed red line
*  ` x ↦ y2 `  should be plotted with a solid green line
* A legend labeling  ` x ↦ y1 `  as  ` C-14 `  and  ` x ↦ y2 `  as  ` Ra-226 `  should be placed in the upper right hand corner of the plot
```bash
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# your code here

```
 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/39eac4e8c8eb71469784.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T135124Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d4795a1346c8387cd05ff3a533301a694638bc0db15b30978a007b401fef9ca2) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x01-plotting ` 
* File:  ` 3-two.py ` 
 Self-paced manual review  Panel footer - Controls 
### 4. Frequency
          mandatory         Progress vs Score  Task Body Complete the following source code to plot a histogram of student scores for a project:
* The x-axis should be labeled  ` Grades ` 
* The y-axis should be labeled  ` Number of Students ` 
* The x-axis should have bins every 10 units
* The title should be  ` Project A ` 
* The bars should be outlined in black
```bash
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here

```
 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/10a48ad296d16ee8fb63.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T135124Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=6e9df0a7520a7ec790eec5bd05ebc2df2c81cbba9edc8639dd077cbae1118024) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x01-plotting ` 
* File:  ` 4-frequency.py ` 
 Self-paced manual review  Panel footer - Controls 
### 5. All in One
          mandatory         Progress vs Score  Task Body Complete the following source code to plot all 5 previous graphs in one figure:
* All axis labels and plot titles should have a font size of  ` x-small `  (to fit nicely in one figure)
* The plots should make a 3 x 2 grid
* The last plot should take up two column widths (see below)
* The title of the figure should be  ` All in One ` 
```bash
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here

```
 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/e58d423ffd060a779753.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T135124Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=407b535e8a7e4d92ee6285fa693eb4135b429bed692b5ab3e79f6a7209d1f966) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x01-plotting ` 
* File:  ` 5-all_in_one.py ` 
 Self-paced manual review  Panel footer - Controls 
### 6. Stacking Bars
          mandatory         Progress vs Score  Task Body Complete the following source code to plot a stacked bar graph:
*  ` fruit `  is a matrix representing the number of fruit various people possess* The columns of  ` fruit `  represent the number of fruit  ` Farrah ` ,  ` Fred ` , and  ` Felicia `  have, respectively
* The rows of  ` fruit `  represent the number of  ` apples ` ,  ` bananas ` ,  ` oranges ` , and  ` peaches ` , respectively

* The bars should represent the number of fruit each person possesses:* The bars should be grouped by person, i.e, the horizontal axis should have one labeled tick per person
* Each fruit should be represented by a specific color:*  ` apples `  = red
*  ` bananas `  = yellow
*  ` oranges `  = orange ( ` #ff8000 ` )
*  ` peaches `  = peach ( ` #ffe5b4 ` )
* A legend should be used to indicate which fruit is represented by each color

* The bars should be stacked in the same order as the rows of  ` fruit ` , from bottom to top
* The bars should have a width of  ` 0.5 ` 

* The y-axis should be labeled  ` Quantity of Fruit ` 
* The y-axis should range from 0 to 80 with ticks every 10 units
* The title should be  ` Number of Fruit per Person ` 
```bash
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# your code here

```
 ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/10/8058e8f96e697612d50d.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220418%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220418T135124Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=652b6e47ee0909981f0ba62bd308f86679cd0c55064bd34f33a395d7f1299b68) 

 Task URLs  Github information Repo:
* GitHub repository:  ` holbertonschool-machine_learning ` 
* Directory:  ` math/0x01-plotting ` 
* File:  ` 6-bars.py ` 
 Self-paced manual review  Panel footer - Controls 
[Done with the mandatory tasks? Unlock 2 advanced tasks now!](https://intranet.hbtn.io/projects/473/unlock_optionals) 

Ready for a  manual review
