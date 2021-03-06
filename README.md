# CS289A_ML_HW1
UC Berkeley CS289A Machine Learning

Download and install Python, PyCharm, and Anaconda3 (make sure to add to PATH when installing). We will make a Virtualenv Environment in PyCharm using python.exe provided by Anaconda3 as the Project Interpreter.

## Running in PyCharm
<p align="center">
  <img width="500" src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/1.PNG">
</p>

Click "Git".

<p align="center">
  <img src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/2.PNG">
</p>

Copy the link here.

<p align="center">
  <img width="500" src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/3.PNG">
</p>

We need to set up the Virtualenv Environment.

<p align="center">
  <img src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/4.png">
</p>

<p align="left">
  Click "Settings" or press <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>S</kbd>.
</p>

<p align="center">
  <img src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/5.png">
</p>

We need to add a Project Interpreter. Click "Add...".

<p align="center">
  <img width="750" src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/6.PNG">
</p>

We need to install packages (numpy, matplotlib, scikit-learn, and scipy). Click "+" near the top right. (Disregard the left portion of the image showing that the packages are already installed.)

<p align="center">
  <img width="750" src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/7.PNG">
</p>

Search all four packages (numpy, matplotlib, scikit-learn, and scipy), and click "Install Packages". In case you want to "Specify version" to install and older version, you can.

<p align="center">
  <img width="750" src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/8.PNG">
</p>

Now you should see all four packages available for use in your project.

<p align="center">
  <img width="750" src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/9.png">
</p>

Brief Github tutorial:

<p align="left">
  For large files (.mat) in this case, Git LFS should be used. Download and install Git LFS (make sure to add to PATH when installing). In Command Prompt, cd to local repository. Enter the following commands: <br />
  
  git config http.postBuffer 2097152000 <br />
  git lfs install <br />
  git lfs track *.mat
</p>

<p align="left">
  Click "VCS"&rarr;"Commit..." or the green "&check;" symbol.
</p>

<p align="center">
  <img src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/10.png">
</p>

Click "Commit and Push...".

<p align="center">
  <img src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/11.png">
</p>

Click "Push"

<p align="center">
  <img width="500" src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/12.PNG">
</p>

In case you add extra folders or files, they may be classified as "Unversioned Files" (will not be pushed to Github). Click "Version Control" near the bottom left.

<p align="center">
  <img src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/13.png">
</p>

Click "browse".

<p align="center">
  <img src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/14.png">
</p>

Right click whatever folder or file you wish to be pushed to Github, and click "Add to VCS".

<p align="center">
  <img src="https://github.com/georgebzhang/CS289A_ML_HW1/blob/master/Readme_Images/15.png">
</p>

Then commit and push as you would normally.



