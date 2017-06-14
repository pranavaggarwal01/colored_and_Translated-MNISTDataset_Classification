Pranav Aggarwal
Date - 4th December 2016
Note: 
- All the programs are implemented with the following details:
Programming language - Lua
IDE – SublimeText       //Download link - https://sublimetext.com/2
Compiler – Torch   (implemented on Terminal of Ubuntu 16.04)
Operating system – Ubuntu 16.04
OS type – 64 bit


List of files

--Colored_train.lua
This program creates LeNet-5 model for colored background MNIST dataset

To execute
th Colored_train.lua


--Colored_test.lua 
This program implements the model created in Colored_train.lua

To execute
th Colored_test.lua


 
--Translated_train.lua
This program creates LeNet-5 model for translated MNIST dataset

To execute
th Translated_train.lua


--Translated_test.lua 
This program implements the model created in Translated_train.lua

To execute
th Translated_test.lua 



To display images I have used interative torch which can be started by typing "qlua -lenv" on the terminal. Then used the following code to display the images
For example This will display the first 10 digits from the above mentioned dataset
require "image"
t7> d1 = torch.load('mnist-p1b-test.t7')
t7> image.display{image=d1.data:narrow(1,1,10)}



