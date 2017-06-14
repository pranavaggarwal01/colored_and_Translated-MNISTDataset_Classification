-- Problem 2c -- Creating LeNet-5 model for translated MNIST dataset  - Created by Pranav Aggarwal

require 'torch'
require "nn"
require "optim"
require "image"
require "gnuplot"

training_dataset = torch.load('mnist-p1b-train.t7')
print(training_dataset)
local acc = torch.DoubleTensor(10)

-- Creation of model
local model_CNN_LeNet = nn.Sequential()
model_CNN_LeNet:add(nn.SpatialConvolution(1, 6, 5, 5));  --C1
model_CNN_LeNet:add(nn.ReLU()) -- non-linearity 
model_CNN_LeNet:add(nn.SpatialMaxPooling(2,2,2,2)) -- S2    
model_CNN_LeNet:add(nn.SpatialConvolution(6, 16, 5, 5)) -- C3
model_CNN_LeNet:add(nn.ReLU())                       -- non-linearity 
model_CNN_LeNet:add(nn.SpatialMaxPooling(2,2,2,2))   -- S4
model_CNN_LeNet:add(nn.View(16*5*5))       -- reshapes 3D tensor into 1D tensor
model_CNN_LeNet:add(nn.Linear(16*5*5, 120))             -- C5, fully connected layer starts here
model_CNN_LeNet:add(nn.ReLU())                       -- non-linearity 
model_CNN_LeNet:add(nn.Linear(120, 84))       -- F6
model_CNN_LeNet:add(nn.ReLU())                       -- non-linearity 
model_CNN_LeNet:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the model_CNN_LeNet (in this case, 10 digits)
model_CNN_LeNet:add(nn.LogSoftMax())   
parameters,gradient_parameters = model_CNN_LeNet:getParameters()

-- defining the creterian function 
local criterion = nn.ClassNLLCriterion()                 

local nEpoch = 2
for e = 1,nEpoch do

  local classes = {'0','1','2','3','4','5','6','7','8','9'}
  local confusion = optim.ConfusionMatrix(classes)

  local size  = training_dataset.data:size()[1]
  local bsize = 10
  local tloss = 0

  --translation operation
  for i = 1,size,2 do
    local translation_x = math.random(5);
    local translation_y = math.random(5);
    training_dataset.data[i] = image.translate(training_dataset.data[i],translation_x,translation_y)

  end
  for iteration  = 1,size,10 do
    local bsize = math.min(bsize,size-iteration+1)
    local input_byte  = training_dataset.data:narrow(1,iteration,bsize)
    local target = training_dataset.label:narrow(1,iteration,bsize)
    local input = input_byte:double()

    --print(input)
    gradient_parameters:zero()    -- making the gradient values 0
    local output = model_CNN_LeNet:forward(input)
    --print("helo")
    local loss   = criterion:forward(output,target)-- calculating loss
    tloss = tloss + loss * bsize
    confusion:batchAdd(output,target)
    -- Backward. The gradient wrt the parameters are internally computed.
    local gradOutput = criterion:backward(output,target)
    local gradInput  = model_CNN_LeNet:backward(input,gradOutput)

    --setting learning rate
    local function feval()
      return loss,gradient_parameters
    end
    -- Specifing the training parameters.
    local config = {
      learningRate = 0.01,
    }
    --Stochastic Gradient Descent
    optim.sgd(feval, parameters, config)
    io.write(string.format("progress: %4d/%4d\r",iteration,size))
    io.flush()
    
  end
  tloss = tloss / size
  -- Updating the confusion matrix.
  confusion:updateValids()
  
  print(string.format('epoch = %2d/%2d  loss = %.2f accuracy = %.2f',e,nEpoch,tloss,100*confusion.totalValid))

  --print(confusion)
  acc[e] = confusion.totalValid
end
--gnuplot.plot(acc)
-- Cleaning temporary data to reduce the size of the model_CNN_LeNet file.
model_CNN_LeNet:clearState()
-- Saving the model_CNN_LeNet.
torch.save('Problem2c.t7',model_CNN_LeNet)