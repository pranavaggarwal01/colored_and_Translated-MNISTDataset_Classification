-- Problem 2b -- Creating LeNet-5 model for colored  background MNIST dataset  - Created by Pranav Aggarwal

require 'torch'
require "nn"
require "optim"
require "image"

training_dataset = torch.load('mnist-p2b-train.t7')
print(training_dataset)

--print(classes[training_dataset.label[100]])
--itorch.image(training_dataset.data[100]) -- display the 100-th image in dataset

--print(training_dataset.data:narrow(1,1,1))
-- Creation of model
local model_CNN_LeNet = nn.Sequential()
model_CNN_LeNet:add(nn.SpatialConvolution(3, 6, 5, 5));  --C1    , 3 depth input
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
model_CNN_LeNet:add(nn.Linear(84, 11))                   -- 11 is the number of outputs of the model_CNN_LeNet (in this case, 10 digits)
model_CNN_LeNet:add(nn.LogSoftMax())   
parameters,gradient_parameters = model_CNN_LeNet:getParameters()

-- defining the creterian function 
local criterion = nn.ClassNLLCriterion()                 

local nEpoch = 30
for e = 1,nEpoch do

	local classes = {'0','1','2','3','4','5','6','7','8','9','No_digit'}
	local confusion = optim.ConfusionMatrix(classes) -- intialising confusion matrix

  local size  = training_dataset.data:size()[1]
  local bsize = 100
  local tloss = 0

  for iteration  = 1,size,100 do
    local bsize = math.min(bsize,size-iteration+1)
    local input1  = training_dataset.data:narrow(1,iteration,bsize)
    local target = training_dataset.label:narrow(1,iteration,bsize)
    --print(target)
    local input = input1:double()

    gradient_parameters:zero()                  -- making the gradient values 0

    local output = model_CNN_LeNet:forward(input)

    local loss   = criterion:forward(output,target)-- calculating loss
    tloss = tloss + loss * bsize
    confusion:batchAdd(output,target)
    -- Backward. The gradient wrt the parameters are internally computed.
    local gradOutput = criterion:backward(output,target)
    local gradInput  = model_CNN_LeNet:backward(input,gradOutput)

    local function feval()
      return loss,gradient_parameters
    end
    -- Specifing the training parameters - learning rate.
    local config = {
      learningRate = 0.001,
    }
    -- Stochastic Gradient Descent
    optim.sgd(feval, parameters, config)

    io.write(string.format("progress: %4d/%4d\r",iteration,size))
    io.flush()
  end
  tloss = tloss / size
  -- Updating the confusion matrix.
  confusion:updateValids()
 
  print(string.format('epoch = %2d/%2d  loss = %.2f accuracy = %.2f',e,nEpoch,tloss,100*confusion.totalValid))

end

-- Clean temporary data to reduce the size of the model_CNN_LeNet file.
model_CNN_LeNet:clearState()
-- Save the model_CNN_LeNet.
torch.save('Problem2b.t7',model_CNN_LeNet)