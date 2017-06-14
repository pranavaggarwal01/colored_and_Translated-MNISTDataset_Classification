-- Problem 2c -- Implementing LeNet-5 model for translated MNIST testing dataset  - Created by Pranav Aggarwal

require 'torch'
require "nn"
require "optim"
require "image"

-- Testing
test_dataset = torch.load('mnist-p1b-test.t7')
print(test_dataset)
final_output = torch.DoubleTensor(test_dataset.data:size()[1])
local model_CNN_LeNet = torch.load('Problem2c.t7')
-- defining the creterian function 
local criterion = nn.ClassNLLCriterion()  
local classes = {'0','1','2','3','4','5','6','7','8','9'}
local confusion = optim.ConfusionMatrix(classes) -- initialising the confusion matrix

local size  = test_dataset.data:size()[1]
local bsize = 100
local tloss = 0
local traslation_x = 5;
local translation_y = 5;
average_percision = 0;
no_of_batches = 0;
final_labels = torch.DoubleTensor(size)
  --translation operation
for i = 1,size do
    test_dataset.data[i] = image.translate(test_dataset.data[i],traslation_x,translation_y)
end
for iteration  = 1,size,100 do
  error_count = 0;
  local bsize = math.min(bsize,size-iteration+1)
  local input_byte  = test_dataset.data:narrow(1,iteration,bsize)
  local target = test_dataset.label:narrow(1,iteration,bsize)
  local input = input_byte:double()

  --print(input)
  
  local test_op = model_CNN_LeNet:forward(input)
  --print(test_op)
  --print(target)
  -- if iteration == 1 then
  --   --print("Printing the outputs for the first 10 samples")
    
  --   --print ("Image_no. digit_found actual_digit  ")
  -- end
  count = 0
  for sample = iteration,iteration + bsize - 1 do
    --print(sample - iteration + 1,sample)
         
    for i = 1,10 do 
      row = test_op[sample - iteration + 1]
       
      --print("          ",math.abs(row[i]))
      if math.abs(row[i]) < 0.1 then                 
        final_labels[sample] = i - 1

        -- if iteration == 1 and count < 10 then
        --   print("    ",sample, final_labels[sample],target[sample] - 1) 
        -- end 

        -- if final_labels[sample] ~= target[sample - iteration + 1] - 1 then
        --   error_count = error_count + 1;
        -- end          
      end 
      
    end
    count = count + 1
  end
  -- average_percision = average_percision + (bsize - error_count) / bsize;
  -- no_of_batches = no_of_batches + 1;
  -- print(average_percision, no_of_batches)
  local loss   = criterion:forward(test_op,target)-- Collect Statistics
  --print(loss)
  tloss = tloss + loss * bsize
  confusion:batchAdd(test_op,target)
  io.write(string.format("progress: %4d/%4d\r",iteration,size))
  io.flush()
end
tloss = tloss / size
-- Updating the confusion matrix.
confusion:updateValids()
-- printing the mAP
print(string.format('Testing data result  loss = %.2f accuracy = %.2f',tloss,100*confusion.averageValid))
-- prinitnf confusion matrix
print(confusion)
-- Testing
print("Choose any sample")
local s = io.read()
local test = test_dataset.data:double()
image.save("problem2c.png", test:select(1,s))
print("The digit is ",final_labels[s])