--[[
Neural Networks Demystified
# Part 7: Overfitting, Training and Testing
#
# Supporting code for short YouTube series on artificial neural networks.
#
# Stephen Welch
# @stephencwelch
# Torch version by Alessio Salman
--]]

----------------------- Part 1 ----------------------------
th = require 'torch'
bestProfit = 6.0

--bad written, just a way to normalize divide every column for its max value
function normalizeTensorAlongCols(tensor)
   local cols = tensor:size()[2]
   for i=1,cols do
      tensor[{ {},i }]:div(tensor:max(1)[1][i])
   end
end

-- X = (num coperti, ore di apertura settimanali), y = profitto lordo annuo in percentuale
torch.setdefaulttensortype('torch.DoubleTensor')

X = th.Tensor({{22,42}, {25,38}, {30,40}})
y = th.Tensor({{2.8},{3.4},{4.2}})

--normalize
normalizeTensorAlongCols(X)
y = y/bestProfit

----------------------- Part 5 ----------------------------
--creating class NN in Lua, using a nice class utility
require 'class'

--init NN
Neural_Network = class(function(net, inputs, hiddens, outputs)
      net.inputLayerSize = inputs
      net.hiddenLayerSize = hiddens
      net.outputLayerSize = outputs
      net.W1 = th.randn(net.inputLayerSize, net.hiddenLayerSize)
      net.W2 = th.randn(net.hiddenLayerSize, net.outputLayerSize)
   end)

--define a forward method
function Neural_Network:forward(X)
   --Propagate inputs though network
   self.z2 = th.mm(X, self.W1)
   self.a2 = th.sigmoid(self.z2)
   self.z3 = th.mm(self.a2, self.W2)
   yHat = th.sigmoid(self.z3)
   return yHat
end

function Neural_Network:sigmoidPrime(z)
   --Gradient of sigmoid
   return th.exp(-z):cdiv( (th.pow( (1+th.exp(-z)),2) ) )
end

function Neural_Network:costFunction(X, y)
   --Compute the cost for given X,y, use weights already stored in class
   self.yHat = self:forward(X)
   --NB torch.sum() isn't equivalent to python sum() built-in method
   --However, for 2D arrays whose one dimension is 1, it won't make any difference
   J = 0.5 * th.sum(th.pow((y-yHat),2))
   return J
end

function Neural_Network:costFunctionPrime(X, y)
   --Compute derivative wrt to W and W2 for a given X and y
   self.yHat = self:forward(X)
   delta3 = th.cmul(-(y-self.yHat), self:sigmoidPrime(self.z3))
   dJdW2 = th.mm(self.a2:t(), delta3)

   delta2 = th.mm(delta3, self.W2:t()):cmul(self:sigmoidPrime(self.z2))
   dJdW1 = th.mm(X:t(), delta2)

   return dJdW1, dJdW2
end

--Helper Functions for interacting with other classes:
function Neural_Network:getParams()
   --Get W1 and W2 unrolled into a vector
   params = th.cat((self.W1:view(self.W1:nElement())), (self.W2:view(self.W2:nElement())))
   return params
end

function Neural_Network:setParams(params)
   --Set W1 and W2 using single paramater vector.
   W1_start = 1 --index starts at 1 in Lua
   W1_end = self.hiddenLayerSize * self.inputLayerSize
   self.W1 = th.reshape(params[{ {W1_start, W1_end} }], self.inputLayerSize, self.hiddenLayerSize)

   W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
   self.W2 = th.reshape(params[{ {W1_end+1, W2_end} }], self.hiddenLayerSize, self.outputLayerSize)
end

--this is like the getParameters(): method in the NN module of torch, i.e. compute the gradients and returns a flattened grads array
function Neural_Network:computeGradients(X, y)
   dJdW1, dJdW2 = self:costFunctionPrime(X, y)
   return th.cat((dJdW1:view(dJdW1:nElement())), (dJdW2:view(dJdW2:nElement())))
end

function computeNumericalGradient(NN, X, y)
   paramsInitial = NN:getParams()
   numgrad = th.zeros(paramsInitial:size())
   perturb = th.zeros(paramsInitial:size())
   e = 1e-4

   for p=1,paramsInitial:nElement() do
      --Set perturbation vector
      perturb[p] = e
      NN:setParams(paramsInitial + perturb)
      loss2 = NN:costFunction(X, y)

      NN:setParams(paramsInitial - perturb)
      loss1 = NN:costFunction(X, y)

      --Compute Numerical Gradient
      numgrad[p] = (loss2 - loss1) / (2*e)

      --Return the value we changed to zero:
      perturb[p] = 0
   end

   --Return Params to original value:
   NN:setParams(paramsInitial)
   return numgrad
end

----------------------- Part 6 ----------------------------
optim = require 'optim'
require 'helper'

--[[
This optimization part is strongly dependent on the libraries we're using. Therefore it will slighly be different from the python version.
The concept of a wrapper function that respects a specific API for an optimization method is still applies.
]]--

Trainer = class(function(tr, NN)
      --Make Local reference to network:
      tr.N = NN
   end)

function Trainer:train(X, y)
   --variables to keep track of the training
   --timer = th.Timer()
   local neval = 0

   params0 = self.N:getParams()
   -- create closure to evaluate f(X) and df/dX
   local feval = function(params0)
      local f = self.N:costFunction(X, y)
      print(f)
      local df_dx = self.N:computeGradients(X, y)
      neval = neval + 1
      logger:add{neval, f} --,timer:time().real}

      return f, df_dx
   end

   if optimMethod == optim.cg then
      newparams,_,_ = optimMethod(feval, params0, optimState)
   else
      for i=1,opt.maxIter do
         newparams,_,_ = optimMethod(feval, params0, optimState)
         self.N:setParams(newparams)
      end
   end
end

-------------------------optimization config---------------------------------
opt = {}
opt.optimization = 'LBFGS'
opt.learningRate = 1e-1
opt.maxIter = 200
opt.weightDecay = 0
opt.momentum = 0.0
--configure optimization method
optimMethod = configOpt(opt)

--[[
We’ll add a couple more data points to make overfitting a bit more obvious and retrain our model on the new dataset. If we re-examine our predictions across our sample space, we begin to see some strange behavior.
--]]

X2 = th.Tensor({ {3,5}, {5,1}, {10,2}, {6, 1.5} })
y2 = th.Tensor({ {75},{82},{93}, {70} })

--normalize
normalizeTensorAlongCols(X)
y2 = y2/bestProfit

nn = Neural_Network(2,3,1)
nn:forward(X2)

print(y2-yHat)

--Test network for various combinations of sleep/study:
hoursSleep = th.linspace(0, 10)
hoursStudy = th.linspace(0, 5)

--Normalize data (same way training data way normalized)
hoursSleepNorm = hoursSleep/10.
hoursStudyNorm = hoursStudy/5.

a = th.repeatTensor(hoursSleepNorm, 100)
temp = th.repeatTensor(hoursStudyNorm, 100,1)
b = temp:t():clone()
b = b:view(b:nElement())

--Join into a single input matrix:
allInputs = th.zeros(a:nElement(), 2)

allInputs[{ {},{1} }] = a
allInputs[{ {},{2} }] = b
allOutputs = nn:forward(allInputs)
--TODO: here 3D plotting in Torch is missing. Right now I don't have the time to read how to use gnuplot or iTorch.plot
--to do such plots but will be added in future.

---------------------------------------------------------------------------
--Training data
trainX = th.Tensor({ {22,42}, {25,38}, {30,40}, {27,40} })
trainY = th.Tensor({ {2.8}, {3.4}, {4.7}, {4.0} })

--Testing Data:
testX = th.Tensor({ {20, 43}, {25,42}, {10,25}, {29, 39} })
testY = th.Tensor({ {2.3}, {3.0}, {2.2}, {4.6} })

--normalize
normalizeTensorAlongCols(trainX)
normalizeTensorAlongCols(testX)
trainY = trainY/bestProfit
testY = testY/bestProfit

--Need to modify trainer class a bit to check testing error during training:
Trainer = class(function(tr, NN)
      --Make Local reference to network:
      tr.N = NN
   end)

function Trainer:train(trainX, trainY, testX, testY)
   --variables to keep track of the training
   --timer = th.Timer()
   local neval = 0

   params0 = self.N:getParams()
   -- create closure to evaluate f(X) and df/dX
   local feval = function(params0)
      local f = self.N:costFunction(trainX, trainY)
      local test = self.N:costFunction(testX, testY)
      print(f..' '..test)
      local df_dx = self.N:computeGradients(trainX, trainY)
      neval = neval + 1
      logger:add{neval, f} --,timer:time().real}
      testLogger:add{neval, test}

      return f, df_dx
   end

   if optimMethod == optim.cg then
      newparams,_,_ = optimMethod(feval, params0, optimState)
   else
      for i=1,opt.maxIter do
         newparams,_,_ = optimMethod(feval, params0, optimState)
         self.N:setParams(newparams)
      end
   end
end

--now let's train and check where exactly the net is Overfitting
nn = Neural_Network(2,3,1)
--nn:forward(trainX)

init_params = nn:getParams()
logtrain = 'train1.log'
logtest = 'test1.log'
logger = optim.Logger(logtrain)
testLogger = optim.Logger(logtest)

trainer = Trainer(nn)
--trainer:train(trainX, trainY)

trainer:train(trainX, trainY, testX, testY)

--[[
So now that we know overfitting is a problem, but how do we fix it? One way is to throw more data at the problem. A simple rule of thumb as presented by Yaser Abu-Mostaf is his excellent machine learning course available from Caltech, is that you should have at least 10 times as many examples as the degrees for freedom in your model. For us, since we have 9 weights that can change, we would need 90 observations, which we certainly don’t have.

Link to course: https://work.caltech.edu/telecourse.html

Another popular and effective way to mitigate overfitting is to use a technique called regularization. One way to implement regularization is to add a term to our cost function that penalizes overly complex models. A simple, but effective way to do this is to add together the square of our weights to our cost function, this way, models with larger magnitudes of weights, cost more. We’ll need to normalize the other part of our cost function to ensure that our ratio of the two error terms does not change with respect to the number of examples. We’ll introduce a regularization hyper parameter, lambda, that will allow us to tune the relative cost – higher values of lambda will impose bigger penalties for high model complexity.
]]

--so, the new Neural_Network class now becomes:

Neural_Network = class(function(net, inputs, hiddens, outputs, lambda)
      net.inputLayerSize = inputs
      net.hiddenLayerSize = hiddens
      net.outputLayerSize = outputs
      net.W1 = th.randn(net.inputLayerSize, net.hiddenLayerSize)
      net.W2 = th.randn(net.hiddenLayerSize, net.outputLayerSize)

      --regularization parameter
      net.lambda = lambda
   end)

--define a forward method
function Neural_Network:forward(X)
   --Propagate inputs though network
   self.z2 = th.mm(X, self.W1)
   self.a2 = th.sigmoid(self.z2)
   self.z3 = th.mm(self.a2, self.W2)
   yHat = th.sigmoid(self.z3)
   return yHat
end

function Neural_Network:sigmoidPrime(z)
   --Gradient of sigmoid
   return th.exp(-z):cdiv( (th.pow( (1+th.exp(-z)),2) ) )
end

function Neural_Network:costFunction(X, y)
   --Compute the cost for given X,y, use weights already stored in class
   self.yHat = self:forward(X)
   --NB torch.sum() isn't equivalent to python sum() built-in method
   --However, for 2D arrays whose one dimension is 1, it won't make any difference
   J = 0.5 * th.sum(th.pow((y-yHat),2))/X:size()[1] + (self.lambda/2) * (th.sum(th.pow(self.W1,2)) + th.sum(th.pow(self.W2, 2)))

   return J
end

function Neural_Network:costFunctionPrime(X, y)
   --Compute derivative wrt to W and W2 for a given X and y
   self.yHat = self:forward(X)
   delta3 = th.cmul(-(y-self.yHat), self:sigmoidPrime(self.z3))
   --Add gradient of regularization term:
   dJdW2 = th.mm(self.a2:t(), delta3)/X:size()[1] + self.lambda*self.W2

   delta2 = th.mm(delta3, self.W2:t()):cmul(self:sigmoidPrime(self.z2))
   --Add gradient of regularization term:
   dJdW1 = th.mm(X:t(), delta2)/X:size()[1] + self.lambda*self.W1

   return dJdW1, dJdW2
end

--Helper Functions for interacting with other classes:
function Neural_Network:getParams()
   --Get W1 and W2 unrolled into a vector
   params = th.cat((self.W1:view(self.W1:nElement())), (self.W2:view(self.W2:nElement())))
   return params
end

function Neural_Network:setParams(params)
   --Set W1 and W2 using single paramater vector.
   W1_start = 1 --index starts at 1 in Lua
   W1_end = self.hiddenLayerSize * self.inputLayerSize
   self.W1 = th.reshape(params[{ {W1_start, W1_end} }], self.inputLayerSize, self.hiddenLayerSize)

   W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
   self.W2 = th.reshape(params[{ {W1_end+1, W2_end} }], self.hiddenLayerSize, self.outputLayerSize)
end

--this is like the getParameters(): method in the NN module of torch, i.e. compute the gradients and returns a flattened grads array
function Neural_Network:computeGradients(X, y)
   dJdW1, dJdW2 = self:costFunctionPrime(X, y)
   return th.cat((dJdW1:view(dJdW1:nElement())), (dJdW2:view(dJdW2:nElement())))
end

--Regularization parameter
lambda = 1e-4

net = Neural_Network(2,3,1,lambda)
net:setParams(init_params)

--log training and testing data to different log files to make a comparison
logtrain = 'train2.log'
logtest = 'test2.log'
logger = optim.Logger(logtrain)
testLogger = optim.Logger(logtest)

trainer = Trainer(net)
trainer:train(trainX, trainY, testX, testY)
