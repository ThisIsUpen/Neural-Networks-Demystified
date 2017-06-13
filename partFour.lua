--[[
# Neural Networks Demystified
# Part 4: Backpropagation
#
# Supporting code for short YouTube series on artificial neural networks.
#
# Stephen Welch
# @stephencwelch
# Torch version by Alessio Salman
--]]


----------------------- Part 1 ----------------------------
th = require 'torch'
bestScore = 100

--bad written, just a way to normalize divide every column for its max value
function normalizeTensorAlongCols(tensor)
    local cols = tensor:size()[2]
    for i=1,cols do
      tensor[{ {},i }]:div(tensor:max(1)[1][i])
    end
end

-- X = (hours sleeping, hours studying), y = Score on test
torch.setdefaulttensortype('torch.FloatTensor')
X = th.Tensor({{3,5}, {5,1}, {10,2}})
y = th.Tensor({{75},{82},{93}})

--normalize
normalizeTensorAlongCols(X)
y = y/bestScore

 ----------------------- Part 4 ----------------------------
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

--[[Notes on Torch vs Numpy:
    numpy.dot == torch.mm
    numpy.multiply = torch.cmul
    the sigmoid function must not be implemented manually
    because torch already has one built-in
--]]
