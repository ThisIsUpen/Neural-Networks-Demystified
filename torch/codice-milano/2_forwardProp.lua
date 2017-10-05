--[[
Build & Train MLP from scratch in Torch. 
Alessio Salman     

2. Forward propagation
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
y = th.Tensor({{2.8},{3.4},{4.4}})

--normalize
normalizeTensorAlongCols(X)
y = y/bestProfit

----------------------- Part 2 ----------------------------
--creating class NN in Lua, using a torch class 
local class = require 'class'

Neural_Network = class('Neural_Network')

--init NN
function Neural_Network:__init(inputs, hiddens, outputs) 
	self.inputLayerSize = inputs
	self.hiddenLayerSize = hiddens
	self.outputLayerSize = outputs
	self.W1 = th.randn(self.inputLayerSize, self.hiddenLayerSize)
	self.W2 = th.randn(self.hiddenLayerSize, self.ouputLayerSize)
end

--Note: I didn't implement manually the sigmoid function as Torch has one built-in.
--define a forward method
function Neural_Network:forward(X)
   --Propagate inputs though network
   self.z2 = th.mm(X, self.W1)
   self.a2 = th.sigmoid(self.z2)
   self.z3 = th.mm(self.a2, self.W2)
   yHat = th.sigmoid(self.z3)
   return yHat
end

