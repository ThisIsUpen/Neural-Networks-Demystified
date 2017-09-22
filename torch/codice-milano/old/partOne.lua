--[[
# Neural Networks Demystified
# Part 2: Forward Propagation
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

--bad written, I think this isn't the proper way to do it in Torch'
--Just a way to normalize, dividing every column for its max value
--equivalent to : X = X/np.amax(X, axis=0) with Numpy
function normalizeTensorAlongCols(tensor)
   local cols = tensor:size()[2]
   for i=1,cols do
      tensor[{ {},i }]:div(tensor:max(1)[1][i])
   end
end

-- X = (hours sleeping, hours studying), y = Score on test
torch.setdefaulttensortype('torch.DoubleTensor')
X = th.Tensor({{3,5}, {5,1}, {10,2}})
y = th.Tensor({{75},{82},{93}})

--normalize
normalizeTensorAlongCols(X)
y = y/bestScore
