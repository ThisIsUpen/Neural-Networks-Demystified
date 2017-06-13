--[[
# Neural Networks Demystified
# Part 1: Data + Architecture
#
# Supporting code for short YouTube series on artificial neural networks.
#
# Stephen Welch
# @stephencwelch
# Torch version by Alessio Salman
--]]

th = require 'torch'
bestScore = 100

-- X = (hours sleeping, hours studying), y = Score on test
torch.setdefaulttensortype('torch.FloatTensor')
X = th.Tensor({{3,5}, {5,1}, {10,2}})
y = th.Tensor({{75},{82},{93}})

--normalize
normalizeTensorAlongDim(2,X)
y = y/bestScore

--bad written, just a way to normalize divide every column for its max value
function normalizeTensorAlongCols(tensor)
    local cols = tensor:size()[2]
    for i=1,cols do
      tensor[{ {},i }]:div(tensor:max(1)[1][i])
    end
end
