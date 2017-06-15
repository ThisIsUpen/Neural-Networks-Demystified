--optimization configs
opt = {}
opt.optimization = 'SGD'
--techniques = {'CG','SGD','ASGD','LBFGS'}
opt.learningRate = 1e-2
opt.maxIter = 1000
opt.weightDecay = 0
opt.momentum = 0.1

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

 elseif opt.optimization == 'ASGD' then
 optimState = {
    eta0 = opt.learningRate,
    t0 = 1 --N.B. trsize NOT DEFINED
 }
 optimMethod = optim.asgd

end

nn = Neural_Network(2,3,1)
nn:forward(X)
tr = Trainer(nn)
tr:train(X,y)
