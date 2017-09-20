--require 'cunn'

--torch.manualSeed(1)

----------------------------------------------------------------------
print '==> executing all'

--dofile 'config.lua'
dofile '../data.lua'
dofile '../model_transfer.lua'
dofile '../train_transfer.lua'

----------------------------------------------------------------------
print '==> training!'
--os.execute('rm local_result.mat')
MAX_ITER = 60000
--for ix=1, MAX_ITER do
while(true) do
   train()
   --test()
end
