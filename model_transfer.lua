require 'nn';
require 'cunn';
--dofile '../customMax.lua'
--dofile '../customMax2.lua'
dofile '../ArgMax.lua'
NUM_OF_PROPOSAL = 20
NUM_OF_FEATURES = 4096
torch.manualSeed(1)
input = torch.rand(50*NUM_OF_PROPOSAL,4096)

path1 = nn.Sequential()
path1:add(nn.Dropout(0.9))
path1:add(nn.Linear(4096,1))
path1:add(nn.View(NUM_OF_PROPOSAL))
path1:add(nn.SoftMax())
path1:add(nn.View(1,NUM_OF_PROPOSAL)) -->>  important

path2 = nn.Sequential()
path2:add(nn.Identity())
path2:add(nn.View(NUM_OF_PROPOSAL,4096)) --> important

path3 = nn.Sequential()
ct = nn.ConcatTable()
ct:add(path1)
ct:add(path2)
path3:add(ct)

ct2 = nn.ConcatTable()

part1 = nn.Sequential()
part1:add(nn.Identity())
part1:add(nn.NarrowTable(1))

ct2:add(nn.MM())
ct2:add(part1)

path3:add(ct2)

pt = nn.ParallelTable()

pt1 = nn.Sequential()
pt1:add(nn.Squeeze())
pt1:add(nn.Dropout(0.8))
domain_ct = nn.ConcatTable()
domain_ct1 = nn.Sequential()
domain_ct2 = nn.Sequential()
domain_ct2:add(nn.Identity())
domain_ct2:add(nn.SoftMax())
domain_ct1:add(nn.Linear(4096, 2))
domain_ct1:add(nn.LogSoftMax())
domain_ct:add(domain_ct1)
domain_ct:add(domain_ct2)

pt1:add(domain_ct)

pt2 = nn.Sequential()
pt2:add(nn.JoinTable(1))--.View(1,NUM_OF_PROPOSAL))
pt2:add(nn.ArgMax(2,2))

pt:add(pt1)
pt:add(pt2)

path3:add(pt)

model_base = path3
pt3 = nn.ParallelTable()
pt3:add(model_base:clone('weight', 'bias', 'gradWeight', 'gradBias'))
pt3:add(model_base:clone('weight', 'bias', 'gradWeight', 'gradBias'))

model = nn.Sequential()
model:add(pt3)
model:add(nn.FlattenTable())
model:add(nn.NarrowTable(1,5))
model = model:cuda()

trainDataYto1 = torch.load('ydataset.t7')
trainDataVoc1 = torch.load('dataset.t7')
input1 = trainDataVoc1.data[1]
input2 = trainDataYto1.data[1]
inp1 = input1[1][{{1,20},{}}]
inp2 = input2[1]
out = model:forward({input:cuda(),input:cuda()})

