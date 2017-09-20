require 'nn';
require 'cunn';
dofile '../ArgMax.lua'
matio = require 'matio';

model = torch.load('results/model.net')
model:evaluate()
dataset = torch.load('ydataset.t7')
--attention_model = model.modules[1].modules[1]
attention_model = model.modules[1].modules[1].modules[1].modules[1]
n_data = #dataset.data;
join = nn.JoinTable(1);

selproposal = torch.Tensor(n_data)
proposals = torch.Tensor(n_data,20, 4)
frame_ix = torch.Tensor(n_data)
label = torch.Tensor(n_data)
gt = torch.Tensor(n_data,4)

for ix=1, n_data do
    local input = dataset.data[ix][1]:cuda()
    local atten_scores = attention_model:forward(input)
    --print(atten_scores)
    local atten_max, atten_indices = torch.max(atten_scores, 2)
    
    selproposal[ix] = atten_indices[1][1]
    --print('.......')
    --print(torch.type(dataset.proposal[ix]))
    if torch.type(dataset.proposal[ix]) == 'table' then 
        proposals[ix] = join:forward(dataset.proposal[ix])
    else proposals[ix] = dataset.proposal[ix]
    end
    gt[ix] = dataset.gt1[ix]
    label[ix] = dataset.label[ix]
end

data = {}
data.selproposal = selproposal
data.proposals = proposals
data.label = label
data.gt = gt
matio.save('model_result.mat', data)
    

