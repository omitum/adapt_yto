-- Some notes here:
-- Seems nn.Parallel is more fitted, than the nn.concat
-- But how to make it as input..
-- for the input, simply concatenate them
-- for the code of layer
-- actually we need no 'nn.Parallel'
-- Maybe not so complicated
-- Maybe refer to the Chinese blog we've seen 
-- Triple loss...
require 'nn'
require 'cunn'

local mmdCriterion, Parent = torch.class('nn.mmdCriterion', 'nn.Criterion')

-- todo-hy: making the c++ style into camel style

function mmdCriterion:__init(kernel_num, kernel_mul, fix_gamma)
    Parent.__init(self)
    self.kernel_num = kernel_num or 5
    self.kernel_mul = kernel_mul or 2.0
    self.fix_gamma = fix_gamma or false
    self.loss_weight = 1
    self.gradInput = {}
end


-- We first write in cpu tensor, then see if we can change by :cuda() directly
-- Expected input: a table with [1] source: self.dim * self.source_num 
-- and [2] target: self.dim * self.target_num
-- todo-hy: can we get rid of target?
function mmdCriterion:updateOutput(input)
    -- init

    
    local source = input[1]
    local target = input[2]
    assert(source:size(2) == target:size(2), "size not match")
    self.dim = source:size(2)
    self.source_num = source:size(1)
    self.target_num = target:size(1)
    self.total_num = self.source_num + self.target_num
    self.gamma = torch.Tensor(1)
    self.kernel_val = {}

    for i=1, self.kernel_num do
        self.kernel_val[i] = torch.Tensor(self.total_num, self.total_num) 
    end

    -- start forward
    -- NOTE: when the tensor is assigned row by row, it must be explicitly 
    -- declared as CudaTensor, even we can criterion:cuda()
    -- Hence, when use cpu version, this line should be modified
    self.distance2 = torch.CudaTensor(self.total_num, self.total_num)
     -- self.distance2 = torch.Tensor(self.total_num, self.total_num)


    -- todo-hy: totally can be paralleled...
    for i=1, self.total_num do
        local ind = 0
        local single_input = -1
        if i > self.source_num then
            ind = i - self.source_num
            single_input = target[ind]
        else
            ind = i
            single_input = source[ind]
        end 
        single_input = single_input:view(self.dim, 1)
        local single_expand = torch.expand(single_input, self.dim, self.total_num)

        -- single_expand: dim * total_num
        local input_concat = torch.cat(source, target)
        local single_diff2 = torch.pow(single_expand - input_concat:transpose(1,2), 2) -- dim * total_num
        local sumup = torch.sum(single_diff2, 1) -- the ith row of distance2
        self.distance2[i] = sumup 
    end
    local bandwidth = torch.sum(self.distance2)
    self.gamma = (self.total_num - 1) * self.total_num / bandwidth

    -- calculate each kernel
    local gamma_times = torch.pow(self.kernel_mul, torch.floor(self.kernel_num / 2))
    local kernel_gamma = self.gamma / gamma_times
    for i=1, self.kernel_num do
        self.kernel_val[i] = torch.exp(self.distance2 * (-kernel_gamma))
        kernel_gamma = kernel_gamma * self.kernel_mul
    end

    -- calculate mmd loss
    -- todo-hy: this can also be paralleled 
    local loss = 0
    local sample_num = self.source_num
    for i=1, sample_num do       
        local s1 = torch.random(1, self.source_num)
        local s2 = torch.random(1, self.source_num)
        if s1 == s2 then
            s2 = s2 % self.source_num + 1
        end
        local t1 = torch.random(1, self.target_num)
        local t2 = torch.random(1, self.target_num)
        if t1 == t2 then
            t2 = t2 % self.target_num + 1
        end
        for i=1, self.kernel_num do
            loss = loss + self.kernel_val[i][s1][s2]
            loss = loss + self.kernel_val[i][t1+self.source_num][t2+self.source_num]
            loss = loss - self.kernel_val[i][s1][t2+self.source_num]
            loss = loss - self.kernel_val[i][s2][t1+self.source_num]
        end
    end
    return loss

    

end -- of the function

function calculateDiff( coeff, num1, num2, input_data1, input_data2, input_diff1, input_diff2, kernel_num, kernel_mul, gamma, loss_weight, sample_num )
    local f1 = input_data1[num1]
    local f2 = input_data2[num2]
    local tmp1 = f1 - f2
    local tmp2 = f2 - f1
    local square_sum = tmp1 * tmp1
    local times = torch.pow(kernel_mul, torch.floor(kernel_num / 2))
    local temp_gamma = gamma / times
    local factor_for_diff = 0
    for i=1, kernel_num do
        local temp_n = (-temp_gamma) * square_sum
        temp_n = torch.exp(temp_n) * coeff
        temp_n = (-2) * temp_gamma * temp_n
        factor_for_diff = factor_for_diff + temp_n
        temp_gamma = temp_gamma * kernel_mul
    end
    tmp1 = tmp1 * loss_weight * factor_for_diff / sample_num 
    tmp2 = tmp2 * loss_weight * factor_for_diff / sample_num 
    input_diff1[num1] = input_diff1[num1] + tmp1
    input_diff2[num2] = input_diff2[num2] + tmp2
end

function mmdCriterion:updateGradInput(input)
    local source = input[1]
    local target = input[2]

    -- NOTE: also explicitly cudaTensor
    local source_diff = torch.CudaTensor()
    local target_diff = torch.CudaTensor()
    -- local source_diff = torch.Tensor()
    -- local target_diff = torch.Tensor()

    source_diff:resize(self.source_num, self.dim)
    source_diff:zero()
    target_diff:resize(self.target_num, self.dim)
    target_diff:zero()
    assert(source:size(2) == target:size(2), "size not match")
    local sample_num = self.source_num
    for i=1, sample_num do       
        local s1 = torch.random(1, self.source_num)
        local s2 = torch.random(1, self.source_num)
        if s1 == s2 then
            s2 = s2 % self.source_num + 1
        end
        local t1 = torch.random(1, self.target_num)
        local t2 = torch.random(1, self.target_num)
        if t1 == t2 then
            t2 = t2 % self.target_num + 1
        end
        calculateDiff(1, s1, s2, source, source, source_diff, source_diff, self.kernel_num, self.kernel_mul, self.gamma, self.loss_weight, sample_num )
        calculateDiff(-1, s1, t2, source, target, source_diff, target_diff, self.kernel_num, self.kernel_mul, self.gamma, self.loss_weight, sample_num )
        calculateDiff(-1, t1, s2, target, source, target_diff, source_diff, self.kernel_num, self.kernel_mul, self.gamma, self.loss_weight, sample_num )
        calculateDiff(1, t1, t2, target, target, target_diff, target_diff, self.kernel_num, self.kernel_mul, self.gamma, self.loss_weight, sample_num )

    end

    self.gradInput[1] = source_diff
    self.gradInput[2] = target_diff
    -- print(torch.mean(source_diff))
    return self.gradInput

end -- of the function

function mmdCriterion:accGradParameters(input, gradOutput)
end

function mmdCriterion:reset()
end
