
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'cunn'
dofile '../CriterionIoU.lua'
dofile '../CriterionDA.lua'
dofile '../mmd.lua'
--require 'cutorch'
--cutorch.setDevice(2)
--require '../corrloc_helper'
----------------------------------------------------------------------
-- parse command line arguments
--[[
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
   cmd:option('-batchSize', 50, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0.5, 'momentum (SGD only)')
   cmd:text()
   opt = cmd:parse(arg or {})
end
]]
opt = {}
opt.save = 'results'
opt.learningRate = 1e-3
opt.batchSize = 50
opt.weightDecay = 0
opt.momentum = 0.5


--setup criterion and model
criterion = nn.ClassNLLCriterion()
criterion:cuda()
criterionIoU = nn.MSECriterion()--nn.CriterionIoU()--SmoothL1Criterion()
criterionIoU:cuda()
criterionYt = nn.ClassNLLCriterion()
criterionYt:cuda()
criterionD = nn.mmdCriterion()

criterionD:cuda()
--model:cuda()
--attention_model = model.modules[1].modules[1].modules[1]
--classes = {1,2,3,4,5,6,7,8,9,10}
--classes
--1 Diving
--2 GolfSwing
--3 Kicking
--4 Lifting
--5 RidingHorse
--6 Running
--7 SkateBoarding
--8 SwingBench
--9 SwingSide
--10 Walking

-- This matrix records the current confusion across classes
--confusion = optim.ConfusionMatrix(classes)

-- Log results to files
--print(opt.save)
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
lossLogger = optim.Logger(paths.concat(opt.save, 'loss.log'))
corrlocLogger = optim.Logger(paths.concat(opt.save, 'corrloc.log'))
-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
parameters,gradParameters = model:getParameters()
--p,g = attention_model:getParameters()
--setting up optimizer
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = 1e-8
}
--optimMethod = optim.sgd
optimMethod = optim.adam

join = nn.JoinTable(1)
join2 = nn.JoinTable()
function compute_accuracy(prediction, ground_truth)
    --print(prediction)
    local max, pred_index = torch.max(prediction, 2)
    --print(pred_index)
    --print(ground_truth)
    --print(torch.eq(pred_index:cuda(), ground_truth))
    sample_correct = sample_correct + torch.sum(torch.eq(pred_index:cuda(), ground_truth))
    total_sample = total_sample + prediction:size(1)
end

function compute_accuracyY(prediction, ground_truth)
    local max, pred_index = torch.max(prediction, 2)
    sample_correctY = sample_correct + torch.sum(torch.eq(pred_index:cuda(), ground_truth))
    total_sampleY = total_sample + prediction:size(1)
end

function compute_proposal(atten_scores)
    local max, pred_index = torch.max(atten_scores:squeeze(), 2)
    for ix=1, pred_index:size(1) do
        print(pred_index[ix])
    end
end

function recursiveResizeAsCopyTyped(t1,t2,type)
  -- This function is borrowed from https://github.com/fmassa/object-detection.torch
  if torch.type(t2) == 'table' then
    t1 = (torch.type(t1) == 'table') and t1 or {t1}
    for key,_ in pairs(t2) do
      t1[key], t2[key] = recursiveResizeAsCopyTyped(t1[key], t2[key], type)
    end
  elseif torch.isTensor(t2) then
    local type = type or t2:type()
    t1 = torch.isTypeOf(t1,type) and t1 or torch.Tensor():type(type)
    t1:resize(t2:size()):copy(t2)
  else
    error("expecting nested tensors or tables. Got "..
    torch.type(t1).." and "..torch.type(t2).." instead")
  end
  return t1, t2
end

function iou_calc(index,proposals,targets,img_id,targets_label)
   local cnt = 0
   total_iou = torch.Tensor(index:size(1),1)
   for id=1,index:size(1) do
      local ind
      ind = index[id][1]
      local bb = proposals[id][ind]
      bb[3] = bb[1] + bb[3] -1;
      bb[4] = bb[2] + bb[4] -1;
      local max_iou = 0.005;
      local bool = 0
      if targets_label[id] == 1 then
      for id2 = 1,#targets[id] do
         local bbgt = targets[id][id2]
         --print(bbgt)
         local bbi = torch.Tensor(4):zero()
         if bb[1] < bbgt[1] then
            bbi[1] = bbgt[1]
         else
            bbi[1] = bb[1]
         end

         if bb[2] < bbgt[2] then
            bbi[2] = bbgt[2]
         else
            bbi[2] = bb[2]
         end

         if bb[3] > bbgt[3] then
            bbi[3] = bbgt[3]
         else
            bbi[3] = bb[3]
         end

         if bb[4] > bbgt[4] then
            bbi[4] = bbgt[4]
         else
            bbi[4] = bb[4]
         end

         local iw = bbi[3]-bbi[1]+1;
         local ih = bbi[4]-bbi[2]+1;
         if (iw > 0 and ih > 0) then
            bool = 1
            local bb_w = (bb[3]-bb[1]+1);
            local bb_h = (bb[4]-bb[2]+1);
            local bbgt_w = (bbgt[3]-bbgt[1]+1);
            local bbgt_h = (bbgt[4]-bbgt[2]+1);
            local intersect_area = iw*ih;
            local union_area = (bb_w * bb_h) + (bbgt_w * bbgt_h) - intersect_area;
            
            local iou = intersect_area/union_area;
            --print(intersect_area,union_area,iou,-math.log(iou),-1/iou)

            if iou > max_iou then
               max_iou = iou
            end
         else 
            --print(iw,ih)
            --print(bbi)
            --print(bb)
            --print(bbgt)
            --print(img_id[id])
            --print(targets_label[id])
         end
     
      end
      if(bool == 0) then
         cnt = cnt+1
      end
      if max_iou > 0.5 then
          total_iou[id][1] = 1
      else total_iou[id][1] = 2
      end
      --if max_iou == 0 then print(#targets[id],total_iou[id][1]) end
   else
      total_iou[id][1] = 2
   end
   
   end
   --print(cnt)
   return total_iou
end


function train()
    sample_correct = 0
    total_sample = 0
    loss = 0
    lossR = 0
    lossY = 0
    lossD = 0
    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

    -- shuffle at each epoch
    shuffle = torch.randperm(trainDataVoc:size())
    print('params:', torch.sum(parameters))
    print('shuffle:', torch.sum(shuffle[{{1, 10}}]))
    -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    
    for t = 1,trainDataVoc:size(),opt.batchSize do
    --for t = 1,2,opt.batchSize do
        -- disp progress
        xlua.progress(t, trainDataVoc:size())

        -- create mini batch
        local inputs_t = {}
        --local inputs2_t = {} 
        local targets_t = {} 
        local inputs_yto = {}
        local targets_yto = {}
        local targets_gt = {}
        local voc_proposals = {}
        local img_id = {}
        --local id = 1
        --[[if trainDataVoc:size()-t+1 >= opt.batchSize then
           targets_gt = torch.Tensor(opt.batchSize,4)
        else
           targets_gt = torch.Tensor(trainDataVoc:size()-t+1,4)
        end]]
        --print('hereeeee')
        for i = t,math.min(t+opt.batchSize-1,trainDataVoc:size()) do
            -- load new sample
            --print(shuffle[i])
            --print('HHHHHHHHHHHHH')
            local input = trainDataVoc.data[shuffle[i]]
            local target = trainDataVoc.label[shuffle[i]]
            target = target:view(-1)
            local proposal = trainDataVoc.proposal[shuffle[i]]
            table.insert(inputs_t, input[1][{{1,20},{}}])
            table.insert(targets_t, target[1])
            table.insert(targets_gt,trainDataVoc.gt[shuffle[i]])
            table.insert(voc_proposals,proposal[{{1,20},{}}])
            table.insert(img_id,trainDataVoc.image_id[shuffle[i]])
            local ind = torch.random(1,trainDataYto:size())
            input = trainDataYto.data[ind]
            target = trainDataYto.label[ind]
            target = target:view(-1)
            table.insert(inputs_yto, input[1]) 
            table.insert(targets_yto, target[1])
            --print(proposal)
            --local target_gt = trainDataVoc.gt[shuffle[i]]
            --targets_gt[id] = target_gt
            --id = id+1
        end
        --print(targets_gt)
        local inputs  = join:forward(inputs_t):clone()
        local targets = torch.Tensor(targets_t) 
        --local targetsGT = join2:forward(targets_gt):clone()
        inputs  = inputs:cuda()
        targets = targets:cuda()
        --targets_GT = targets_GT:cuda()
        --local targets_GT = torch.CudaTensor()
        --local prop = torch.CudaTensor()
        --targets_GT,targets_gt = recursiveResizeAsCopyTyped(targets_GT,targets_gt,'torch.CudaTensor')
        --prop,voc_proposals = recursiveResizeAsCopyTyped(proposals,voc_proposals,'torch.CudaTensor')
    
        local inputsY  = join:forward(inputs_yto):clone()
        local targetsY = torch.Tensor(targets_yto)
        inputsY = inputsY:cuda()
        targetsY = targetsY:cuda()
        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients 
            gradParameters:zero()

            local outputs = model:forward({inputs,inputsY})
            
            local err = criterion:forward(outputs[1], targets)
            
            local iou = iou_calc(outputs[3],voc_proposals,targets_gt,img_id,targets)
            local errR = criterionIoU:forward(iou:cuda(),targets)
            local errY = criterionYt:forward(outputs[4],targetsY)
            local errD = criterionD:forward({outputs[2],outputs[5]})     
       
            local df_do = criterion:backward(outputs[1], targets)
            local df_doIoU1 = torch.Tensor(outputs[1]:size(1),1):zero()--criterionIoU:backward(proposals,targets_GT)
            df_doI = criterionIoU:backward(iou:cuda(),targets)
            local df_doY = criterionYt:backward(outputs[4], targetsY)
            local df_doIoU2 = torch.Tensor(outputs[1]:size(1),1):zero()
            local df_doD = {}
            df_doD = criterionD:backward({outputs[2],outputs[5]}) 
            --df_doD[1] = torch.Tensor(outputs[1]:size(1),4096):zero():cuda()
            --df_doD[2] = torch.Tensor(outputs[1]:size(1),4096):zero():cuda()
            --print(df_doD) 
            --print(outputs[1]) 
            model:backward({inputs,inputsY}, {df_do,df_doD[1],df_doI:cuda(),df_doY,df_doD[2]})
            compute_accuracy(outputs[1], targets)
            compute_accuracyY(outputs[4], targetsY)
            loss = loss + err
            lossR = lossR + errR
            lossY = lossY + errY
            lossD = lossD + errD
            -- return f and df/dX
            return err+errR+errY+errD,gradParameters
        end

        -- optimize on current mini-batch
        optimMethod(feval, parameters, optimState)
        --os.exit()
    end
    xlua.progress(trainDataVoc:size(), trainDataVoc:size())

    -- time taken
    time = sys.clock() - time
    time = time / trainDataVoc:size()
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    --print(confusion)
    --accTrain = compute_localization_accuracy(trainData, attention_model)
    
    --print('train corrLoc: ' .. string.format('%.2f', accTrain) .. '%')
    print('train accuracy: ' .. string.format('%.2f',(sample_correct/total_sample)*100) .. '%')
    print('train accuracy for Yto: ' .. string.format('%.2f',(sample_correctY/total_sampleY)*100) .. '%')
    print('train loss: ' ..  loss)
    print('train loss Yto: ' ..  lossY)
    print('Domain loss: ' .. lossD)
    -- update logger/plot
    trainLogger:add{['% mean class accuracy (train set)'] = (sample_correct/total_sample) * 100}
    lossLogger:add{['loss'] = loss}
    corrlocLogger:add{['corrloc'] = accTrain} 
    
    

    -- save/log current net
    local filename = paths.concat(opt.save, 'model.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    --if (epoch % 10 == 0) then --save in every xth iteration
    if (epoch % 100 == 0) then
       print('==> saving model to '..filename)
       torch.save(filename, model)
       os.execute('sh ../evaluate.sh') --for all classes
       --os.execute('sh evaluate.sh') --for class 15-person
    end
    --if (sample_correct/total_sample) > 0.99 then
    --    os.exit()
    --end
    -- next epoch
    --confusion:zero()
    sample_correct = 0
    total_sample = 0
    epoch = epoch + 1
end
