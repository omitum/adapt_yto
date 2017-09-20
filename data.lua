
local xml = require 'xml'
dofile 'config.lua'
if trainDataYto == nil then
   trainDataYto = torch.load('ydataset.t7')
   function trainDataYto:size()
      return #self.data
   end
end

if trainDataVoc == nil then
   trainDataVoc = torch.load('dataset.t7') 
   function trainDataVoc:size() 
      return #self.data
   end
end
ANNO_PATH = '../../dataset/VOCdevkit/VOC2007/Annotations/'
annopath = paths.concat(ANNO_PATH,'%s.xml')

local function parsePascalAnnotation(ann,ind,parent)
  local res = {}
  for i,j in ipairs(ann) do
    if #j == 1 then
      res[j.xml] = j[1]
    else
      local sub = parsePascalAnnotation(j,i,j.xml)
      if not res[j.xml] then
        res[j.xml] = sub
      elseif #res[j.xml] == 0 then
        res[j.xml] = {res[j.xml]}
        table.insert(res[j.xml],sub)
      else
        table.insert(res[j.xml],sub)
      end
    end
  end
  return res
end


function getAnnotation(image_id)
  --print(self.annopath,self.img_ids[i])
  local ann = xml.loadpath(string.format(annopath,image_id))
  local parsed = parsePascalAnnotation(ann,1,{})
  if parsed.object and #parsed.object == 0 then
    parsed.object = {parsed.object}
  end
  return parsed
end
--torch.Tensor(trainDataVoc:size(),4)
gt = {}
local one = 0
for i=1,trainDataVoc:size() do
   gt1 = {}
   local image_id = trainDataVoc.image_id[i][1]
   parsed = getAnnotation(image_id)
   num_gt = 0
   for idx,obj in ipairs(parsed.object) do
      num_gt = num_gt + 1
   end
   
   
   local gt_boxes = torch.IntTensor()
   gt_boxes:resize(num_gt,4)
   for idx = 1,num_gt do
    gt_boxes[idx][1] = parsed.object[idx].bndbox.xmin
    gt_boxes[idx][2] = parsed.object[idx].bndbox.ymin
    gt_boxes[idx][3] = parsed.object[idx].bndbox.xmax
    gt_boxes[idx][4] = parsed.object[idx].bndbox.ymax
  end
  lab = torch.sum(trainDataVoc.label[i][1]:int())
  id = 1
  if num_gt == 1 then
     gt1[id] = gt_boxes[1]
     --one = one+1
  elseif lab == 1 then
   
     for idx,obj in ipairs(parsed.object) do
         
        if parsed.object[idx].name == config.class then
           --print(parsed.object[idx].name)
           gt1[id] = gt_boxes[idx]
           id = id+1
           --one = one+1
           --print(config.class)
           --break
        end
     end
  else
    
     for idx,obj in ipairs(parsed.object) do 
        gt1[id] = gt_boxes[idx]
        id = id+1
     end
  end
  gt[i] = gt1
end

trainDataVoc.gt = gt
--print(trainDataVoc:size())
--print(trainDataYto:size())
print(trainDataVoc.image_id[14])
print(trainDataVoc.gt[14])
--print(trainDataVoc.gt[14][1])
--print(trainDataVoc.gt[14][2])
--print(trainDataVoc.gt[14][5])
--print(trainDataVoc.gt[1][3])
--print(trainDataVoc.label[262])
--trainDataVoc1 = {}
--[[
gt2 = {}
label2 = {}
image_id2 = {}
proposal2 = {}
data2 = {}
local ii = 0
for i=1,trainDataVoc:size() do
    lab = trainDataVoc.label[i]
    lab = lab:view(-1)
    if lab[1] == 1 then
        --print(i)
        ii = ii+1
        gt2[ii] = trainDataVoc.gt[i]
        label2[ii] = trainDataVoc.label[i]
        image_id2[ii] = trainDataVoc.image_id[i]
        proposal2[ii] = trainDataVoc.proposal[i]
        data2[ii] = trainDataVoc.data[i]
    end
end

trainDataVoc.gt = gt2
trainDataVoc.data = data2
trainDataVoc.proposal = proposal2
trainDataVoc.image_id = image_id2
trainDataVoc.label = label2
print(trainDataVoc:size())
--function trainDataVoc:size()
  --  return #self.data
--end

]]
