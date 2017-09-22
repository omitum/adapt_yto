--- All parameters goes here
config = {}

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Detectors')
cmd:text()
cmd:text('')
	-- Parameters
cmd:option('-class','cow')
config = cmd:parse(arg or {})
	
