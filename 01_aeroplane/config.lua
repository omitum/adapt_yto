--- All parameters goes here
config = {}

local cmd = torch.CmdLine()
cmd:text()
cmd:text('')
	-- Parameters
cmd:option('-class','aeroplane')
config = cmd:parse(arg or {})
	
