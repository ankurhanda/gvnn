-- you can easily test specific units like this:
-- th -lnn -e "nn.test{'LookupTable'}"
-- th -lnn -e "nn.test{'LookupTable', 'Add'}"

--require 'stn'

--require 'torch'

--function main()

local mytester = torch.Tester()
local jac
local sjac

local precision = 1e-5
local expprecision = 1e-4

local gvnntest = {}

--function gvnntest.TestB()
--    local a = 10
--    local b = 9
--    tester:assertlt(a,b,'a < b')
--    tester:assertgt(a,b,'a > b')
--end


function gvnntest.AffineGridGeneratorBHWD_batch()
   local nframes = torch.random(2,10)
   local height = torch.random(2,5)
   local width = torch.random(2,5)
   local input = torch.zeros(nframes, 2, 3):uniform()
   local module = nn.AffineGridGeneratorBHWD(height, width)

   local err = jac.testJacobian(module,input)

   print ('\n')  
   print (err) 
   print ('\n')  
   print ('\n')  
   print ('\n')  
   print ('\n')  
   print ('\n')  

   mytester:assertlt(err,precision, 'error on state ')
   -- mytester:assertgt(err,precision, 'error on state ')
	
   -- IO
   local ferr,berr = jac.testIO(module,input)
 
   print (ferr)
 
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end

function gvnntest.AffineGridGeneratorBHWD_single()
   local height = torch.random(2,5)
   local width = torch.random(2,5)
   local input = torch.zeros(2, 3):uniform()
   local module = nn.AffineGridGeneratorBHWD(height, width)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end

function gvnntest.DenseAffineOpticFlowBHWD_single()

   local height = torch.random(2,5)
   local width  = torch.random(2,5)
   --local input  = torch.zeros(1, height,width,6):uniform()
   local input  = torch.zeros(1, height,width,6):uniform()

   local module = nn.DenseAffineOpticFlowBHWD(height, width)
	
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end

function gvnntest.NonRigidSE2_single()


   local height = torch.random(2,5)
   local width  = torch.random(2,5)
   --local input  = torch.zeros(1, height,width,6):uniform()
   local input  = torch.zeros(1, height,width,3):uniform()

   local module = nn.NonRigidSE2BHWD(height, width)
	
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end


function gvnntest.NonRigidRotationSO3_single()
    

    local height = 3--torch.random(2,5)
    local width  = 4--torch.random(2,5)

    local input = torch.zeros(2,height,width,3):uniform()

    --input[1][1][1][1] = 0.1090
    --input[1][1][1][2] = 0.8495
    --input[1][1][1][3] = 0.9774

    --print('input..')
    --print(input)

    local module = nn.NonRigidRotationSO3()


    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')

    -- IO
    local ferr,berr = jac.testIO(module,input)
    mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
    mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end


function gvnntest.NonRigidPerPixelSE3_single()


    local height = 3--torch.random(2,5)
    local width  = 4--torch.random(2,5)

    local input = torch.zeros(2,height,width,6):uniform()

                                            --input[1][1][1][1] = 0.1090
                                            --input[1][1][1][2] = 0.8495
                                            --input[1][1][1][3] = 0.9774

    --print('input..')
--print(input)

    local module = nn.NonRigidPerPixelSE3()


    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')

    -- IO
    local ferr,berr = jac.testIO(module,input)
    mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
    mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end



function gvnntest.Transform3DPoints_R_single()
   local height = torch.random(2,5)
   local width = torch.random(2,5)
   local input = torch.zeros(3, 3, 3):uniform()
   
   local fx = 10
   local fy = 10
   local u0 = 2
   local v0 = 2
	
	
   local module = nn.Transform3DPoints_R(height, width, fx, fy, u0, v0)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end

function gvnntest.Transform3DPoints_depth_single()
   local height = 3--torch.random(2,5)
   local width  = 3--torch.random(2,5)
   local depth  = torch.zeros(1, 3, 3):uniform()

   local fx = 1
   local fy = 1
   local u0 = 0
   local v0 = 0


   local module = nn.Transform3DPoints_depth(height, width, fx, fy, u0, v0)

   local err = jac.testJacobian(module,depth)
   mytester:assertlt(err,precision, 'error on state ')

   -- IO
   --local ferr,berr = jac.testIO(module,input)
   --mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   --mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end



function gvnntest.Transform3DPoints_Rt_single()
   local height = 3--torch.random(2,5)
   local width =  3--torch.random(2,5)
   local batchsize = 4
   local inputMatrix = torch.Tensor(batchsize, 3, 4):uniform()

   local fx = 1
   local fy = 1
   local u0 = 0
   local v0 = 0

   local depth  = torch.Tensor(batchsize,height,width):uniform()
   local module = nn.Transform3DPoints_Rt(height, width, fx, fy, u0, v0)

   module._updateOutput = module.updateOutput
   function module:updateOutput(input)
     return self:_updateOutput({input, depth})
   end

   module._updateGradInput = module.updateGradInput
   function module:updateGradInput(input, gradOutput)
      self:_updateGradInput({input, depth}, gradOutput)
      return self.gradInput[1]
   end

   local errInputMatrix = jac.testJacobian(module,inputMatrix)
   mytester:assertlt(errInputMatrix,precision, 'error on state ')

   -- test grids (second element of input table)
   function module:updateOutput(input)
      return self:_updateOutput({inputMatrix, input})
   end

   function module:updateGradInput(input, gradOutput)
      self:_updateGradInput({inputMatrix, input}, gradOutput)
      return self.gradInput[2]
   end

   local err = jac.testJacobian(module,depth)
   mytester:assertlt(err,precision, 'error on state ')


   -- IO
   --local ferr,berr = jac.testIO(module,input)
   --mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   --mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end

function gvnntest.BilinearSamplerBHWD_single()
   local height = torch.random(1,5)
   local width = torch.random(1,5)
   local channels = torch.random(1,6)
   local inputImages = torch.zeros(height, width, channels):uniform()
   local grids = torch.zeros(height, width, 2):uniform()
   local module = nn.BilinearSamplerBHWD()

   -- test input images (first element of input table)
   module._updateOutput = module.updateOutput
   function module:updateOutput(input)
      return self:_updateOutput({input, grids})
   end
   
   module._updateGradInput = module.updateGradInput
   function module:updateGradInput(input, gradOutput)
      self:_updateGradInput({input, grids}, gradOutput)
      return self.gradInput[1]
   end

   local errImages = jac.testJacobian(module,inputImages)
   mytester:assertlt(errImages,precision, 'error on state ')

   -- test grids (second element of input table)
   function module:updateOutput(input)
      return self:_updateOutput({inputImages, input})
   end
   
   function module:updateGradInput(input, gradOutput)
      self:_updateGradInput({inputImages, input}, gradOutput)
      return self.gradInput[2]
   end

   local errGrids = jac.testJacobian(module,grids)
   mytester:assertlt(errGrids,precision, 'error on state ')
end


function gvnntest.BilinearSamplerBHWD_batch()
   local nframes = torch.random(2,10)
   local height = torch.random(1,5)
   local width = torch.random(1,5)
   local channels = torch.random(1,6)
   local inputImages = torch.zeros(nframes, height, width, channels):uniform()
   local grids = torch.zeros(nframes, height, width, 2):uniform()
   local module = nn.BilinearSamplerBHWD()

   -- test input images (first element of input table)
   module._updateOutput = module.updateOutput
   function module:updateOutput(input)
      return self:_updateOutput({input, grids})
   end
   
   module._updateGradInput = module.updateGradInput
   function module:updateGradInput(input, gradOutput)
      self:_updateGradInput({input, grids}, gradOutput)
      return self.gradInput[1]
   end

   local errImages = jac.testJacobian(module,inputImages)
   mytester:assertlt(errImages,precision, 'error on state ')

   -- test grids (second element of input table)
   function module:updateOutput(input)
      return self:_updateOutput({inputImages, input})
   end
   
   function module:updateGradInput(input, gradOutput)
      self:_updateGradInput({inputImages, input}, gradOutput)
      return self.gradInput[2]
   end

   local errGrids = jac.testJacobian(module,grids)
   mytester:assertlt(errGrids,precision, 'error on state ')
end


function gvnntest.AffineTransformMatrixGenerator_batch()
   -- test all possible transformations
   for _,useRotation in pairs{true,false} do
      for _,useScale in pairs{true,false} do
         for _,useTranslation in pairs{true,false} do
            local currTest = ''
            if useRotation then currTest = currTest..'rotation ' end
            if useScale then currTest = currTest..'scale ' end
            if useTranslation then currTest = currTest..'translation' end
            if currTest=='' then currTest = 'full' end

            local nbNeededParams = 0
            if useRotation then nbNeededParams = nbNeededParams + 1 end
            if useScale then nbNeededParams = nbNeededParams + 1 end
            if useTranslation then nbNeededParams = nbNeededParams + 2 end
            if nbNeededParams == 0 then nbNeededParams = 6 end -- full affine case

            local nframes = torch.random(2,10)
            local params = torch.zeros(nframes,nbNeededParams):uniform()
            local module = nn.AffineTransformMatrixGenerator(useRotation,useScale,useTranslation)

            local err = jac.testJacobian(module,params)
            mytester:assertlt(err,precision, 'error on state for test '..currTest)

            -- IO
            local ferr,berr = jac.testIO(module,params)
            mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err for test '..currTest)
            mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err for test '..currTest)

         end
      end
   end
end

function gvnntest.AffineTransformMatrixGenerator_single()
   -- test all possible transformations
   for _,useRotation in pairs{true,false} do
      for _,useScale in pairs{true,false} do
         for _,useTranslation in pairs{true,false} do
            local currTest = ''
            if useRotation then currTest = currTest..'rotation ' end
            if useScale then currTest = currTest..'scale ' end
            if useTranslation then currTest = currTest..'translation' end
            if currTest=='' then currTest = 'full' end

            local nbNeededParams = 0
            if useRotation then nbNeededParams = nbNeededParams + 1 end
            if useScale then nbNeededParams = nbNeededParams + 1 end
            if useTranslation then nbNeededParams = nbNeededParams + 2 end
            if nbNeededParams == 0 then nbNeededParams = 6 end -- full affine case

            local params = torch.zeros(nbNeededParams):uniform()
            local module = nn.AffineTransformMatrixGenerator(useRotation,useScale,useTranslation)

            local err = jac.testJacobian(module,params)
            mytester:assertlt(err,precision, 'error on state for test '..currTest)

            -- IO
            local ferr,berr = jac.testIO(module,params)
            mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err for test '..currTest)
            mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err for test '..currTest)

         end
      end
   end
end

function gvnntest.PHCP_single()

	    local n_batches = 1--torch.random(2,3)
	    local height    = 10--torch.random(2,5)
	    local width     = 15--torch.random(2,5)

            local module = nn.PinHoleCameraProjectionBHWD(height, width)

            local params = torch.Tensor(n_batches,height,width,3):uniform()-0.5

	    --print(params)

            local err = jac.testJacobian(module,params)
            mytester:assertlt(err,precision, 'error on state for test ')

            -- IO
            local ferr,berr = jac.testIO(module,params)
            mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err for test ')
            mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err for test ')
end


function gvnntest.TransformationRotationSO3_single()

	local height=30
	local width =40
	local fx = 30
	local fy = 30
	local u0 = 20
	local v0 = 15

	local module = nn.Sequential()
	module:add(nn.TransformationRotationSO3())
	module:add(nn.Transform3DPoints_R(height,width,fx,fy,u0,v0))
	module:add(nn.PinHoleCameraProjectionBHWD(height,width,fx,fy, u0,v0))

	local params = torch.Tensor(1,3):zero()--uniform()*0.001

	local err = jac.testJacobian(module, params)
	mytester:assertlt(err,precision, 'error on state for test')

        -- IO
        local ferr,berr = jac.testIO(module,params)
        mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err for test ')
        mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err for test ')


end


function gvnntest.TransformationMatrix3x4GeneratorSO3_single()
   -- test all possible transformations
   -- for _,useRotation in pairs{false} do
   --   for _,useScale in pairs{false} do
   --      for _,useTranslation in pairs{true} do
   --         if useRotation then currTest = currTest..'rotation ' end
   --          if useScale then currTest = currTest..'scale ' end
   --         if useTranslation then currTest = currTest..'translation' end
   --         if currTest=='' then currTest = 'full' end
		
              local currTest = 'T3x4'
		
   --         local nbNeededParams = 0
   --         if useRotation then nbNeededParams = nbNeededParams + 1 end
   --         if useScale then nbNeededParams = nbNeededParams + 1 end
   --         if useTranslation then nbNeededParams = nbNeededParams + 3 end
   --         if nbNeededParams == 0 then nbNeededParams = 6 end -- full affine case

            --local params = torch.zeros(3):uniform()

--	    print ('---before---\n')
--            print (params)

            local params = torch.Tensor(10,6):uniform()
	    --local 	
	
	    --params[1]=1

	    --local q_0 = params:select(2,1)
	    --local q_1 = params:select(2,2)
	    --local q_2 = params:select(2,3)
	    --local q_3 = params:select(2,4)

--	    local q_sqr = torch.pow(q_0,2) + torch.pow(q_1,2) + torch.pow(q_2,2) + torch.pow(q_3,2)

--	    local q_norm = torch.pow(q_sqr,0.5)	

--	    print ('---q_norm---\n')
--	    print ( q_norm )	

--	    params:div(q_norm)
	
--	    print ('---after---\n')
	    print ( params ) 	
--
            --local module = nn.TransformationMatrix3x4Euler(true,false,false)
            local module = nn.TransformationMatrix3x4SO3(true,false,true)

            local err = jac.testJacobian(module,params)
            mytester:assertlt(err,precision, 'error on state for test '..currTest)

            -- IO
            local ferr,berr = jac.testIO(module,params)
            mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err for test '..currTest)
            mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err for test '..currTest)

   --      end
   --   end
   --end
end



mytester:add(gvnntest)

--if not nn then
--   require 'nn'
--   print('')
 --  print('Testing nn with type = cuda')
--   print('')
 --  jac = nn.Jacobian
--   sjac = nn.SparseJacobian
--   mytester:run()

--else
   jac = nn.Jacobian
   sjac = nn.SparseJacobian
   function gvnn.test(tests)
      -- randomize stuff
   print('')
   print('Testing nn with type = cuda')
   print('')
      math.randomseed(os.time())
      mytester:run(tests)
       return mytester
   end
--end

--stn.test("TransformationMatrix3x4GeneratorSO3_single")
--stn.test("DenseAffineOpticFlowBHWD_single")
--stn.test("NonRigidSE2_single")
--stn.test("NonRigidRotationSO3_single")
gvnn.test("NonRigidPerPixelSE3_single")
--stn.test("PHCP_single")
--stn.test("TransformationRotationSO3_single")
--stn.test("Transform3DPoints_R_single")
--stn.test("Transform3DPoints_Rt_single")
--stn.test("Transform3DPoints_depth_single")

--end

--main()
