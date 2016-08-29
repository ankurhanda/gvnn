local TransformationRotationSO3, parent = torch.class('nn.TransformationRotationSO3', 'nn.Module')

--[[
TransformMatrixGenerator(useRotation, useScale, useTranslation) :
TransformMatrixGenerator:updateOutput(transformParams)
TransformMatrixGenerator:updateGradInput(transformParams, gradParams)

This module can be used in between the localisation network (that outputs the
parameters of the transformation) and the AffineGridGeneratorBHWD (that expects
an affine transform matrix as input).

The goal is to be able to use only specific transformations or a combination of them.

If no specific transformation is specified, it uses a fully parametrized
linear transformation and thus expects 6 parameters as input. In this case
the module is equivalent to nn.View(2,3):setNumInputDims(2).

Any combination of the 3 transformations (rotation, scale and/or translation)
can be used. The transform parameters must be supplied in the following order:
rotation (1 param), scale (1 param) then translation (2 params).

Example:
AffineTransformMatrixGenerator(true,false,true) expects as input a tensor of
if size (B, 3) containing (rotationAngle, translationX, translationY).
]]

function TransformationRotationSO3:__init()
  parent.__init(self)

  -- if no specific transformation, use fully parametrized version
  --[[self.fullMode = not(useRotation or useScale or useTranslation)

  if not self.fullMode then
    self.useRotation = useRotation
    self.useScale = useScale
    self.useTranslation = useTranslation
  end
   ]] -- 

   self.threshold = 1e-12

end

function TransformationRotationSO3:check(input)
  --[[ if self.fullMode then
    assert(input:size(2)==7, 'Expected 7 parameters, got ' .. input:size(2))
  else ]]--
    local numberParameters = 0
    --[[if self.useRotation then
      numberParameters = numberParameters + 3
    end
    if self.useScale then
      numberParameters = numberParameters + 1
    end
    if self.useTranslation then
    end ]] --
    
    numberParameters = numberParameters + 3
      	
    assert(input:size(2)==numberParameters, 'Expected '..numberParameters..
                                            ' parameters, got ' .. input:size(2))
   --end
end

local function addOuterDim(t)
   local sizes = t:size()
   local newsizes = torch.LongStorage(sizes:size()+1)
   newsizes[1]=1
   for i=1,sizes:size() do
      newsizes[i+1]=sizes[i]
   end
   return t:view(newsizes)
end


local function dR_by_dvi(transparams, RotMats, which_vi, threshold)

      local omega_x = transparams:select(2,1)
      local omega_y = transparams:select(2,2)
      local omega_z = transparams:select(2,3)

      local omega_skew = torch.Tensor(RotMats:size()):typeAs(transparams)
      omega_skew:zero()
      omega_skew:select(3,1):select(2,2):copy(omega_z)
      omega_skew:select(3,1):select(2,3):copy(-omega_y)

      omega_skew:select(3,2):select(2,1):copy(-omega_z)
      omega_skew:select(3,2):select(2,3):copy(omega_x)

      omega_skew:select(3,3):select(2,1):copy(omega_y)
      omega_skew:select(3,3):select(2,2):copy(-omega_x)


      --print('omega_skew..')
      --print(omega_skew)
    

      local Id_minus_R_ei = torch.Tensor(RotMats:size(1),RotMats:size(2),1):zero():typeAs(transparams)      
	
      Id_minus_R_ei:select(2,which_vi):add(1)	 

      local I = torch.Tensor(RotMats:size(1), RotMats:size(2), RotMats:size(3)):zero():typeAs(transparams)

      assert(RotMats:size(2) == 3) 
      assert(RotMats:size(3) == 3)	

      I:select(2,1):select(2,1):add(1)
      I:select(2,2):select(2,2):add(1)
      I:select(2,3):select(2,3):add(1)

      Id_minus_R_ei = torch.bmm(torch.add(I,-RotMats), Id_minus_R_ei)	

    
      --print('Id_minus_R_ei..')
      --print(Id_minus_R_ei)

      --Id_minus_R_ei:select(2,1):add(1):add(-RotMats:select(2,1,1):select(2,1))	
      --Id_minus_R_ei:select(2,2):add(RotMats:select(3,1,1):select(2,2,1))	
      --Id_minus_R_ei:select(2,3):add(RotMats:select(3,1,1):select(2,3))	 

      --- cross product 

      local v_cross_Id_minus_R_ei = torch.bmm(omega_skew,Id_minus_R_ei)

      local cross_x = v_cross_Id_minus_R_ei:select(2,1)	
      local cross_y = v_cross_Id_minus_R_ei:select(2,2)	
      local cross_z = v_cross_Id_minus_R_ei:select(2,3)	

      local vcross = torch.Tensor(RotMats:size()):typeAs(transparams)
	
      vcross:zero()
      vcross:select(3,1):select(2,2):copy(cross_z)
      vcross:select(3,1):select(2,3):copy(-cross_y)

      vcross:select(3,2):select(2,1):copy(-cross_z)
      vcross:select(3,2):select(2,3):copy(cross_x)

      vcross:select(3,3):select(2,1):copy(cross_y)
      vcross:select(3,3):select(2,2):copy(-cross_x)

 
      	
     --print('v_cross_Id_minus_R_ei...')
      --print(v_cross_Id_minus_R_ei)

      --print('v_cross...')
      --print(vcross)

      local omega_mag = torch.pow(omega_x,2) + torch.pow(omega_y,2) + torch.pow(omega_z,2)

      --print('omega_mag = ')
      --print(omega_mag)

      local omega_selected = transparams:select(2,which_vi)

      for b = 1, omega_skew:size(1) do 

	if  omega_mag[b] > threshold  then

		local v_i = omega_selected[b]

    	--omega_skew[b] = torch.cdiv(torch.mm(torch.Tensor(omega_skew:size(2),omega_skew:size(3)):fill(v_i),omega_skew[b]) + vcross[b], torch.Tensor(omega_skew:size(2),omega_skew:size(3)):fill(omega_mag[b]))
    	--omega_skew[b] = torch.cdiv(omega_skew[b]:mul(v_i) + vcross[b], torch.Tensor(omega_skew:size(2),omega_skew:size(3)):fill(omega_mag[b]))
	
         --print('omega_skew_vi + vcross..')
         --check = omega_skew[b]:clone()
         --print(check:mul(v_i) + vcross[b])   
		 
         omega_skew[b] = omega_skew[b]:mul(v_i) + vcross[b]
		 omega_skew[b]:div(omega_mag[b])	
        

	else

		local e_i = torch.Tensor(3,1):typeAs(transparams):zero()
		e_i:select(1,which_vi):fill(1)

		local eMat = torch.Tensor(3,3):typeAs(transparams):zero()
	
		--[[

	
		  [a]x = ( 0  -a3  a2

			   a3   0 -a1

			  -a2  a1  0 )

		--]]



		eMat[1][2] = -e_i[3]
		eMat[1][3] =  e_i[2]	
	
		eMat[2][1] =  e_i[3]
		eMat[2][3] = -e_i[1]
	
		eMat[3][1] = -e_i[2]
		eMat[3][2] =  e_i[1]

		omega_skew[b] = eMat 
	
	end

    end

     --print('omega_sknew_new..')
     --print(omega_skew)

      return torch.bmm(omega_skew, RotMats)

      --[[ 
		from http://arxiv.org/pdf/1312.0788.pdf
      --]]

	
      --- v_i [v]x + [v x (Id - R)e_i]x 
	  ----------------------------  R 
      --- 	 ||v||^{2}

      
end


function TransformationRotationSO3:updateOutput(_tranformParams)
  local transformParams
  if _tranformParams:nDimension()==1 then
    transformParams = addOuterDim(_tranformParams)
  else
    transformParams = _tranformParams
  end

  --print('transformParams = ')
  --print(transformParams)

  self:check(transformParams)
  local batchSize = transformParams:size(1)

  --if self.fullMode then
  --  self.output = transformParams:view(batchSize, 3, 4)
  --else
    local completeTransformation = torch.zeros(batchSize,3,3):typeAs(transformParams)
    completeTransformation:select(3,1):select(2,1):add(1)
    completeTransformation:select(3,2):select(2,2):add(1)
    completeTransformation:select(3,3):select(2,3):add(1)
    --completeTransformation:select(3,4):select(2,4):add(1)
    local transformationBuffer = torch.Tensor(batchSize,3,3):typeAs(transformParams)

    local paramIndex = 1
    --if self.useRotation then
      --local alphas = transformParams:select(2, paramIndex)

      local omega_x = transformParams:select(2,paramIndex)	
      local omega_y = transformParams:select(2,paramIndex+1)	
      local omega_z = transformParams:select(2,paramIndex+2)

      --paramIndex = paramIndex + 3    

      local omega_skew = torch.Tensor(batchSize,3,3):typeAs(transformParams)
      omega_skew:zero()
      omega_skew:select(3,1):select(2,2):copy(omega_z)	
      omega_skew:select(3,1):select(2,3):copy(-omega_y)	
      
      omega_skew:select(3,2):select(2,1):copy(-omega_z)	
      omega_skew:select(3,2):select(2,3):copy(omega_x)	
      
      omega_skew:select(3,3):select(2,1):copy(omega_y)	
      omega_skew:select(3,3):select(2,2):copy(-omega_x)	

      omega_skew_sqr = torch.bmm(omega_skew,omega_skew)

      local theta_sqr = torch.pow(omega_x,2) + torch.pow(omega_y,2) + torch.pow(omega_z,2)	
      local theta = torch.pow(theta_sqr,0.5)	

      local sin_theta = torch.sin(theta)
      local sin_theta_div_theta = torch.cdiv(sin_theta,theta)

      local one_minus_cos_theta = torch.ones(theta:size()):typeAs(transformParams) - torch.cos(theta)
      --local one_minus_cos_theta = torch.add(torch.ones(theta:size()), -torch.cos(theta) )
      local one_minus_cos_div_theta_sqr = torch.cdiv(one_minus_cos_theta,theta_sqr)

      local sin_theta_div_theta_tensor  = torch.ones(omega_skew:size()):typeAs(transformParams)
      local one_minus_cos_div_theta_sqr_tensor = torch.ones(omega_skew:size()):typeAs(transformParams)

         
      for b = 1, batchSize do

	if theta_sqr[b] > self.threshold then
	
		sin_theta_div_theta_tensor[b]  = sin_theta_div_theta_tensor[b]:fill(sin_theta_div_theta[b])
		one_minus_cos_div_theta_sqr_tensor[b] = one_minus_cos_div_theta_sqr_tensor[b]:fill(one_minus_cos_div_theta_sqr[b])	
	else 

		sin_theta_div_theta_tensor[b]  = sin_theta_div_theta_tensor[b]:fill(1)
		one_minus_cos_div_theta_sqr_tensor[b] = one_minus_cos_div_theta_sqr_tensor[b]:fill(0)	
	end
 
      end 

     --- need to add boundary conditions i.e. when the size of the rot vector is very small ~ = 0      	

      completeTransformation = completeTransformation + torch.cmul(sin_theta_div_theta_tensor,omega_skew) + torch.cmul(one_minus_cos_div_theta_sqr_tensor, omega_skew_sqr)
 
  --    print (completeTransformation)

--      transformationBuffer:zero()
--      transformationBuffer:select(3,3):select(2,3):add(1)
--      local cosines = torch.cos(alphas)
--      local sinuses = torch.sin(alphas)
--      transformationBuffer:select(3,1):select(2,1):copy(cosines)
--      transformationBuffer:select(3,2):select(2,2):copy(cosines)
--      transformationBuffer:select(3,1):select(2,2):copy(sinuses)
--      transformationBuffer:select(3,2):select(2,1):copy(-sinuses)

--      completeTransformation = torch.bmm(completeTransformation, transformationBuffer)
    --end
    self.rotationOutput = completeTransformation:narrow(2,1,3):narrow(3,1,3):clone()

    --if self.useScale then
    --  local scaleFactors = transformParams:select(2,paramIndex)
    --  paramIndex = paramIndex + 1

    --  transformationBuffer:zero()
    --  transformationBuffer:select(3,1):select(2,1):copy(scaleFactors)
    --  transformationBuffer:select(3,2):select(2,2):copy(scaleFactors)
    --  transformationBuffer:select(3,3):select(2,3):add(1)

    --  completeTransformation = torch.bmm(completeTransformation, transformationBuffer)
    --end

    self.scaleOutput = completeTransformation:narrow(2,1,3):narrow(3,1,3):clone()

--    print ( self.scaleOutput ) 

    --if self.useTranslation then
    --  local txs = transformParams:select(2,paramIndex)
    --  local tys = transformParams:select(2,paramIndex+1)
    --  local tzs = transformParams:select(2,paramIndex+2)

    --  transformationBuffer:zero()
    --  transformationBuffer:select(3,1):select(2,1):add(1)
    --  transformationBuffer:select(3,2):select(2,2):add(1)
    --  transformationBuffer:select(3,3):select(2,3):add(1)
    --  transformationBuffer:select(3,4):select(2,4):add(1)
      
    --  transformationBuffer:select(3,4):select(2,1):copy(txs)
    --  transformationBuffer:select(3,4):select(2,2):copy(tys)
    --  transformationBuffer:select(3,4):select(2,3):copy(tzs)

--      print (transformationBuffer)

    --  completeTransformation = torch.bmm(completeTransformation, transformationBuffer)

--      print (completeTransformation)
    --end

    self.output=completeTransformation:narrow(2,1,3)

    print('dR_by_dv1..') 
    print(dR_by_dvi(transformParams,self.output,1, self.threshold))
    print('dR_by_dv2..') 
    print(dR_by_dvi(transformParams,self.output,2, self.threshold))
    print('dR_by_dv3..') 
    print(dR_by_dvi(transformParams,self.output,3, self.threshold))


    
--    print(self.output)
  --end

  if _tranformParams:nDimension()==1 then
    self.output = self.output:select(1,1)
  end
  return self.output
end


function TransformationRotationSO3:updateGradInput(_tranformParams, _gradParams)

  local transformParams, gradParams

  if _tranformParams:nDimension()==1 then
    transformParams = addOuterDim(_tranformParams)
    gradParams = addOuterDim(_gradParams):clone()
  else
    transformParams = _tranformParams
    gradParams = _gradParams:clone()
  end

  local batchSize = transformParams:size(1)

  if self.fullMode then

    self.gradInput = gradParams:view(batchSize, 6)

  else

    local paramIndex = transformParams:size(2)
    self.gradInput:resizeAs(transformParams)

    --[[ if self.useTranslation then

      local gradInputTranslationParams = self.gradInput:narrow(2,paramIndex-2,3)
      local tParams = torch.Tensor(batchSize, 1, 3):typeAs(transformParams)

      tParams:select(3,1):copy(transformParams:select(2,paramIndex-2))
      tParams:select(3,2):copy(transformParams:select(2,paramIndex-1))
      tParams:select(3,3):copy(transformParams:select(2,paramIndex))
      paramIndex = paramIndex-3

      local selectedOutput     = self.scaleOutput
      local selectedGradParams = gradParams:narrow(3,1,4):narrow(3,4,1):transpose(2,3)
      gradInputTranslationParams:copy(torch.bmm(selectedGradParams, selectedOutput))

      local gradientCorrection = torch.bmm(selectedGradParams:transpose(2,3), tParams)
      gradParams:narrow(3,1,3):narrow(3,1,3):add(1,gradientCorrection)

    end
    ]] --

    --[[ if self.useScale then

      local gradInputScaleparams = self.gradInput:narrow(2,paramIndex,1)
      local sParams = transformParams:select(2,paramIndex)
      paramIndex = paramIndex-1

      local selectedOutput = self.rotationOutput
      local selectedGradParams = gradParams:narrow(2,1,2):narrow(3,1,2)
      gradInputScaleparams:copy(torch.cmul(selectedOutput, selectedGradParams):sum(2):sum(3))

      gradParams:select(3,1):select(2,1):cmul(sParams)
      gradParams:select(3,2):select(2,1):cmul(sParams)
      gradParams:select(3,1):select(2,2):cmul(sParams)
      gradParams:select(3,2):select(2,2):cmul(sParams)

    end
    ]]--

    --if self.useRotation then

      --local rParams = transformParams:select(2,paramIndex)

      local rotationDerivative = torch.zeros(batchSize, 3, 3):typeAs(transformParams)

      local gradInputRotationParams = self.gradInput:narrow(2,1,1)
      
      --torch.sin(rotationDerivative:select(3,1):select(2,1),-rParams)
      --torch.sin(rotationDerivative:select(3,2):select(2,2),-rParams)
      --torch.cos(rotationDerivative:select(3,1):select(2,2),rParams)
      --torch.cos(rotationDerivative:select(3,2):select(2,1),rParams):mul(-1)

      rotationDerivative = dR_by_dvi(transformParams,self.rotationOutput,1, self.threshold)	

      local selectedGradParams = gradParams:narrow(2,1,3):narrow(3,1,3)
      gradInputRotationParams:copy(torch.cmul(rotationDerivative,selectedGradParams):sum(2):sum(3))
      
      rotationDerivative = dR_by_dvi(transformParams,self.rotationOutput,2, self.threshold)	

      --local selectedGradParams = gradParams:narrow(2,1,3):narrow(3,1,3)
      gradInputRotationParams = self.gradInput:narrow(2,2,1)

      gradInputRotationParams:copy(torch.cmul(rotationDerivative,selectedGradParams):sum(2):sum(3))
      
      rotationDerivative = dR_by_dvi(transformParams,self.rotationOutput,3, self.threshold)
	
      --local selectedGradParams = gradParams:narrow(2,1,3):narrow(3,1,3)
      gradInputRotationParams = self.gradInput:narrow(2,3,1)
      gradInputRotationParams:copy(torch.cmul(rotationDerivative,selectedGradParams):sum(2):sum(3))

    --end
  end

  if _tranformParams:nDimension()==1 then
    self.gradInput = self.gradInput:select(1,1)
  end
  return self.gradInput
end
