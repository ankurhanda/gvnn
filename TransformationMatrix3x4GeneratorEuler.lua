local TransformationMatrix3x4Euler, parent = torch.class('nn.TransformationMatrix3x4Euler', 'nn.Module')

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

function TransformationMatrix3x4Euler:__init(useRotation, useScale, useTranslation)
  parent.__init(self)

  -- if no specific transformation, use fully parametrized version
  self.fullMode = not(useRotation or useScale or useTranslation)

  if not self.fullMode then
    self.useRotation = useRotation
    self.useScale = useScale
    self.useTranslation = useTranslation
  end
end

function TransformationMatrix3x4Euler:check(input)
  if self.fullMode then
    assert(input:size(2)==7, 'Expected 7 parameters, got ' .. input:size(2))
  else
    local numberParameters = 0
    if self.useRotation then
      numberParameters = numberParameters + 3
    end
    if self.useScale then
      numberParameters = numberParameters + 1
    end
    if self.useTranslation then
      numberParameters = numberParameters + 3
    end
    assert(input:size(2)==numberParameters, 'Expected '..numberParameters..
                                            ' parameters, got ' .. input:size(2))
  end
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


local function EulerRot(batchSize,cos_angle, sin_angle, which_axis)

      local R_phi  = torch.Tensor(batchSize,3,3):zero()

      if which_axis == 1 then
	
      R_phi:select(2,3):select(2,3):add(1)

      R_phi:select(2,1):select(2,1):copy(cos_angle)
      R_phi:select(2,1):select(2,2):copy(sin_angle)

      R_phi:select(2,2):select(2,1):copy(-sin_angle)
      R_phi:select(2,2):select(2,2):copy(cos_angle)


     elseif which_axis == 2 then
      
      R_phi:select(2,2):select(2,2):add(1)

      R_phi:select(2,1):select(2,1):copy(cos_angle)
      R_phi:select(2,1):select(2,3):copy(-sin_angle)

      R_phi:select(2,3):select(2,1):copy(sin_angle)
      R_phi:select(2,3):select(2,3):copy(cos_angle)

     elseif which_axis == 3 then

      R_phi:select(2,1):select(2,1):add(1)

      R_phi:select(2,2):select(2,2):copy(cos_angle)
      R_phi:select(2,2):select(2,3):copy(sin_angle)

      R_phi:select(2,3):select(2,2):copy(-sin_angle)
      R_phi:select(2,3):select(2,3):copy(cos_angle)

     end
   
      return R_phi	
end



local function dR_by_dangle(transparams, RotMats, which_angle)


      local phi = transparams:select(2,1)
      local theta = transparams:select(2,2)
      local psi = transparams:select(2,3)

      local cos_phi = torch.cos(phi) 	
      local sin_phi = torch.sin(phi) 	

      local cos_theta = torch.cos(theta) 	
      local sin_theta = torch.sin(theta) 	

      local cos_psi = torch.cos(psi) 	
      local sin_psi = torch.sin(psi) 	

      if which_angle == 1 then
	
	local Rdash_phi = torch.Tensor(RotMats:size()):zero()
        Rdash_phi:select(2,1):select(2,1):copy(-sin_phi)	
        Rdash_phi:select(2,1):select(2,2):copy(cos_phi)	
        Rdash_phi:select(2,2):select(2,1):copy(-cos_phi)	
        Rdash_phi:select(2,2):select(2,2):copy(-sin_phi)	
	
        local R_theta = EulerRot(RotMats:size(1),cos_theta,sin_theta,2)
        local R_psi = EulerRot(RotMats:size(1),cos_psi,sin_psi,3)

	return torch.bmm(torch.bmm(Rdash_phi,R_theta),R_psi)
	
      elseif which_angle == 2 then 	

	local Rdash_theta = torch.Tensor(RotMats:size()):zero()
      	Rdash_theta:select(2,1):select(2,1):copy(-sin_theta)
      	Rdash_theta:select(2,1):select(2,3):copy(-cos_theta)

        Rdash_theta:select(2,3):select(2,1):copy(cos_theta)
        Rdash_theta:select(2,3):select(2,3):copy(-sin_theta)
	
	return torch.bmm(torch.bmm(EulerRot(RotMats:size(1),cos_phi,sin_phi,1),Rdash_theta),EulerRot(RotMats:size(1),cos_psi,sin_psi,3))
	
      elseif which_angle == 3 then

	local Rdash_psi = torch.Tensor(RotMats:size()):zero()

      	Rdash_psi:select(2,2):select(2,2):copy(-sin_psi)
        Rdash_psi:select(2,2):select(2,3):copy(cos_psi)

        Rdash_psi:select(2,3):select(2,2):copy(-cos_psi)
        Rdash_psi:select(2,3):select(2,3):copy(-sin_psi)

	return torch.bmm(torch.bmm(EulerRot(RotMats:size(1),cos_phi,sin_phi,1),EulerRot(RotMats:size(1),cos_theta,sin_theta,2)),Rdash_psi)

      end
end





function TransformationMatrix3x4Euler:updateOutput(_tranformParams)
  local transformParams
  if _tranformParams:nDimension()==1 then
    transformParams = addOuterDim(_tranformParams)
  else
    transformParams = _tranformParams
  end

  self:check(transformParams)
  local batchSize = transformParams:size(1)

  if self.fullMode then
    self.output = transformParams:view(batchSize, 3, 4)
  else
    local completeTransformation = torch.zeros(batchSize,4,4):typeAs(transformParams)
    completeTransformation:select(3,1):select(2,1):add(1)
    completeTransformation:select(3,2):select(2,2):add(1)
    completeTransformation:select(3,3):select(2,3):add(1)
    completeTransformation:select(3,4):select(2,4):add(1)
    local transformationBuffer = torch.Tensor(batchSize,4,4):typeAs(transformParams)

    local paramIndex = 1
    if self.useRotation then
      --local alphas = transformParams:select(2, paramIndex)

      local phi   = transformParams:select(2,paramIndex)	
      local theta = transformParams:select(2,paramIndex+1)	
      local psi   = transformParams:select(2,paramIndex+2)

      paramIndex = paramIndex + 3


      local cos_phi = torch.cos(phi)   	
      local sin_phi = torch.sin(phi)   	
 
      local R_phi   = EulerRot(batchSize,cos_phi,sin_phi,1)
	
      local cos_theta = torch.cos(theta)
      local sin_theta = torch.sin(theta)

      local R_theta  = EulerRot(batchSize,cos_theta,sin_theta,2)

      local cos_psi = torch.cos(psi)
      local sin_psi = torch.sin(psi)

      local R_psi  = EulerRot(batchSize,cos_psi,sin_psi,3)

	
      --local R_phi_times_R_theta_times_R_psi = torch.bmm(torch.bmm(R_phi,R_theta),R_psi)	
       local R_phi_times_R_theta_times_R_psi = torch.bmm(torch.bmm(R_phi,R_theta),R_psi)	

       completeTransformation:sub(1,batchSize,1,3,1,3):copy(R_phi_times_R_theta_times_R_psi)
	
    end
    self.rotationOutput = completeTransformation:narrow(2,1,3):narrow(3,1,3):clone()

    if self.useScale then
    --  local scaleFactors = transformParams:select(2,paramIndex)
      paramIndex = paramIndex + 1

      transformationBuffer:zero()
      transformationBuffer:select(3,1):select(2,1):copy(scaleFactors)
      transformationBuffer:select(3,2):select(2,2):copy(scaleFactors)
      transformationBuffer:select(3,3):select(2,3):add(1)

      completeTransformation = torch.bmm(completeTransformation, transformationBuffer)
    end

    self.scaleOutput = completeTransformation:narrow(2,1,3):narrow(3,1,3):clone()

--    print ( self.scaleOutput ) 

    if self.useTranslation then
      local txs = transformParams:select(2,paramIndex)
      local tys = transformParams:select(2,paramIndex+1)
      local tzs = transformParams:select(2,paramIndex+2)

      transformationBuffer:zero()
      transformationBuffer:select(3,1):select(2,1):add(1)
      transformationBuffer:select(3,2):select(2,2):add(1)
      transformationBuffer:select(3,3):select(2,3):add(1)
      transformationBuffer:select(3,4):select(2,4):add(1)
      
      transformationBuffer:select(3,4):select(2,1):copy(txs)
      transformationBuffer:select(3,4):select(2,2):copy(tys)
      transformationBuffer:select(3,4):select(2,3):copy(tzs)

--      print (transformationBuffer)

      completeTransformation = torch.bmm(completeTransformation, transformationBuffer)

--      print (completeTransformation)
    end

    self.output=completeTransformation:narrow(2,1,3)
  end

  if _tranformParams:nDimension()==1 then
    self.output = self.output:select(1,1)
  end
  return self.output
end


function TransformationMatrix3x4Euler:updateGradInput(_tranformParams, _gradParams)

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

    if self.useTranslation then

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

    if self.useScale then

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

    if self.useRotation then

      --local rParams = transformParams:select(2,paramIndex)

      local rotationDerivative = torch.zeros(batchSize, 3, 3):typeAs(transformParams)

      local gradInputRotationParams = self.gradInput:narrow(2,1,1)
      
      --torch.sin(rotationDerivative:select(3,1):select(2,1),-rParams)
      --torch.sin(rotationDerivative:select(3,2):select(2,2),-rParams)
      --torch.cos(rotationDerivative:select(3,1):select(2,2),rParams)
      --torch.cos(rotationDerivative:select(3,2):select(2,1),rParams):mul(-1)

      rotationDerivative = dR_by_dangle(transformParams,self.rotationOutput,1)	

      local selectedGradParams = gradParams:narrow(2,1,3):narrow(3,1,3)
      gradInputRotationParams:copy(torch.cmul(rotationDerivative,selectedGradParams):sum(2):sum(3))
      
      rotationDerivative = dR_by_dangle(transformParams,self.rotationOutput,2)	

      --local selectedGradParams = gradParams:narrow(2,1,3):narrow(3,1,3)
      gradInputRotationParams = self.gradInput:narrow(2,2,1)

      gradInputRotationParams:copy(torch.cmul(rotationDerivative,selectedGradParams):sum(2):sum(3))
      
      rotationDerivative = dR_by_dangle(transformParams,self.rotationOutput,3)
	
      --local selectedGradParams = gradParams:narrow(2,1,3):narrow(3,1,3)
      gradInputRotationParams = self.gradInput:narrow(2,3,1)
      gradInputRotationParams:copy(torch.cmul(rotationDerivative,selectedGradParams):sum(2):sum(3))

    end
  end

  if _tranformParams:nDimension()==1 then
    self.gradInput = self.gradInput:select(1,1)
  end
  return self.gradInput
end


