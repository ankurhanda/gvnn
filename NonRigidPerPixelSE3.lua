local NonRigidPerPixelSE3, parent = torch.class('nn.NonRigidPerPixelSE3', 'nn.Module')

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

function NonRigidPerPixelSE3:__init()
  parent.__init(self)

   self.threshold = 1e-12

end

local function dR_by_dvi(transparams, RotMats, which_vi, threshold)
	

      local omega_x = transparams:select(4,1)
      local omega_y = transparams:select(4,2)
      local omega_z = transparams:select(4,3)

      local omega_skew = torch.Tensor(RotMats:size()):typeAs(transparams)
    
      omega_skew:zero()
      omega_skew:select(5,1):select(4,2):copy(omega_z)
      omega_skew:select(5,1):select(4,3):copy(-omega_y)

      omega_skew:select(5,2):select(4,1):copy(-omega_z)
      omega_skew:select(5,2):select(4,3):copy(omega_x)

      omega_skew:select(5,3):select(4,1):copy(omega_y)
      omega_skew:select(5,3):select(4,2):copy(-omega_x)

      --print('omega_skew..') 
      --print(omega_skew)

      --- omega_skew_ge is initialised to omega_skew 
      local omega_skew_ge = torch.Tensor(RotMats:size()):copy(omega_skew):typeAs(transparams)
	
      --- this cannot go wrong...
      local Id_minus_R_ei = torch.Tensor(RotMats:size(1),RotMats:size(2),RotMats:size(3), 3,1):zero():typeAs(transparams)      	
      Id_minus_R_ei:select(4,which_vi):add(1)	 

      local I = torch.Tensor(RotMats:size(1), RotMats:size(2), RotMats:size(3), RotMats:size(4), RotMats:size(5)):zero():typeAs(transparams)
      I:select(5,1):select(4,1):add(1)
      I:select(5,2):select(4,2):add(1)
      I:select(5,3):select(4,3):add(1)
	
      local I_minus_RotMats   = torch.add(I, -RotMats)	

      local Id_minus_R_ei_new = torch.Tensor(RotMats:size(1),RotMats:size(2),RotMats:size(3), 3,1):zero():typeAs(transparams)      

      Id_minus_R_ei_new:select(4,1):copy(torch.cmul(I_minus_RotMats:select(4,1),Id_minus_R_ei):sum(4))
      Id_minus_R_ei_new:select(4,2):copy(torch.cmul(I_minus_RotMats:select(4,2),Id_minus_R_ei):sum(4))
      Id_minus_R_ei_new:select(4,3):copy(torch.cmul(I_minus_RotMats:select(4,3),Id_minus_R_ei):sum(4))
      		    

      Id_minus_R_ei = Id_minus_R_ei_new

      --print('Id_minus_R_ei..')
      --print(Id_minus_R_ei)

      --- cross product 
      local v_cross_Id_minus_R_ei = torch.Tensor(Id_minus_R_ei:size()):typeAs(transparams)

      v_cross_Id_minus_R_ei:select(4,1):copy(torch.cmul(omega_skew:select(4,1),Id_minus_R_ei):sum(4))
      v_cross_Id_minus_R_ei:select(4,2):copy(torch.cmul(omega_skew:select(4,2),Id_minus_R_ei):sum(4))
      v_cross_Id_minus_R_ei:select(4,3):copy(torch.cmul(omega_skew:select(4,3),Id_minus_R_ei):sum(4))

      local cross_x = v_cross_Id_minus_R_ei:select(4,1)	
      local cross_y = v_cross_Id_minus_R_ei:select(4,2)	
      local cross_z = v_cross_Id_minus_R_ei:select(4,3)	

      local vcross = torch.Tensor(RotMats:size()):typeAs(transparams)
	
      vcross:zero()
      vcross:select(5,1):select(4,2):copy(cross_z)
      vcross:select(5,1):select(4,3):copy(-cross_y)

      vcross:select(5,2):select(4,1):copy(-cross_z)
      vcross:select(5,2):select(4,3):copy(cross_x)

      vcross:select(5,3):select(4,1):copy(cross_y)
      vcross:select(5,3):select(4,2):copy(-cross_x)

      
      --print('vcross..')
      --print(vcross)

    
      local omega_mag = torch.pow(omega_x,2) + torch.pow(omega_y,2) + torch.pow(omega_z,2)
      local v_i = transparams:select(4,which_vi)

      --print('v_i..')
      --print(v_i)

      omega_skew:select(5,1):select(4,1):cmul(v_i)
      omega_skew:select(5,1):select(4,2):cmul(v_i)
      omega_skew:select(5,1):select(4,3):cmul(v_i)

      omega_skew:select(5,2):select(4,1):cmul(v_i)
      omega_skew:select(5,2):select(4,2):cmul(v_i)
      omega_skew:select(5,2):select(4,3):cmul(v_i)

      omega_skew:select(5,3):select(4,1):cmul(v_i)
      omega_skew:select(5,3):select(4,2):cmul(v_i)
      omega_skew:select(5,3):select(4,3):cmul(v_i)

      omega_skew = omega_skew + vcross	

      --print('omega_skew_before_division..')
      --print(omega_skew)

      --print('omega_mag..')
      --print(omega_mag)

      omega_skew:select(5,1):select(4,1):cdiv(omega_mag)
      omega_skew:select(5,1):select(4,2):cdiv(omega_mag)
      omega_skew:select(5,1):select(4,3):cdiv(omega_mag)

      omega_skew:select(5,2):select(4,1):cdiv(omega_mag)
      omega_skew:select(5,2):select(4,2):cdiv(omega_mag)
      omega_skew:select(5,2):select(4,3):cdiv(omega_mag)

      omega_skew:select(5,3):select(4,1):cdiv(omega_mag)
      omega_skew:select(5,3):select(4,2):cdiv(omega_mag)
      omega_skew:select(5,3):select(4,3):cdiv(omega_mag)

      --print('omega_skew_new..')
      --print(omega_skew)

      --- do multiplication of rotation matrices with the omega skew
    
      local B = RotMats:size(1)
      local H = RotMats:size(2)
      local W = RotMats:size(3)

      local BHW = B*H*W

      BHW33Mat = torch.bmm(omega_skew:view(BHW,3,3), RotMats:view(BHW, 3,3))

      return BHW33Mat:view(B,H,W,3,3)

      --[[ 
		from http://arxiv.org/pdf/1312.0788.pdf
      --]]

	
      --- v_i [v]x + [v x (Id - R)e_i]x 
	  ----------------------------  R 
      --- 	 ||v||^{2}

end


function NonRigidPerPixelSE3:updateOutput(_tranformParams)

      local transformParams
  
      transformParams = _tranformParams

      --print('--transform params')
      --print(transformParams)
        

     --transformParams[1][1][1][1] = 0.1090
     --transformParams[1][1][1][2] = 0.8495 
     --transformParams[1][1][1][3] = 0.9774

      local batchSize = transformParams:size(1)
      local height    = transformParams:size(2)
      local width     = transformParams:size(3)

      local completeTransformation = torch.zeros(batchSize,height, width, 3,3):typeAs(transformParams):zero()

      completeTransformation:select(5,1):select(4,1):add(1)
      completeTransformation:select(5,2):select(4,2):add(1)
      completeTransformation:select(5,3):select(4,3):add(1)

      local omega_x = transformParams:select(4,1)	
      local omega_y = transformParams:select(4,2)	
      local omega_z = transformParams:select(4,3)

      local omega_skew = torch.Tensor(batchSize,height, width, 3,3):typeAs(transformParams)

      omega_skew:zero()
      omega_skew:select(5,1):select(4,2):copy(omega_z)	
      omega_skew:select(5,1):select(4,3):copy(-omega_y)	
      
      omega_skew:select(5,2):select(4,1):copy(-omega_z)	
      omega_skew:select(5,2):select(4,3):copy(omega_x)	
      
      omega_skew:select(5,3):select(4,1):copy(omega_y)	
      omega_skew:select(5,3):select(4,2):copy(-omega_x)
	

      omega_skew_sqr = torch.Tensor(batchSize,height,width, 3, 3):zero()

      omega_skew_sqr:select(5,1):select(4,1):copy(-torch.pow(omega_z,2)-torch.pow(omega_y,2))
      omega_skew_sqr:select(5,1):select(4,2):copy(torch.cmul(omega_x,omega_y))
      omega_skew_sqr:select(5,1):select(4,3):copy(torch.cmul(omega_x,omega_z))

      omega_skew_sqr:select(5,2):select(4,1):copy(torch.cmul(omega_y,omega_x))
      omega_skew_sqr:select(5,2):select(4,2):copy(-torch.pow(omega_z,2) -torch.pow(omega_x,2))
      omega_skew_sqr:select(5,2):select(4,3):copy(torch.cmul(omega_y,omega_z))

      omega_skew_sqr:select(5,3):select(4,1):copy(torch.cmul(omega_x,omega_z))
      omega_skew_sqr:select(5,3):select(4,2):copy(torch.cmul(omega_y,omega_z))
      omega_skew_sqr:select(5,3):select(4,3):copy(-torch.pow(omega_y,2)-torch.pow(omega_x,2))

      local theta_sqr = torch.pow(omega_x,2) + torch.pow(omega_y,2) + torch.pow(omega_z,2)	
      local theta     = torch.pow(theta_sqr,0.5)	

      local sin_theta           = torch.sin(theta)
      local sin_theta_div_theta = torch.cdiv(sin_theta,theta)

      local one_minus_cos_theta         = torch.ones(theta:size()):typeAs(transformParams) - torch.cos(theta)
      local one_minus_cos_div_theta_sqr = torch.cdiv(one_minus_cos_theta,theta_sqr)

      local sin_theta_div_theta_tensor         = torch.ones(omega_skew:size()):typeAs(transformParams)
      local one_minus_cos_div_theta_sqr_tensor = torch.ones(omega_skew:size()):typeAs(transformParams)

      sin_theta_div_theta_tensor:select(5,1):select(4,1):copy(sin_theta_div_theta)
      sin_theta_div_theta_tensor:select(5,1):select(4,2):copy(sin_theta_div_theta)
      sin_theta_div_theta_tensor:select(5,1):select(4,3):copy(sin_theta_div_theta)

      sin_theta_div_theta_tensor:select(5,2):select(4,1):copy(sin_theta_div_theta)
      sin_theta_div_theta_tensor:select(5,2):select(4,2):copy(sin_theta_div_theta)
      sin_theta_div_theta_tensor:select(5,2):select(4,3):copy(sin_theta_div_theta)
      
      sin_theta_div_theta_tensor:select(5,3):select(4,1):copy(sin_theta_div_theta)
      sin_theta_div_theta_tensor:select(5,3):select(4,2):copy(sin_theta_div_theta)
      sin_theta_div_theta_tensor:select(5,3):select(4,3):copy(sin_theta_div_theta)
     
    
      one_minus_cos_div_theta_sqr_tensor:select(5,1):select(4,1):copy(one_minus_cos_div_theta_sqr)
      one_minus_cos_div_theta_sqr_tensor:select(5,1):select(4,2):copy(one_minus_cos_div_theta_sqr)
      one_minus_cos_div_theta_sqr_tensor:select(5,1):select(4,3):copy(one_minus_cos_div_theta_sqr)

      one_minus_cos_div_theta_sqr_tensor:select(5,2):select(4,1):copy(one_minus_cos_div_theta_sqr)
      one_minus_cos_div_theta_sqr_tensor:select(5,2):select(4,2):copy(one_minus_cos_div_theta_sqr)
      one_minus_cos_div_theta_sqr_tensor:select(5,2):select(4,3):copy(one_minus_cos_div_theta_sqr)

      one_minus_cos_div_theta_sqr_tensor:select(5,3):select(4,1):copy(one_minus_cos_div_theta_sqr)
      one_minus_cos_div_theta_sqr_tensor:select(5,3):select(4,2):copy(one_minus_cos_div_theta_sqr)
      one_minus_cos_div_theta_sqr_tensor:select(5,3):select(4,3):copy(one_minus_cos_div_theta_sqr)
      
      --- need to add boundary conditions i.e. when the size of the rot vector is very small ~ = 0      	

      completeTransformation = completeTransformation + torch.cmul(sin_theta_div_theta_tensor,omega_skew) + torch.cmul(one_minus_cos_div_theta_sqr_tensor, omega_skew_sqr)
    
      self.output = torch.zeros(batchSize,height, width, 3,4):typeAs(transformParams):zero()

      self.rotationOutput = completeTransformation

      self.output:narrow(5,1,3):narrow(4,1,3):copy(completeTransformation)
      self.output:select(5,4):select(4,1):copy(transformParams:narrow(4,4,1))
      self.output:select(5,4):select(4,2):copy(transformParams:narrow(4,5,1))
      self.output:select(5,4):select(4,3):copy(transformParams:narrow(4,6,1))
--      self.output:select(5,4):select(4,4):fill(1)

      --print('self.output')
      --print(self.output)	

      --[[print('dR_by_dv1..')	
      print(dR_by_dvi(transformParams,self.rotationOutput,1,self.threshold))
      
      print('dR_by_dv2..')	
      print(dR_by_dvi(transformParams,self.rotationOutput,2,self.threshold))
      
      print('dR_by_dv3..')	
      print(dR_by_dvi(transformParams,self.rotationOutput,3,self.threshold))]]--

  return self.output
end


function NonRigidPerPixelSE3:updateGradInput(_tranformParams, _gradParams)

      local transformParams, gradParams

      transformParams = _tranformParams
      gradParams = _gradParams:clone()

      local batchSize  = transformParams:size(1)
      local height     = transformParams:size(2)
      local width      = transformParams:size(2)

      self.gradInput:resizeAs(transformParams)

      local rotationDerivative = torch.zeros(batchSize, height, width, 3, 3):typeAs(transformParams)

      local gradInputRotationParams = self.gradInput:narrow(4,1,1)
      rotationDerivative = dR_by_dvi(transformParams,self.rotationOutput,1, self.threshold)	

      local selectedGradParams = gradParams:narrow(5,1,3):narrow(4,1,3)

      gradInputRotationParams:copy(torch.cmul(rotationDerivative,selectedGradParams):sum(4):sum(5))      
      rotationDerivative       = dR_by_dvi(transformParams,self.rotationOutput,2, self.threshold)	

      gradInputRotationParams  = self.gradInput:narrow(4,2,1)

      gradInputRotationParams:copy(torch.cmul(rotationDerivative,selectedGradParams):sum(4):sum(5))      
      rotationDerivative       = dR_by_dvi(transformParams,self.rotationOutput,3, self.threshold)
	
      gradInputRotationParams  = self.gradInput:narrow(4,3,1)
      gradInputRotationParams:copy(torch.cmul(rotationDerivative,selectedGradParams):sum(4):sum(5))

      --[[print('size of gradParams ')
      print(gradParams:size())

      print('size of self.gradInput = ')
      print(self.gradInput)]]--
    
      self.gradInput:select(4,4,1):copy(gradParams:select(5,4):select(4,1)) 
      self.gradInput:select(4,5,1):copy(gradParams:select(5,4):select(4,2))   
      self.gradInput:select(4,6,1):copy(gradParams:select(5,4):select(4,3))   

      return self.gradInput

end
