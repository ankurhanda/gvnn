local Transform3DPoints_depth, parent = torch.class('nn.Transform3DPoints_depth', 'nn.Module')

--[[

   PinHoleCameraProjectionBHWD(height, width) :
   PinHoleCameraProjectionBHWD:updateOutput(transformMatrix)
   PinHoleCameraProjectionBHWD:updateGradInput(transformMatrix, gradGrids)

   PinHoleCameraProjectionBHWD will take 2x3 an affine image transform matrix (homogeneous 
   coordinates) as input, and output a grid, in normalized coordinates* that, once used
   with the Bilinear Sampler, will result in an affine transform.

   PinHoleCameraProjection 
   - takes (B,2,3)-shaped transform matrices as input (B=batch).
   - outputs a grid in BHWD layout, that can be used directly with BilinearSamplerBHWD
   - initialization of the previous layer should biased towards the identity transform :
      | 1  0  0 |
      | 0  1  0 |

   *: normalized coordinates [-1,1] correspond to the boundaries of the input image. 

]]--

function Transform3DPoints_depth:__init(height, width, fx, fy, u0, v0)

   parent.__init(self)
  
   assert(height > 1)
   assert(width > 1)

   self.height = height
   self.width  = width
   
   self.fx = fx 
   self.fy = fy 

   self.u0 = u0 
   self.v0 = v0  
 
   self.baseGrid = torch.Tensor(height, width, 3)
   
   for i=1,self.height do
      self.baseGrid:select(3,2):select(1,i):fill( (i-self.v0)/self.fy )
   end
   
   for j=1,self.width do
      self.baseGrid:select(3,1):select(2,j):fill( (j-self.u0)/self.fx )
   end
   
   self.baseGrid:select(3,3):fill(1)
   self.points3D  = torch.Tensor(1, height, width, 3):fill(1)
 
end

function Transform3DPoints_depth:updateOutput(depth)

   depths = depth

   local batchsize = depth:size(1)

   if self.points3D:size(1) ~= batchsize then 
	    
        self.points3D:resize(batchsize,self.height,self.width,3)
	    self.points3D:fill(1)
   end
  
   for b = 1, batchsize do

	   --[[ (u-u0)/fx, (v-v0)/fy, 1 ]]--	  
	   
       local u_minus_u0 = self.baseGrid:select(3,1)
	   local v_minus_v0 = self.baseGrid:select(3,2)

	   local u_times_depth = torch.cmul(u_minus_u0,depths[b])	
	   local v_times_depth = torch.cmul(v_minus_v0,depths[b])	

       self.points3D[b]:select(3,1):copy(u_times_depth) 	
	   self.points3D[b]:select(3,2):copy(v_times_depth)
	   self.points3D[b]:select(3,3):copy(depths[b]) 	

   end
 
   -- 3D points
   self.output = self.points3D
  
   return self.output

end

function Transform3DPoints_depth:updateGradInput(_input, _gradGrid)
   
   local batchsize = _input:size(1)
   local gradGrid  = _gradGrid:clone()

   --- saving the grads with respect to the transformed 3d points
   
   local x1 = gradGrid:select(4,1)	
   local x2 = gradGrid:select(4,2)	
   local x3 = gradGrid:select(4,3)

   --- do d * [x y z] --- 		
   local Rp = torch.Tensor(batchsize, self.height, self.width, 3):zero()

   for b = 1, batchsize do 
    
        Rp[b]:select(3,1):copy(self.baseGrid:select(3,1))
        Rp[b]:select(3,2):copy(self.baseGrid:select(3,2))
        Rp[b]:select(3,3):copy(self.baseGrid:select(3,3))
   
   end

   local y1 = Rp:select(4,1)	
   local y2 = Rp:select(4,2)	
   local y3 = Rp:select(4,3)
   
   self.gradInput = torch.add(torch.add(torch.cmul(x1,y1),torch.cmul(x2,y2)),torch.cmul(x3,y3))   

   return self.gradInput

end
