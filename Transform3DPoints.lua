local Transform3DPoints, parent = torch.class('nn.Transform3DPoints', 'nn.Module')

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
]]

function Transform3DPoints:__init(height, width, fx, fy, u0, v0)
  parent.__init(self)
   assert(height > 1)
   assert(width > 1)

   self.height = height
   self.width = width
   
   self.baseGrid = torch.Tensor(height, width, 3)
   
   for i=1,self.height do
      self.baseGrid:select(3,1):select(1,i):fill(-1 + (i-1)/(self.height-1) * 2)
   end
   
   for j=1,self.width do
      self.baseGrid:select(3,2):select(2,j):fill(-1 + (j-1)/(self.width-1) * 2)
   end
   
   self.baseGrid:select(3,3):fill(1)
   self.batchGrid = torch.Tensor(1, height, width, 3):copy(self.baseGrid)

   self.fx = fx/(self.width-1)
   self.fy = fy/(self.height-1)

   self.u0 = -1 + (u0-1)/(self.width-1) * 2 
   self.v0 = -1 + (v0-1)/(self.height-1) * 2 

   self.points3D = torch.Tensor(1,height,width,4):fill(1)
 
end

function Transform3DPoints:updateOutput(transformMatrix_and_depths)

   _transformMatrix, depths = unpack(transformMatrix_and_depths)

   local transformMatrix = _transformMatrix

   assert(transformMatrix:nDimension()==3
          and transformMatrix:size(2)==3
          and transformMatrix:size(3)==4
          , 'please input transformation matrix of size (bx3x4)')
 
   local batchsize = transformMatrix:size(1)

   if self.points3D:size(1) ~= batchsize then 
	self.points3D:resize(batchsize,self.height,self.width,4)
	self.points:fill(1)
   end
  
   for b = 1, batchsize do
  
	   local u = baseGrid:select(3,1)
	   local v = baseGrid:select(3,2)
	 
	   local u_times_depth = torch.cmul(u:add(-self.u0),depths[b])	
	   local v_times_depth = torch.cmul(v:add(-self.v0),depths[b])	
	   
           self.points3D[b]:select(3,1):copy(u_times_depth):mul(1/self.fx) 	
	   self.points3D[b]:select(3,2):copy(v_times_depth):mul(1/self.fy) 	
	   self.points3D[b]:select(3,3):copy(depths) 	

   end
     
   local flattenedBatchGrid = self.points3D:view(batchsize, self.width*self.height, 4)
   local flattenedOutput = torch.Tensor(batchsize, self.width*self.height, 3):zero()

   torch.bmm(flattenedOutput, flattenedBatchGrid, transformMatrix:transpose(2,3))

   self.output = flattenedOutput:view(batchsize,self.height,self.width,3)
  
   if _transformMatrix:nDimension()==2 then
      self.output = self.output:select(1,1)
   end
   
   return self.output

end

function Transform3DPoints:updateGradInput(transformMatrix_and_depths, _gradGrid)
   
   _trasformMatrix, depths = unpack(transformMatrix_and_depths)

   local transformMatrix, gradGrid
   if _transformMatrix:nDimension()==2 then
      transformMatrix = addOuterDim(_transformMatrix)
      gradGrid = addOuterDim(_gradGrid)
   else
      transformMatrix = _transformMatrix
      gradGrid = _gradGrid
   end

   local batchsize = transformMatrix:size(1)

   local flattenedGradGrid = gradGrid:view(batchsize, self.width*self.height, 3)

   local points3D = self.points3D:view(batchsize, self.width*self.height, 4)

   self.gradInput:resizeAs(transformMatrix):zero()
   self.gradInput:baddbmm(flattenedGradGrid:transpose(2,3), points3D)

   if _transformMatrix:nDimension()==2 then
      self.gradInput = self.gradInput:select(1,1)
   end

   return {self.gradInput, torch.Tensor(batchsize,self.height,self.width):zero()}

end
