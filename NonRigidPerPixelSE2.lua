local NonRigidSE2, parent = torch.class('nn.NonRigidSE2BHWD', 'nn.Module')

--[[
   AffineGridGeneratorBHWD(height, width) :
   AffineGridGeneratorBHWD:updateOutput(transformMatrix)
   AffineGridGeneratorBHWD:updateGradInput(transformMatrix, gradGrids)

   AffineGridGeneratorBHWD will take 2x3 an affine image transform matrix (homogeneous 
   coordinates) as input, and output a grid, in normalized coordinates* that, once used
   with the Bilinear Sampler, will result in an affine transform.

   AffineGridGenerator 
   - takes (B,2,3)-shaped transform matrices as input (B=batch).
   - outputs a grid in BHWD layout, that can be used directly with BilinearSamplerBHWD
   - initialization of the previous layer should biased towards the identity transform :
      | 1  0  0 |
      | 0  1  0 |

   *: normalized coordinates [-1,1] correspond to the boundaries of the input image. 
]]

function NonRigidSE2:__init(height, width)
   parent.__init(self)
   assert(height > 1)
   assert(width > 1)
   self.height = height
   self.width = width
   
   self.baseGrid = torch.Tensor(height, width, 3)
   for i=1,self.height do
      self.baseGrid:select(3,2):select(1,i):fill(-1 + (i-1)/(self.height-1) * 2)
   end
   for j=1,self.width do
      self.baseGrid:select(3,1):select(2,j):fill(-1 + (j-1)/(self.width-1) * 2)
   end
   self.baseGrid:select(3,3):fill(1)
   self.batchGrid = torch.Tensor(1, height, width, 3):copy(self.baseGrid)
end

function NonRigidSE2:updateOutput(_PerPixelAffineMatrixParams)

   local PerPixelAffineMatrixParams = _PerPixelAffineMatrixParams

   assert(PerPixelAffineMatrixParams:nDimension()==4
	  and PerPixelAffineMatrixParams:size(2)==self.height
          and PerPixelAffineMatrixParams:size(3)==self.width
          and PerPixelAffineMatrixParams:size(4)==3
          , 'please input affine per-pixel transformations (bxhxwx6)')

   local batchsize = PerPixelAffineMatrixParams:size(1)
   
   if self.batchGrid:size(1) ~= batchsize then

      self.batchGrid:resize(batchsize, self.height, self.width, 3)

      for i=1,batchsize do
         self.batchGrid:select(1,i):copy(self.baseGrid)
      end

   end

   self.output:resize(batchsize, self.height, self.width, 2)

   local sin_theta = torch.sin(PerPixelAffineMatrixParams:select(4,1))
   local cos_theta = torch.cos(PerPixelAffineMatrixParams:select(4,1))

   --self.output:select(4,1):copy(torch.cmul(xx,a0) + torch.cmul(yy,a1) + a2)
   self.output:select(4,1):copy(torch.cmul(self.batchGrid:select(4,1),cos_theta) - torch.cmul(self.batchGrid:select(4,2),sin_theta) + PerPixelAffineMatrixParams:select(4,2))
   --self.output:select(4,2):copy(torch.cmul(xx,a3) + torch.cmul(yy,a4) + a5)
   self.output:select(4,2):copy(torch.cmul(self.batchGrid:select(4,1),sin_theta) + torch.cmul(self.batchGrid:select(4,2),cos_theta) + PerPixelAffineMatrixParams:select(4,3))
   
   return self.output

end

function NonRigidSE2:updateGradInput(_PerPixelAffineMatrixParams, _gradGrid)

   local batchsize = _PerPixelAffineMatrixParams:size(1)

   self.gradInput:resizeAs(_PerPixelAffineMatrixParams):zero()
   
   local sin_theta = torch.sin(_PerPixelAffineMatrixParams:select(4,1))
   local cos_theta = torch.cos(_PerPixelAffineMatrixParams:select(4,1))

   local Lx_theta = torch.cmul(_gradGrid:select(4,1), -torch.cmul(self.batchGrid:select(4,1),sin_theta) - torch.cmul(self.batchGrid:select(4,2),cos_theta))	
   local Ly_theta = torch.cmul(_gradGrid:select(4,2),  torch.cmul(self.batchGrid:select(4,1),cos_theta) - torch.cmul(self.batchGrid:select(4,2),sin_theta))
	
   self.gradInput:select(4,1):copy(Lx_theta + Ly_theta)
   self.gradInput:select(4,2):copy(_gradGrid:select(4,1))
   self.gradInput:select(4,3):copy(_gradGrid:select(4,2))

   return self.gradInput

end
