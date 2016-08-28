local DenseAffineOpticFlow, parent = torch.class('nn.DenseAffineOpticFlowBHWD', 'nn.Module')

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

function DenseAffineOpticFlow:__init(height, width)
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

function DenseAffineOpticFlow:updateOutput(_PerPixelAffineMatrixParams)

   local PerPixelAffineMatrixParams = _PerPixelAffineMatrixParams

   assert(PerPixelAffineMatrixParams:nDimension()==4
	  and PerPixelAffineMatrixParams:size(2)==self.height
          and PerPixelAffineMatrixParams:size(3)==self.width
          and PerPixelAffineMatrixParams:size(4)==6
          , 'please input affine per-pixel transformations (bxhxwx6)')

   local batchsize = PerPixelAffineMatrixParams:size(1)
   
   if self.batchGrid:size(1) ~= batchsize then

      self.batchGrid:resize(batchsize, self.height, self.width, 3)

      for i=1,batchsize do
         self.batchGrid:select(1,i):copy(self.baseGrid)
      end

   end

   self.output:resize(batchsize, self.height, self.width, 2)

   --[[local a0 = PerPixelAffineMatrixParams:select(4,1)	
   local a1 = PerPixelAffineMatrixParams:select(4,2)	

   local a2 = PerPixelAffineMatrixParams:select(4,3)
   local a3 = PerPixelAffineMatrixParams:select(4,4)

   local a4 = PerPixelAffineMatrixParams:select(4,5)
   local a5 = PerPixelAffineMatrixParams:select(4,6)

   local xx = self.batchGrid:select(4,1)
   local yy = self.batchGrid:select(4,2)]]
 
   local Affx = PerPixelAffineMatrixParams:narrow(4,1,3)
   local Affy = PerPixelAffineMatrixParams:narrow(4,4,3)

   --self.output:select(4,1):copy(torch.cmul(xx,a0) + torch.cmul(yy,a1) + a2)
   self.output:select(4,1):copy(torch.sum(torch.cmul(Affx, self.batchGrid),4))
   --self.output:select(4,2):copy(torch.cmul(xx,a3) + torch.cmul(yy,a4) + a5)
   self.output:select(4,2):copy(torch.sum(torch.cmul(Affy, self.batchGrid),4))
   
   return self.output

end

function DenseAffineOpticFlow:updateGradInput(_PerPixelAffineMatrixParams, _gradGrid)

   local batchsize = _PerPixelAffineMatrixParams:size(1)

   self.gradInput:resizeAs(_PerPixelAffineMatrixParams):zero()

   local Lx_x = torch.cmul(_gradGrid:select(4,1), self.batchGrid:select(4,1))	
   local Lx_y = torch.cmul(_gradGrid:select(4,1), self.batchGrid:select(4,2))
	
   local Ly_x = torch.cmul(_gradGrid:select(4,2), self.batchGrid:select(4,1))	
   local Ly_y = torch.cmul(_gradGrid:select(4,2), self.batchGrid:select(4,2))	

   self.gradInput:select(4,1):copy(Lx_x)
   self.gradInput:select(4,2):copy(Lx_y)
   self.gradInput:select(4,3):copy(_gradGrid:select(4,1))

   self.gradInput:select(4,4):copy(Ly_x)
   self.gradInput:select(4,5):copy(Ly_y)
   self.gradInput:select(4,6):copy(_gradGrid:select(4,2))

   return self.gradInput

end
